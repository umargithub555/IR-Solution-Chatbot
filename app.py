# # import os
# # import faiss
# # import numpy as np
# # import streamlit as st
# # from io import BytesIO
# # from PyPDF2 import PdfReader
# # from dotenv import load_dotenv
# # import json
# # import hashlib

# # from langchain_groq import ChatGroq
# # from langchain.prompts import PromptTemplate
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_community.embeddings import HuggingFaceEmbeddings

# # # Load API key from environment
# # load_dotenv()
# # groq_api_key = os.environ.get("GROQ_API_KEY")

# # # Directory to save processed data
# # DATA_DIR = "processed_data"
# # if not os.path.exists(DATA_DIR):
# #     os.makedirs(DATA_DIR)

# # # Hashing function for creating a unique identifier for each PDF
# # def get_pdf_hash(file_bytes):
# #     """Generates a unique hash for the PDF file content."""
# #     return hashlib.sha256(file_bytes).hexdigest()

# # # Function to save processed data to disk
# # def save_processed_data(pdf_hash, index, chunks):
# #     """Saves the FAISS index and chunks to the DATA_DIR."""
# #     index_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
# #     chunks_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
    
# #     faiss.write_index(index, index_file)
# #     with open(chunks_file, 'w') as f:
# #         json.dump(chunks, f)

# # # Function to load processed data from disk
# # def load_processed_data(pdf_hash):
# #     """Loads the FAISS index and chunks from the DATA_DIR if they exist."""
# #     index_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
# #     chunks_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")

# #     if os.path.exists(index_file) and os.path.exists(chunks_file):
# #         st.write("Loading pre-processed data from disk...")
# #         index = faiss.read_index(index_file)
# #         with open(chunks_file, 'r') as f:
# #             chunks = json.load(f)
# #         return index, chunks
# #     return None, None

# # # Function to load and extract text from PDF
# # def load_pdf(uploaded_file_bytes):
# #     pdf_file = BytesIO(uploaded_file_bytes)
# #     reader = PdfReader(pdf_file)
# #     text = "".join([page.extract_text() for page in reader.pages])
# #     return text

# # # Function to process the PDF, split text, and create embeddings
# # def process_pdf(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# #     chunks = text_splitter.split_text(text)

# #     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     embeddings = embedding_model.embed_documents(chunks)
    
# #     embedding_matrix = np.array(embeddings).astype("float32")
    
# #     dim = embedding_matrix.shape[1]
# #     index = faiss.IndexFlatL2(dim)
# #     index.add(embedding_matrix)
    
# #     return index, chunks

# # # Function to retrieve relevant chunks using FAISS
# # def faiss_retriever(question, index, chunks):
# #     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     question_embedding = np.array([embedding_model.embed_query(question)]).astype("float32")
    
# #     _, indices = index.search(question_embedding, k=5)
# #     return [chunks[i] for i in indices[0]]

# # # Function to chat with PDF using Groq API
# # def chat_with_pdf(question, index, chunks, model):
# #     relevant_chunks = faiss_retriever(question, index, chunks)
# #     context = "\n\n".join(relevant_chunks)

# #     prompt_template = PromptTemplate.from_template("""
# #     You are a helpful assistant for question answering from documents.
# #     Only answer using the information provided in the context.
# #     If the answer is not explicitly contained in the context, respond with:
# #     "I could not find the answer in the provided document."

# #     Context:
# #     {context}

# #     Question: {question}
# #     """)

# #     prompt = prompt_template.format(context=context, question=question)
    
# #     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model)
# #     response = chat_groq.invoke(prompt)

# #     return response.content if hasattr(response, "content") else str(response)

# # # Streamlit UI
# # st.title("PDF Chatbot with Groq & FAISS")

# # uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# # if uploaded_file:
# #     uploaded_file_bytes = uploaded_file.read()
# #     pdf_hash = get_pdf_hash(uploaded_file_bytes)
    
# #     index, chunks = load_processed_data(pdf_hash)

# #     if index is None or chunks is None:
# #         st.write("Processing new PDF...")
# #         # Process the new PDF
# #         text = load_pdf(uploaded_file_bytes)
# #         index, chunks = process_pdf(text)
        
# #         # Save the processed data for future use
# #         save_processed_data(pdf_hash, index, chunks)
# #         st.write("PDF processed and saved! You can now ask questions.")

# #     else:
# #         st.write("PDF data loaded from disk. You can now ask questions.")

# #     # Main interaction logic
# #     model = st.sidebar.selectbox("Select Groq Model", ['qwen/qwen3-32b', 'gemma2-9b-it','llama-3.1-8b-instant'])
# #     question = st.text_input("Ask a question about the PDF:")

# #     if question:
# #         st.write("Fetching answer...")
# #         response = chat_with_pdf(question, index, chunks, model)
# #         st.write("Answer:", response)

# # else:
# #     st.write("Upload a PDF to begin!")


# import os
# import faiss
# import numpy as np
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import json
# import hashlib
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # ------------------ CONFIG ------------------ #
# PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"  # Path to your PDF
# DATA_DIR = "processed_data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Load API key
# load_dotenv()
# groq_api_key = os.environ.get("GROQ_API_KEY")


# # ------------------ HELPERS ------------------ #
# def get_pdf_hash(file_bytes):
#     return hashlib.sha256(file_bytes).hexdigest()

# def save_processed_data(pdf_hash, index, chunks):
#     faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
#     with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
#         json.dump(chunks, f)

# def load_processed_data(pdf_hash):
#     idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
#     chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
#     if os.path.exists(idx_file) and os.path.exists(chk_file):
#         index = faiss.read_index(idx_file)
#         with open(chk_file, 'r') as f:
#             chunks = json.load(f)
#         return index, chunks
#     return None, None

# def load_pdf(file_bytes):
#     reader = PdfReader(BytesIO(file_bytes))
#     return "".join([page.extract_text() for page in reader.pages])

# def process_pdf(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedding_model.embed_documents(chunks)
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings).astype("float32"))
#     return index, chunks

# def faiss_retriever(question, index, chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
#     _, idxs = index.search(q_emb, k=5)
#     return [chunks[i] for i in idxs[0]]

# def chat_with_pdf(question, index, chunks, model):
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     context = "\n\n".join(relevant_chunks)
#     prompt = PromptTemplate.from_template("""
#     You are a helpful assistant for answering from documents.
#     Only use the provided context. If answer not in context, reply:
#     "I could not find the answer in the provided document."

#     Context:
#     {context}

#     Question: {question}
#     """).format(context=context, question=question)

#     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model)
#     response = chat_groq.invoke(prompt)
#     return getattr(response, "content", str(response))


# # ------------------ APP ------------------ #
# st.title("Team WorkFlow & Responsibilities")

# # Load PDF once
# with open(PDF_PATH, "rb") as f:
#     file_bytes = f.read()

# pdf_hash = get_pdf_hash(file_bytes)
# index, chunks = load_processed_data(pdf_hash)

# if index is None or chunks is None:
#     st.write("Processing PDF in the background...")
#     text = load_pdf(file_bytes)
#     index, chunks = process_pdf(text)
#     save_processed_data(pdf_hash, index, chunks)
#     st.success("PDF processed! You can start chatting.")

# model = st.sidebar.selectbox("Groq Model", [ 'gemma2-9b-it'])
# question = st.text_input("💬 Ask something from the document:")

# if question:
#     st.write("Fetching answer...")
#     answer = chat_with_pdf(question, index, chunks, model)
#     st.markdown(f"**Answer:** {answer}")


# working well

# import os
# import faiss
# import numpy as np
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import json
# import hashlib
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # ------------------ CONFIG ------------------ #
# PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"  # Update path
# DATA_DIR = "processed_data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Load API key
# load_dotenv()
# groq_api_key = os.environ.get("GROQ_API_KEY")

# # ------------------ HELPERS ------------------ #
# def get_pdf_hash(file_bytes):
#     return hashlib.sha256(file_bytes).hexdigest()

# def save_processed_data(pdf_hash, index, chunks):
#     faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
#     with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
#         json.dump(chunks, f)

# def load_processed_data(pdf_hash):
#     idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
#     chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
#     if os.path.exists(idx_file) and os.path.exists(chk_file):
#         index = faiss.read_index(idx_file)
#         with open(chk_file, 'r') as f:
#             chunks = json.load(f)
#         return index, chunks
#     return None, None

# def load_pdf(file_bytes):
#     reader = PdfReader(BytesIO(file_bytes))
#     return "".join([page.extract_text() for page in reader.pages])

# def process_pdf(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedding_model.embed_documents(chunks)
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings).astype("float32"))
#     return index, chunks

# def faiss_retriever(question, index, chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
#     _, idxs = index.search(q_emb, k=5)
#     return [chunks[i] for i in idxs[0]]

# def chat_with_pdf(question, index, chunks, model):
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     context = "\n\n".join(relevant_chunks)
#     prompt = PromptTemplate.from_template("""
#     You are a helpful assistant for answering from documents.
#     Only use the provided context. If answer not in context, reply:
#     "I could not find the answer in the provided document."

#     Context:
#     {context}

#     Question: {question}
#     """).format(context=context, question=question)

#     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model)
#     response = chat_groq.invoke(prompt)
#     return getattr(response, "content", str(response))

# # ------------------ APP ------------------ #
# st.title("📄 Team Workflow Document")

# # Load PDF once
# with open(PDF_PATH, "rb") as f:
#     file_bytes = f.read()

# pdf_hash = get_pdf_hash(file_bytes)
# index, chunks = load_processed_data(pdf_hash)

# if index is None or chunks is None:
#     st.write("Processing PDF...")
#     text = load_pdf(file_bytes)
#     index, chunks = process_pdf(text)
#     save_processed_data(pdf_hash, index, chunks)
#     st.success("PDF processed! You can start chatting.")

# # Initialize session state for chat
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Sidebar for model choice
# model = 'gemma2-9b-it'

# # Chat messages display
# chat_container = st.container()
# with chat_container:
#     for msg in st.session_state.messages:
#         role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
#         st.markdown(f"**{role}:** {msg['content']}")

# # Chat input at bottom
# if question := st.chat_input("Type your question..."):
#     st.session_state.messages.append({"role": "user", "content": question})
#     answer = chat_with_pdf(question, index, chunks, model)
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.rerun()




# import os
# import faiss
# import numpy as np
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import json
# import hashlib
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

# # ------------------ CONFIG ------------------ #

# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# # PDF_PATH = "sample_docs/deep learning.pdf"  # Update path
# PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"
# DATA_DIR = "processed_data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Load API key
# load_dotenv()
# groq_api_key = os.environ.get("GROQ_API_KEY")

# # ------------------ HELPERS ------------------ #
# def get_pdf_hash(file_bytes):
#     return hashlib.sha256(file_bytes).hexdigest()

# def save_processed_data(pdf_hash, index, chunks):
#     faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
#     with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
#         json.dump(chunks, f)

# def load_processed_data(pdf_hash):
#     idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
#     chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
#     if os.path.exists(idx_file) and os.path.exists(chk_file):
#         index = faiss.read_index(idx_file)
#         with open(chk_file, 'r') as f:
#             chunks = json.load(f)
#         return index, chunks
#     return None, None

# def load_pdf(file_bytes):
#     reader = PdfReader(BytesIO(file_bytes))
#     return "".join([page.extract_text() for page in reader.pages])

# def process_pdf(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedding_model.embed_documents(chunks)
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings).astype("float32"))
#     return index, chunks

# def faiss_retriever(question, index, chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
#     _, idxs = index.search(q_emb, k=5)
#     return [chunks[i] for i in idxs[0]]

# def chat_with_pdf(question, index, chunks, model, chat_history=None):
#     # Handle greetings and thanks
#     greetings = ["hi", "hello", "hey", "greetings"]
#     thanks = ["thanks", "thank you", "appreciate it"]
    
#     if any(word in question.lower() for word in greetings):
#         return "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
    
#     if any(word in question.lower() for word in thanks):
#         return "You're welcome! Is there anything else I can help you with?"
    
#     # Get relevant context from document
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     context = "\n\n".join(relevant_chunks)
    
#     # Include chat history in context if available
#     history_context = ""
#     if chat_history:
#         history_context = "\nPrevious conversation:\n" + "\n".join(
#             [f"User: {msg['content']}\nAssistant: {msg['response']}" 
#              for msg in chat_history]  # Last 3 exchanges
#         )
    
#     prompt = PromptTemplate.from_template("""
#     You are a helpful assistant for IR Solutions, answering questions about company documents.
#     Use the provided context and any previous conversation history to answer.
#     Be professional but friendly in your responses.
    
#     {history_context}
    
#     Document Context:
#     {context}
    
#     Current Question: {question}
    
#     If the answer isn't in the context, respond:
#     "I couldn't find that information in the document. Would you like me to clarify anything else?"
#     """).format(
#         context=context,
#         question=question,
#         history_context=history_context
#     )

#     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model)
#     response = chat_groq.invoke(prompt)
#     return getattr(response, "content", str(response))

# # ------------------ APP ------------------ #
# st.title("📄 IR Solution ChatBot")

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     # Add welcome message
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": "Hi there! Have questions about IR solutions or need help finding the right service for your practice? I’m here to help—just ask!"
#     })

# # Load PDF once
# with open(PDF_PATH, "rb") as f:
#     file_bytes = f.read()

# pdf_hash = get_pdf_hash(file_bytes)
# index, chunks = load_processed_data(pdf_hash)

# if index is None or chunks is None:
#     with st.spinner("Processing document..."):
#         text = load_pdf(file_bytes)
#         index, chunks = process_pdf(text)
#         save_processed_data(pdf_hash, index, chunks)

# # Model selection (simplified)
# model = 'gemma2-9b-it'

# # Display chat messages
# for message in st.session_state.messages:
#     if message["role"] == "assistant":
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(message["content"])
#     else:
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask about the document..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(prompt)
    
#     # Get chat history for context (excluding welcome message)
#     chat_history = [
#         {"content": m["content"], "response": st.session_state.messages[i+1]["content"]}
#         for i, m in enumerate(st.session_state.messages[:-1])
#         if m["role"] == "user" and i+1 < len(st.session_state.messages)
#     ]
    
#     # Display assistant response
#     with st.chat_message("assistant", avatar="🤖"):
#         with st.spinner("Thinking..."):
#             response = chat_with_pdf(
#                 prompt, 
#                 index, 
#                 chunks, 
#                 model,
#                 chat_history
#             )
#         st.markdown(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Add some custom CSS for better chat alignment
# st.markdown("""
# <style>
#     /* User messages on right */
#     [data-testid="stChatMessage"] {
#         padding: 12px;
#         border-radius: 8px;
#         margin-bottom: 8px;
#     }
#     /* Assistant messages on left */
#     [data-testid="stChatMessage"] div:first-child {
#         justify-content: flex-start;
#     }
#     /* User messages on right */
#     [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
#         justify-content: flex-end;
#     }
# </style>
# """, unsafe_allow_html=True)





# import os
# import faiss
# import numpy as np
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import json
# import hashlib
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

# # ------------------ CONFIG ------------------ #

# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# # PDF_PATH = "sample_docs/deep learning.pdf"  # Update path
# PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"
# DATA_DIR = "processed_data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Load API key
# load_dotenv()
# groq_api_key = os.environ.get("GROQ_API_KEY")

# # ------------------ HELPERS ------------------ #
# def get_pdf_hash(file_bytes):
#     return hashlib.sha256(file_bytes).hexdigest()

# def save_processed_data(pdf_hash, index, chunks):
#     faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
#     with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
#         json.dump(chunks, f)

# def load_processed_data(pdf_hash):
#     idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
#     chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
#     if os.path.exists(idx_file) and os.path.exists(chk_file):
#         index = faiss.read_index(idx_file)
#         with open(chk_file, 'r') as f:
#             chunks = json.load(f)
#         return index, chunks
#     return None, None

# def load_pdf(file_bytes):
#     reader = PdfReader(BytesIO(file_bytes))
#     return "".join([page.extract_text() for page in reader.pages])

# def process_pdf(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedding_model.embed_documents(chunks)
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings).astype("float32"))
#     return index, chunks

# def faiss_retriever(question, index, chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
#     _, idxs = index.search(q_emb, k=5)
#     return [chunks[i] for i in idxs[0]]

# def chat_with_pdf(question, index, chunks, model, chat_history=None):
#     # Handle greetings and thanks with more precise matching
#     question_lower = question.lower().strip()
    
#     # More precise greeting detection - check if question starts with or is exactly these words
#     greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
#     thanks = ["thanks", "thank you", "appreciate it", "thx"]
    
#     # Check if the question is primarily a greeting (starts with greeting or is just a greeting)
#     is_greeting = (
#         any(question_lower.startswith(greeting) for greeting in greetings) or
#         question_lower in greetings or
#         question_lower in ["hi there", "hello there", "hey there"]
#     )
    
#     # Check if the question is primarily a thank you
#     is_thanks = (
#         any(question_lower.startswith(thank) for thank in thanks) or
#         question_lower in thanks or
#         "thank you" in question_lower and len(question_lower.split()) <= 3
#     )
    
#     if is_greeting:
#         return "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
    
#     if is_thanks:
#         return "You're welcome! Is there anything else I can help you with?"
    
#     # Get relevant context from document
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     context = "\n\n".join(relevant_chunks)
    
#     # Include chat history in context if available
#     history_context = ""
#     if chat_history:
#         history_context = "\nPrevious conversation:\n" + "\n".join(
#             [f"User: {msg['content']}\nAssistant: {msg['response']}" 
#              for msg in chat_history]  # Last 3 exchanges
#         )
    
#     prompt = PromptTemplate.from_template("""
#     You are a helpful assistant for IR Solutions, answering questions about company documents.
#     Use the provided context and any previous conversation history to answer.
#     Be professional but friendly in your responses.
    
#     {history_context}
    
#     Document Context:
#     {context}
    
#     Current Question: {question}
    
#     If the answer isn't in the context, respond:
#     "I couldn't find that information in the document. Would you like me to clarify anything else?"
#     """).format(
#         context=context,
#         question=question,
#         history_context=history_context
#     )

#     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model,streaming=True)
#     response = chat_groq.invoke(prompt)
#     return getattr(response, "content", str(response))

# # ------------------ APP ------------------ #
# st.title("📄 IR Solution ChatBot")

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     # Add welcome message
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": "Hi there! Have questions about IR solutions or need help finding the right service for your practice? I'm here to help—just ask!"
#     })

# # Load PDF once
# with open(PDF_PATH, "rb") as f:
#     file_bytes = f.read()

# pdf_hash = get_pdf_hash(file_bytes)
# index, chunks = load_processed_data(pdf_hash)

# if index is None or chunks is None:
#     with st.spinner("Processing document..."):
#         text = load_pdf(file_bytes)
#         index, chunks = process_pdf(text)
#         save_processed_data(pdf_hash, index, chunks)

# # Model selection (simplified)
# model = 'gemma2-9b-it'

# # Display chat messages
# for message in st.session_state.messages:
#     if message["role"] == "assistant":
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(message["content"])
#     else:
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask about the document..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(prompt)
    
#     # Get chat history for context (excluding welcome message)
#     chat_history = [
#         {"content": m["content"], "response": st.session_state.messages[i+1]["content"]}
#         for i, m in enumerate(st.session_state.messages[:-1])
#         if m["role"] == "user" and i+1 < len(st.session_state.messages)
#     ]
    
#     # Display assistant response
#     with st.chat_message("assistant", avatar="🤖"):
#         with st.spinner("Thinking..."):
#             response = chat_with_pdf(
#                 prompt, 
#                 index, 
#                 chunks, 
#                 model,
#                 chat_history
#             )
#         st.markdown(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Add some custom CSS for better chat alignment
# st.markdown("""
# <style>
#     /* User messages on right */
#     [data-testid="stChatMessage"] {
#         padding: 12px;
#         border-radius: 8px;
#         margin-bottom: 8px;
#     }
#     /* Assistant messages on left */
#     [data-testid="stChatMessage"] div:first-child {
#         justify-content: flex-start;
#     }
#     /* User messages on right */
#     [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
#         justify-content: flex-end;
#     }
# </style>
# """, unsafe_allow_html=True)


# import os
# # import faiss
# try:
#     import faiss
# except ModuleNotFoundError:
#     import faiss_cpu as faiss
# import numpy as np
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import json
# import hashlib
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings

# # ------------------ CONFIG ------------------ #

# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# # PDF_PATH = "sample_docs/deep learning.pdf"  # Update path
# PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"
# DATA_DIR = "processed_data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Load API key
# load_dotenv()
# groq_api_key = os.environ.get("GROQ_API_KEY")

# # ------------------ HELPERS ------------------ #
# def get_pdf_hash(file_bytes):
#     """Generates a SHA256 hash for the given PDF file bytes."""
#     return hashlib.sha256(file_bytes).hexdigest()

# def save_processed_data(pdf_hash, index, chunks):
#     """Saves the FAISS index and text chunks to disk."""
#     faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
#     with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
#         json.dump(chunks, f)

# def load_processed_data(pdf_hash):
#     """Loads the FAISS index and text chunks from disk if they exist."""
#     idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
#     chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
#     if os.path.exists(idx_file) and os.path.exists(chk_file):
#         index = faiss.read_index(idx_file)
#         with open(chk_file, 'r') as f:
#             chunks = json.load(f)
#         return index, chunks
#     return None, None

# def load_pdf(file_bytes):
#     """Extracts text from a PDF file."""
#     reader = PdfReader(BytesIO(file_bytes))
#     return "".join([page.extract_text() for page in reader.pages])

# def process_pdf(text):
#     """Splits text into chunks, embeds them, and creates a FAISS index."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedding_model.embed_documents(chunks)
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings).astype("float32"))
#     return index, chunks

# def faiss_retriever(question, index, chunks):
#     """Retrieves the most relevant chunks from the FAISS index based on a question."""
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
#     _, idxs = index.search(q_emb, k=5)
#     return [chunks[i] for i in idxs[0]]

# def chat_with_pdf(question, index, chunks, model, chat_history=None):
#     """
#     Generates a response to a question based on document context and chat history,
#     streaming the response word by word.
#     """
#     question_lower = question.lower().strip()
    
#     # More precise greeting detection - check if question starts with or is exactly these words
#     greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
#     thanks = ["thanks", "thank you", "appreciate it", "thx"]
    
#     # Check if the question is primarily a greeting (starts with greeting or is just a greeting)
#     is_greeting = (
#         any(question_lower.startswith(greeting) for greeting in greetings) or
#         question_lower in greetings or
#         question_lower in ["hi there", "hello there", "hey there"]
#     )
    
#     # Check if the question is primarily a thank you
#     is_thanks = (
#         any(question_lower.startswith(thank) for thank in thanks) or
#         question_lower in thanks or
#         ("thank you" in question_lower and len(question_lower.split()) <= 3)
#     )
    
#     if is_greeting:
#         yield "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
#         return
    
#     if is_thanks:
#         yield "You're welcome! Is there anything else I can help you with?"
#         return
    
#     # Get relevant context from document
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     context = "\n\n".join(relevant_chunks)
    
#     # Include chat history in context if available
#     history_context = ""
#     if chat_history:
#         history_context = "\nPrevious conversation:\n" + "\n".join(
#             [f"User: {msg['content']}\nAssistant: {msg['response']}" 
#              for msg in chat_history[-3:]]
#         )
    
#     prompt = PromptTemplate.from_template("""
#     You are a helpful assistant for IR Solutions, answering questions about company documents.
#     Use the provided context and any previous conversation history to answer.
#     Be professional but friendly in your responses.
    
#     {history_context}
    
#     Document Context:
#     {context}
    
#     Current Question: {question}
    
#     If the answer isn't in the context, respond:
#     "I couldn't find that information in the document. Would you like me to clarify anything else?"
#     """).format(
#         context=context,
#         question=question,
#         history_context=history_context
#     )

#     chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model, streaming=True)
#     stream = chat_groq.stream(prompt)
    
#     # Yield each chunk from the stream
#     for chunk in stream:
#         yield chunk.content

# # ------------------ APP ------------------ #
# st.title("📄 IR Solution ChatBot")

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     # Add welcome message
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": "Hi there! Have questions about IR solutions or need help finding the right service for your practice? I'm here to help—just ask!"
#     })

# # Load PDF once
# with open(PDF_PATH, "rb") as f:
#     file_bytes = f.read()

# pdf_hash = get_pdf_hash(file_bytes)
# index, chunks = load_processed_data(pdf_hash)

# if index is None or chunks is None:
#     with st.spinner("Processing document..."):
#         text = load_pdf(file_bytes)
#         index, chunks = process_pdf(text)
#         save_processed_data(pdf_hash, index, chunks)

# # Model selection (simplified)
# model = 'gemma2-9b-it'

# # Display chat messages
# for message in st.session_state.messages:
#     if message["role"] == "assistant":
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(message["content"])
#     else:
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask about the document..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(prompt)
    
#     # Get chat history for context (excluding welcome message)
#     # This collects user messages and the assistant's previous responses
#     chat_history = [
#         {"content": st.session_state.messages[i]["content"], 
#          "response": st.session_state.messages[i+1]["content"]}
#         for i, m in enumerate(st.session_state.messages[:-1])
#         if m["role"] == "user" and i+1 < len(st.session_state.messages) and st.session_state.messages[i+1]["role"] == "assistant"
#     ]
    
#     # Display assistant response
#     with st.chat_message("assistant", avatar="🤖"):
#         with st.spinner("Thinking"):
#             placeholder = st.empty()
#             full_response = ""
#             for chunk in chat_with_pdf(prompt, index, chunks, model, chat_history):
#                 full_response += chunk
#                 placeholder.markdown(full_response + "▌")  # Use a blinking cursor for better UX

#         # Remove the cursor at the end and add the final response to the history
#         placeholder.markdown(full_response)
        
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# # Add some custom CSS for better chat alignment
# st.markdown("""
# <style>
#     /* User messages on right */
#     [data-testid="stChatMessage"] {
#         padding: 12px;
#         border-radius: 8px;
#         margin-bottom: 8px;
#     }
#     /* Assistant messages on left */
#     [data-testid="stChatMessage"] div:first-child {
#         justify-content: flex-start;
#     }
#     /* User messages on right */
#     [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
#         justify-content: flex-end;
#     }
# </style>
# """, unsafe_allow_html=True)



import os
# import faiss
try:
    import faiss
except ModuleNotFoundError:
    import faiss_cpu as faiss
import numpy as np
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import json
import hashlib
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------ CONFIG ------------------ #

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_WATCHDOG"] = "false"

# PDF_PATH = "sample_docs/deep learning.pdf"  # Update path
PDF_PATH = "sample_docs/Team_Workflow_Responsbilities.pdf"
DATA_DIR = "processed_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load API key
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Configuration for summarization
SUMMARIZATION_THRESHOLD = 5  # Number of messages before triggering summarization
MAX_SUMMARY_LENGTH = 400  # Maximum length of summary




@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder="hf_cache",   # optional: local cache
        model_kwargs={"use_auth_token": os.getenv("HUGGINGFACE_HUB_TOKEN")}
    )

# Instead of re-creating everywhere, call this
embedding_model = get_embedding_model()





# ------------------ HELPERS ------------------ #
def get_pdf_hash(file_bytes):
    """Generates a SHA256 hash for the given PDF file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()

def save_processed_data(pdf_hash, index, chunks):
    """Saves the FAISS index and text chunks to disk."""
    faiss.write_index(index, os.path.join(DATA_DIR, f"{pdf_hash}.faiss"))
    with open(os.path.join(DATA_DIR, f"{pdf_hash}.json"), 'w') as f:
        json.dump(chunks, f)

def load_processed_data(pdf_hash):
    """Loads the FAISS index and text chunks from disk if they exist."""
    idx_file = os.path.join(DATA_DIR, f"{pdf_hash}.faiss")
    chk_file = os.path.join(DATA_DIR, f"{pdf_hash}.json")
    if os.path.exists(idx_file) and os.path.exists(chk_file):
        index = faiss.read_index(idx_file)
        with open(chk_file, 'r') as f:
            chunks = json.load(f)
        return index, chunks
    return None, None

def load_pdf(file_bytes):
    """Extracts text from a PDF file."""
    reader = PdfReader(BytesIO(file_bytes))
    return "".join([page.extract_text() for page in reader.pages])

def process_pdf(text):
    """Splits text into chunks, embeds them, and creates a FAISS index."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = text_splitter.split_text(text)
    # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = embedding_model.embed_documents(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def faiss_retriever(question, index, chunks):
    """Retrieves the most relevant chunks from the FAISS index based on a question."""
    # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
    _, idxs = index.search(q_emb, k=5)
    return [chunks[i] for i in idxs[0]]

def summarize_conversation_history(messages, model):
    """
    Summarizes the conversation history using the LLM.
    Takes a list of messages and returns a concise summary.
    """
    if len(messages) <= 2:  # No need to summarize very short conversations
        return ""
    
    # Filter out system messages and format conversation
    conversation_text = ""
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
    
    if not conversation_text.strip():
        return ""
    
    summarization_prompt = PromptTemplate.from_template("""
    Please provide a concise summary of the following conversation between a user and an AI assistant about IR Solutions documents. 
    Focus on:
    1. Key topics discussed
    2. Important information provided
    3. Questions asked and answered
    4. Any specific details about IR Solutions services or processes
    
    Keep the summary under {max_length} characters and maintain context that would be helpful for continuing the conversation.
    
    Conversation:
    {conversation}
    
    Summary:
    """)
    
    prompt = summarization_prompt.format(
        conversation=conversation_text,
        max_length=MAX_SUMMARY_LENGTH
    )
    
    try:
        chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model)
        response = chat_groq.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error in summarization: {e}")
        return ""

def get_conversation_context(messages, model):
    """
    Manages conversation context using summarization approach.
    Returns either a summary of older messages + recent messages, or just recent messages.
    """
    if len(messages) <= SUMMARIZATION_THRESHOLD:
        # Use recent conversation as-is if below threshold
        recent_messages = [
            {"content": messages[i]["content"], 
             "response": messages[i+1]["content"]}
            for i, m in enumerate(messages[:-1])
            if m["role"] == "user" and i+1 < len(messages) and messages[i+1]["role"] == "assistant"
        ]
        
        if recent_messages:
            return "\nRecent conversation:\n" + "\n".join(
                [f"User: {msg['content']}\nAssistant: {msg['response']}" 
                 for msg in recent_messages[-3:]]
            )
        return ""
    
    else:
        # Use summarization approach for longer conversations
        # Keep last 2-3 exchanges as recent context
        recent_messages = messages[-(SUMMARIZATION_THRESHOLD-2):]
        older_messages = messages[:-(SUMMARIZATION_THRESHOLD-2)]
        
        # Generate summary of older messages
        summary = summarize_conversation_history(older_messages, model)
        
        # Format recent messages
        recent_context = []
        for i, msg in enumerate(recent_messages[:-1]):
            if msg["role"] == "user" and i+1 < len(recent_messages) and recent_messages[i+1]["role"] == "assistant":
                recent_context.append({
                    "content": msg["content"],
                    "response": recent_messages[i+1]["content"]
                })
        
        context_parts = []
        if summary:
            context_parts.append(f"\nConversation summary: {summary}")
        
        if recent_context:
            context_parts.append("\nRecent conversation:\n" + "\n".join(
                [f"User: {msg['content']}\nAssistant: {msg['response']}" 
                 for msg in recent_context]
            ))
        
        return "\n".join(context_parts)



def chat_with_pdf(question, index, chunks, model, messages=None):
    """
    Generates a response to a question based on document context and chat history,
    streaming the response word by word.
    Uses summarization for long conversation histories.
    """
    question_lower = question.lower().strip()
    
    # More precise greeting detection - check if question starts with or is exactly these words
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    thanks = ["thanks", "thank you", "appreciate it", "thx"]
    


    if any(g in question_lower.split() for g in greetings) and len(question_lower.split()) <= 3:
        yield "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
        return

    if any(t in question_lower for t in thanks) and len(question_lower.split()) <= 3:
        yield "You're welcome! Is there anything else I can help you with?"
        return




    # Check if the question is primarily a greeting (starts with greeting or is just a greeting)
    # is_greeting = (
    #     any(question_lower.startswith(greeting) for greeting in greetings) or
    #     question_lower in greetings or
    #     question_lower in ["hi there", "hello there", "hey there"]
    # )
    
    # # Check if the question is primarily a thank you
    # is_thanks = (
    #     any(question_lower.startswith(thank) for thank in thanks) or
    #     question_lower in thanks or
    #     ("thank you" in question_lower and len(question_lower.split()) <= 3)
    # )
    
    # if is_greeting:
    #     yield "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
    #     return
    
    # if is_thanks:
    #     yield "You're welcome! Is there anything else I can help you with?"
    #     return
    
    # Get relevant context from document
    relevant_chunks = faiss_retriever(question, index, chunks)
    # print(relevant_chunks)
    context = "\n\n".join(relevant_chunks)
    
    # Get conversation context using summarization approach
    history_context = ""
    if messages and len(messages) > 1:
        history_context = get_conversation_context(messages, model)
    
    # prompt = PromptTemplate.from_template("""
    # ### Core Identity and Rules
    # You are the **IR Solutions chatbot**. Your **SOLE AND UNCHANGING PURPOSE** is to assist with questions that are **directly related** to the IR Solutions company, its policies, and the provided company documents.

    # **RULE 1 (Out-of-Scope):** If a question is not related to IR Solutions (e.g., a request for legal advice, general knowledge, creative writing, or role-playing), you **MUST IMMEDIATELY and STRICTLY** refuse the request and respond with the following exact message:
    # "I am IR Solutions chatbot and can only help you with company-specific information."

    # **RULE 2  (Response consideration):** Answer all relevant questions using only the provided `Document Context` and `History Context`. Do not use any other external knowledge.
                                                                                
    # **RULE 3 (Source Reference):** You **MUST NOT** mention or refer to the provided document, the context, or any external sources in your responses. Answer as if the information is part of your inherent knowledge.

    # **RULE 4 (Tone):** Be professional, friendly, and helpful.

    # ---
    
    # ### Provided Information
    
    # History Context:
    # {history_context}
    
    # Document Context:
    # {context}
    
    # ---
    
    # ### Employee Query
    # {question}
    
    # ---
    
    # **Based on the rules and provided information, formulate your response.**
    
    # If the answer is not available in the provided context, respond with the following exact message:
    # "I am IR Solutions chatbot and can only help you with company-related information. Would you like me to clarify anything else?"
    # """).format(
    #     context=context,
    #     question=question,
    #     history_context=history_context
    # )


    prompt = PromptTemplate.from_template("""
    ### Core Identity and Rules
    You are the **IR Solutions chatbot**. Your **SOLE AND UNCHANGING PURPOSE** is to assist with questions that are **directly related** to the IR Solutions company, its policies, and the provided company documents currently Team Workflow and Responsibilities.

    **RULE 1 (Out-of-Scope):** If a question is not related to IR Solutions (e.g., a request for legal advice, general knowledge, creative writing, or role-playing), you **MUST IMMEDIATELY and STRICTLY** refuse the request and respond with the following exact message:
    "I am IR Solutions chatbot and can only help you with company-specific information."

    **RULE 2 (Knowledge Use):** Only use the information provided in the internal knowledge below.
    - Use the Previous Conversation or history context when asked about previous interactions or when required.                                       
    - Never mention where the knowledge comes from.
    - Never say "outlined in this document ","the document says," "the context states," or any similar phrasing any where in your response. 
    - Respond as if this knowledge is part of your inherent expertise.
                                                                                
    **RULE 3 (Tone):** Be professional, friendly, and helpful.

    **RULE 4 (Elaboration):** When providing an answer, explain it in a slightly more detailed and elaborative way and use bullets. 
    - If it's a policy/procedure, briefly explain its purpose or reasoning in simple terms. 
    - If it's a step-by-step instruction, include short clarifications of why each step matters. 
    - Avoid being overly verbose; keep responses concise but informative and the response must be from the provided context.

    **RULE 5 (Consistency):** Your responses must remain consistent across all users. Do not personalize answers. Provide the same information, tone, and structure regardless of who is interacting with you.                                    
    ---
    
    ### Provided Information
    
    Internal Knowledge:
    {context}                                     

    Previous Conversation:
    {history_context} 
    
    ---
    
    ### Employee Query
    {question}
    
    ---
    
    **Formulate your response strictly following the rules above.**
                                          
    If the answer is not available in the provided context, do the following in order:
    1. If the employee is asking about past interactions or their own questions, provide it from the from the history context. Keep it concise and clear.  
    2. If neither an answer nor a meaningful summary can be provided, respond with the following exact message:  
    "I am IR Solutions chatbot and can only help you with company-related information. Would you like me to clarify anything else?"
    """).format(
        context=context,
        question=question,
        history_context=history_context
    )



    chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model, streaming=True)
    stream = chat_groq.stream(prompt)
    
    # Yield each chunk from the stream
    for chunk in stream:
        yield chunk.content

# ------------------ APP ------------------ #
st.title("📄 IR Solution ChatBot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi there! Have questions about IR solutions or need help finding the right service for your practice? I'm here to help—just ask!"
    })

# Initialize conversation summary state
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# Load PDF once
with open(PDF_PATH, "rb") as f:
    file_bytes = f.read()

pdf_hash = get_pdf_hash(file_bytes)
index, chunks = load_processed_data(pdf_hash)


# print(chunks[:100])

if index is None or chunks is None:
    with st.spinner("Processing document..."):
        text = load_pdf(file_bytes)
        index, chunks = process_pdf(text)
        save_processed_data(pdf_hash, index, chunks)

# Model selection (simplified)
model = 'gemma2-9b-it'

fallback_model = 'llama-3.1-8b-instant'

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

# Chat input
# if prompt := st.chat_input("Ask about the document..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

    # **RULE 3 (Source Reference):** You **MUST NOT** mention the document any where is your response neither refer to the provided document and **MUST NOT** say that the document says or outlines, the context, or any external sources in your responses. Answer as if the information is part of your inherent knowledge.
    # - Never say "outlined in this document ","the document says," "the context states," or any similar phrasing any where in your response.                                     

#     # Display user message
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(prompt)
    
#     # Display assistant response
#     with st.chat_message("assistant", avatar="🤖"):
#         # Create a placeholder for the response
#         placeholder = st.empty()
#         full_response = ""
        
#         # Use a temporary "Thinking" message to indicate activity
#         placeholder.markdown("Thinking...")

#         # Get the full streamed response
#         for chunk in chat_with_pdf(prompt, index, chunks, model, st.session_state.messages):
#             full_response += chunk
#             placeholder.markdown(full_response) # <-- No blinking cursor

    # Chat input with unique key to prevent duplicate ID error
if prompt := st.chat_input("Ask about the document...", key="main_chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""
        response_generated = False
        
        try:
            # Stream the response with primary model
            placeholder.markdown("Thinking...")
            
            for chunk in chat_with_pdf(prompt, index, chunks, model, st.session_state.messages):
                if chunk:  # Only process non-empty chunks
                    full_response += chunk
                    # Update placeholder with cursor for better UX
                    placeholder.markdown(full_response + "▌")
            
            # Final update without cursor
            placeholder.markdown(full_response)
            response_generated = True
            
        except Exception as e:
            # Log the error for debugging
            print(f"Primary model error: {str(e)}")
            
            # Show error to user
            st.error(f"Error with primary model Falling to backup Model")
            
            # Try fallback model
            try:
                st.info("Trying backup model...")
                full_response = ""  # Reset response
                
                for chunk in chat_with_pdf(prompt, index, chunks, fallback_model, st.session_state.messages):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                
                # Final update without cursor
                placeholder.markdown(full_response)
                response_generated = True
                
            except Exception as fallback_error:
                # Log fallback error
                print(f"Fallback model error: {str(fallback_error)}")
                
                # Show final error message
                st.error(f"Both models failed. Please try again later.")
                full_response = "I'm sorry, I'm experiencing technical difficulties. Please try again later."
                placeholder.markdown(full_response)
                response_generated = True

    # Only add assistant response to chat history if we generated a response
    if response_generated and full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        print(f"Total messages: {len(st.session_state.messages)}")
        
        # Rerun to update the entire chat history cleanly
        st.rerun()
# Add sidebar with conversation statistics
# with st.sidebar:
#     st.header("📊 Conversation Info")
    
    
#     if len(st.session_state.messages) > SUMMARIZATION_THRESHOLD:
#         st.write("✅ Using conversation summarization")
#         if st.button("🔄 Force Summary Refresh"):
#             # This could be used to manually refresh the summary if needed
#             st.rerun()
#     else:
#         st.write("📝 Using recent conversation context")
    
#     if st.button("🗑️ Clear Conversation"):
#         st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
#         st.session_state.conversation_summary = ""
#         st.rerun()

# Add some custom CSS for better chat alignment
st.markdown("""
<style>
    /* User messages on right */
    [data-testid="stChatMessage"] {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    /* Assistant messages on left */
    [data-testid="stChatMessage"] div:first-child {
        justify-content: flex-start;
    }
    /* User messages on right */
    [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)