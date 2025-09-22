import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse,JSONResponse
from uuid import uuid4
# from app.chat import schemas
from .. import schemas
from app.core.services import chat_with_pdf
from app.config import settings
from app.core.utils import( embedding_model,get_pdf_hash,get_combined_pdfs_hash ,load_processed_data, load_pdf, process_pdf,save_processed_data,merge_chunks,merge_faiss_indexes, load_combined_processed_data,save_combined_processed_data)
import glob
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np





router = APIRouter()


chat_history_store = {}



# def process_all_pdfs_separately():
#     pdf_files = glob.glob(os.path.join(settings.knowledge_docs_path, "*.pdf"))

#     indexes = []
#     chunks_list = []

#     for file_path in pdf_files:
#         pdf_name = os.path.basename(file_path)
#         print(f"Processing... {pdf_name}")

#         try:
#             with open(file_path, 'rb') as f:
#                 file_bytes = f.read()

#             pdf_hash = get_pdf_hash(file_bytes)
#             index, chunks = load_processed_data(pdf_hash)

#             if index is None or chunks is None:
#                 print("Embeddings not found - Processing Pdf")
#                 text = load_pdf(file_bytes)
#                 index, chunks = process_pdf(text)
#                 save_processed_data(pdf_hash, index, chunks)
#             else:
#                 print("Embeddings found for the pdf fetching them")

            
#             tagged_chunks = [{"text": chunk, "source": pdf_name} for chunk in chunks]

#             indexes.append(index)
#             chunks_list.append(tagged_chunks)

#         except Exception as e:
#             print(f"Error processing {pdf_name}: {e}")

#     final_index = merge_faiss_indexes(indexes)
#     final_chunks = merge_chunks(chunks_list)  # already returns a flat list
#     return final_index, final_chunks



def process_all_pdfs_separately():
    pdf_files = glob.glob(os.path.join(settings.knowledge_docs_path, "*.pdf"))
    results = {}

    for file_path in pdf_files:
        pdf_name = os.path.basename(file_path)
        print(f"Processing... {pdf_name}")

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            pdf_hash = get_pdf_hash(file_bytes)
            index, chunks = load_processed_data(pdf_hash)

            if index is None or chunks is None:
                print("Embeddings not found - Processing Pdf")
                text = load_pdf(file_bytes)
                index, chunks = process_pdf(text)
                save_processed_data(pdf_hash, index, chunks)
            else:
                print("Embeddings found for the pdf fetching them")

            # store results for this PDF
            results[pdf_name] = (index, chunks)

        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")

    return results





def process_all_pdfs_combined():
    pdf_files = glob.glob(os.path.join(settings.knowledge_docs_path, "*.pdf"))
    
    # Generate combined hash for all PDFs
    combined_hash = get_combined_pdfs_hash(pdf_files)
    
    # Try to load combined result first
    final_index, final_chunks = load_combined_processed_data(combined_hash)
    if final_index is not None:
        print("Combined embeddings found - loading from cache")
        return final_index, final_chunks
    
    print("Combined embeddings not found - processing all PDFs")
    
    all_chunks = []
    all_embeddings = []
    
    for file_path in pdf_files:
        pdf_name = os.path.basename(file_path)
        print(f"Processing... {pdf_name}")

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            text = load_pdf(file_bytes)
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
            chunks = text_splitter.split_text(text)
            
            # Tag chunks with source
            tagged_chunks = [{"text": chunk, "source": pdf_name} for chunk in chunks]
            all_chunks.extend(tagged_chunks)
            
            # Create embeddings for this PDF's chunks
            embeddings = embedding_model.embed_documents(chunks)
            all_embeddings.extend(embeddings)
            
        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")

    # Create single FAISS index for all embeddings
    if all_embeddings:
        dim = len(all_embeddings[0])
        final_index = faiss.IndexFlatL2(dim)
        final_index.add(np.array(all_embeddings).astype("float32"))
    else:
        final_index = None
    
    # Save combined result
    save_combined_processed_data(combined_hash, final_index, all_chunks)
    
    return final_index, all_chunks






def get_pdf_data_by_name(pdf_name: str):
    """Get processed data (index, chunks) for a specific PDF by name"""
    pdf_path = os.path.join(settings.knowledge_docs_path, pdf_name)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF '{pdf_name}' not found in knowledge docs directory")
    
    try:
        with open(pdf_path, 'rb') as f:
            file_bytes = f.read()

        pdf_hash = get_pdf_hash(file_bytes)
        index, chunks = load_processed_data(pdf_hash)

        if index is None or chunks is None:
            raise ValueError(f"No processed data found for PDF '{pdf_name}'. Please process it first.")
        
        print(f"Loaded processed data for {pdf_name}")
        return index, chunks

    except Exception as e:
        print(f"Error loading data for {pdf_name}: {e}")
        raise e



@router.get("/make_embeddings")
async def process_all_docs_separately():
    results = process_all_pdfs_separately()
    return {"content": "all pdfs processed successfully"}




def get_available_pdfs():
    """Get list of available PDF files in the knowledge docs directory"""
    pdf_files = glob.glob(os.path.join(settings.knowledge_docs_path, "*.pdf"))
    return [os.path.basename(pdf) for pdf in pdf_files]



# Modified route
@router.post("/chat", response_model=schemas.ChatResponse)
async def query_document(request: schemas.ChatRequest):
    """
    Accepts a user query and selected PDF name, returns a streaming response from the chatbot.
    """
    # Get the selected PDF from the request
    selected_pdf = request.selected_pdf  # You'll need to add this field to ChatRequest schema
    
    if not selected_pdf:
        return {"error": "Please select a PDF document"}
    
    try:
        # Load processed data for the selected PDF only
        index, chunks = get_pdf_data_by_name(selected_pdf)
    except FileNotFoundError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error loading PDF data: {str(e)}"}
    
    # Generate response using the selected PDF's data
    async def generate_response():
        try:
            # Try primary model
            gen = chat_with_pdf(request.question, index, chunks, settings.primary_model, request.messages, selected_pdf)
            
            async for chunk in gen:
                yield chunk
        except Exception as e:
            # Fallback to the backup model
            print(f"Primary model error: {e}. Falling back...")
            fallback_gen = chat_with_pdf(request.question, index, chunks, settings.fallback_model, request.messages, selected_pdf)
            
            try:
                async for chunk in fallback_gen:
                    yield chunk
            except Exception as fallback_e:
                print(f"Fallback model also failed: {fallback_e}")
                yield "I'm sorry, I'm experiencing technical difficulties. Please try again later."
                
    return StreamingResponse(generate_response(), media_type="text/event-stream")



















# For Streaming Response will uncomment this for streamlit UI

# @router.post("/chat", response_model=schemas.ChatResponse)
# async def query_document(request: schemas.ChatRequest, processed_data: tuple = Depends(process_all_pdfs_combined)):
#     """
#     Accepts a user query and returns a streaming response from the chatbot.
#     """
#     index, chunks = processed_data
    
#     # We will handle streaming in the endpoint directly using an async generator
#     async def generate_response():
#         try:
#             # Try primary model
#             gen = chat_with_pdf(request.question, index, chunks, settings.primary_model, request.messages)
            
#             async for chunk in gen:
#                 yield chunk
#         except Exception as e:
#             # Fallback to the backup model
#             print(f"Primary model error: {e}. Falling back...")
#             fallback_gen = chat_with_pdf(request.question, index, chunks, settings.fallback_model, request.messages)
            
#             try:
#                 async for chunk in fallback_gen:
#                     yield chunk
#             except Exception as fallback_e:
#                 print(f"Fallback model also failed: {fallback_e}")
#                 yield "I'm sorry, I'm experiencing technical difficulties. Please try again later."
                
#     return StreamingResponse(generate_response(), media_type="text/event-stream")






















# Dependency to get processed document data
# For processsing one document
# def get_processed_data():
#     """
#     Loads or processes the PDF and returns the FAISS index and text chunks.
#     This function will only run the processing once.
#     """
#     try:
#         # Load PDF once
#         with open(settings.docs_path, "rb") as f:
#             file_bytes = f.read()
        
#         pdf_hash = get_pdf_hash(file_bytes)
#         index, chunks = load_processed_data(pdf_hash)

#         if index is None or chunks is None:
#             # Re-process if not found
#             print("Processing Pdf")
#             try:
#                 text = load_pdf(file_bytes)
#                 index, chunks = process_pdf(text)
#                 save_processed_data(pdf_hash,index,chunks)
#             except Exception as e:
#                 print(f"An error occur while processing pdf {e}")
            
#         return index, chunks
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing document: {e}")



# @router.post("/chat", response_model=schemas.ChatResponse)
# async def query_document(request: schemas.ChatRequest, processed_data: tuple = Depends(get_processed_data)):
#     """
#     Accepts a user query and returns chatbot response (non-streaming).
#     """
#     index, chunks = processed_data
#     response_text = ""

#     try:
#         gen = chat_with_pdf(request.question, index, chunks, settings.primary_model, request.messages)
#         async for chunk in gen:
#             response_text += chunk
#     except Exception as e:
#         print(f"Primary model error: {e}. Falling back...")
#         try:
#             fallback_gen = chat_with_pdf(request.question, index, chunks, settings.fallback_model, request.messages)
#             async for chunk in fallback_gen:
#                 response_text += chunk
#         except Exception as fallback_e:
#             print(f"Fallback model also failed: {fallback_e}")
#             response_text = "I'm sorry, I'm experiencing technical difficulties. Please try again later."

#     return JSONResponse(content={"answer": response_text})


















# /chat with session_id in case we want to use summarization but no frontend

# @router.post("/chat", response_model=schemas.ChatResponse)
# async def query_document(
#     request: schemas.ChatRequest,
#     processed_data: tuple = Depends(get_processed_data)
# ):
#     """
#     Accepts a user query and returns the chatbot response (non-streaming for Postman).
#     """
#     index, chunks = processed_data
#     session_id = request.session_id or str(uuid4())

#     if session_id not in chat_history_store:
#         chat_history_store[session_id] = []

#     chat_history_store[session_id].append({"role":"user","content":request.question})
#     response_text = ""
#     try:
#         # Try primary model
#         gen = chat_with_pdf(request.question, index, chunks, settings.primary_model, chat_history_store[session_id])
        
#         async for chunk in gen:
#             response_text += chunk

#     except Exception as e:
#         # Fallback to backup model
#         print(f"Primary model error: {e}. Falling back...")
#         try:
#             fallback_gen = chat_with_pdf(request.question, index, chunks, settings.fallback_model, chat_history_store[session_id])
            
#             async for chunk in fallback_gen:
#                 response_text += chunk

#         except Exception as fallback_e:
#             print(f"Fallback model also failed: {fallback_e}")
#             response_text = "I'm sorry, I'm experiencing technical difficulties. Please try again later."

#     chat_history_store[session_id].append({"role": "assistant", "content": response_text})


#     return JSONResponse(content={"answer": response_text, "session_id":session_id})



@router.get("/chat-testing")
def chat():
    return "Welcome to IR Solutions Chatbot"