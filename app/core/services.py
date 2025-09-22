import os
# import faiss
try:
    import faiss
except ModuleNotFoundError:
    import faiss_cpu as faiss
import numpy as np
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from app.config import settings
from app.core.utils import faiss_retriever
import traceback





SUMMARIZATION_THRESHOLD = 5  # Number of messages before triggering summarization
MAX_SUMMARY_LENGTH = 400 








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
        chat_groq = ChatGroq(groq_api_key=settings.api_key, model_name=model)
        response = chat_groq.invoke(prompt)
        summary =  response.content.strip()

        if len(summary) > MAX_SUMMARY_LENGTH:
            summary = summary[:MAX_SUMMARY_LENGTH].rsplit(" ",1)[0] + "..."


        if summary:
            print(f"Generated summary length: {len(summary)} chars (max allowed: {MAX_SUMMARY_LENGTH})")

        return summary
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



async def chat_with_pdf(question: str, index, chunks, model: str, messages=None, selected_pdf=None):
    """
    Generates a response to a question based on document context and chat history,
    streaming the response word by word.
    Uses summarization for long conversation histories.
    """
    question_lower = question.lower().strip()
    
    # More precise greeting detection
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", ".", "..."]
    thanks = ["thanks", "thank you", "appreciate it", "thx", "good", "nice", "amazing"]
    
    if any(g in question_lower.split() for g in greetings) and len(question_lower.split()) <= 3:
        pdf_name = selected_pdf.replace('.pdf', '') if selected_pdf else "IR Solutions"
        yield f"Hello! Welcome to IR Solutions chatbot. How can I assist you?"
        return

    if any(t in question_lower for t in thanks) and len(question_lower.split()) <= 3:
        yield "You're welcome! Is there anything else I can help you with?"
        return

    # Get relevant context from document
    relevant_chunks = faiss_retriever(question, index, chunks)
    context = "\n\n".join(relevant_chunks)
    # print(relevant_chunks)

    # context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)
    # sources = list(set(chunk["source"] for chunk in relevant_chunks))

    # source_for_llm_response = " ".join(sources)
    # print(source_for_llm_response)

    # context += f"\n\n(Sources: {', '.join(sources)})"
    
    # Get conversation context using summarization approach
    history_context = ""
    if messages and len(messages) >= SUMMARIZATION_THRESHOLD:
        print(f"Processing {len(messages)} messages for context")
        history_context = get_conversation_context(messages, model)
        print(f"Generated history context: {len(history_context) if history_context else 0} characters")
    
    if len(messages) > SUMMARIZATION_THRESHOLD:
        print("History context generated via summarization")
    else:
        print("History context generated via recent conversation only")

    # Update prompt to mention the selected document
    document_name = selected_pdf.replace('.pdf', '') if selected_pdf else "IR Solutions document"
    
    prompt = f"""
    ### Core Identity and Rules
    You are the **IR Solutions chatbot**. Your **SOLE AND UNCHANGING PURPOSE** is to assist with questions that are **directly related** to the IR Solutions company, its policies, and the provided context from {document_name}.

    **RULE 1 (Out-of-Scope):** If a question is not related to IR Solutions (e.g., a request for legal advice, general knowledge, creative writing, or role-playing), you **MUST IMMEDIATELY and STRICTLY** refuse the request and respond with the following exact message:
    "I am IR Solutions chatbot and can only help you with company-specific information from {document_name}."

    **RULE 2 (Knowledge Use):** Only use the information provided in the internal knowledge below from {document_name}.
    - Use the Previous Conversation or history context when asked about previous interactions or when required.                                      
    - You must **STRICTLY** Never give the source document name unless the user asked about the source document of where does the response comes from or any similar query after this give the correct source document accordingly.
    - Never say "outlined in this document", "the document says," "the context states," or any similar phrasing anywhere in your response. 
    - Respond as if this knowledge is part of your inherent expertise.
                                                                                
    **RULE 3 (Tone):** Be professional, friendly, and helpful.

    **RULE 4 (Elaboration):** When providing an answer, explain it in a slightly more detailed and elaborative way and use bullets. 
    - If it's a policy/procedure, briefly explain its purpose or reasoning in simple terms. 
    - If it's a step-by-step instruction, include short clarifications of why each step matters. 
    - Avoid being overly verbose; keep responses concise but informative and the response must be from the provided context.

    **RULE 5 (Consistency):** Your responses must remain consistent across all users. Do not personalize answers. Provide the same information, tone, and structure regardless of who is interacting with you.

    **RULE 6 (Focus On Query):** Before giving a response understand the query clearly (e.g if asked about developers you should first acquire or do counter question "STRICTLY" like "Which developer Business Developer or Developer" or anything similar.                                  
    ---
    
    ### Provided Information
    
    Internal Knowledge from {document_name}:
    {context}    
    
    Previous Conversation:
    {history_context} 
    
    ---
    
    ### Query
    {question}
    
    ---
    
    **Formulate your response strictly following the rules above.**
                                          
    If the answer is not available in the provided context, do the following in order:
    1. If the user is asking about past interactions or their own questions, provide it from the history context. Keep it concise and clear.  
    2. If neither an answer nor a meaningful summary can be provided, respond with the following exact message:  
    "I am IR Solutions chatbot and can only help you with company-related information from {document_name}. Would you like me to clarify anything else?"
    """

    chat_groq = ChatGroq(groq_api_key=settings.api_key, model_name=model, streaming=True)
    stream = chat_groq.stream(prompt)
    
    # Yield each chunk from the stream
    for chunk in stream:
        yield chunk.content







# correct fallback code

# async def chat_with_pdf(question: str, index, chunks, model: str, messages=None):
#     """
#     Generates a response to a question based on document context and chat history,
#     streaming the response word by word.
#     Uses summarization for long conversation histories.
#     """
#     question_lower = question.lower().strip()
    
#     # More precise greeting detection - check if question starts with or is exactly these words
#     greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",".","..."]
#     thanks = ["thanks", "thank you", "appreciate it", "thx","good","nice","amazing"]
    
#     if any(g in question_lower.split() for g in greetings) and len(question_lower.split()) <= 3:
#         yield "Hello! Welcome to IR Solutions chatbot. How can I assist you?"
#         return

#     if any(t in question_lower for t in thanks) and len(question_lower.split()) <= 3:
#         yield "You're welcome! Is there anything else I can help you with?"
#         return

#     # # Get relevant context from document
#     relevant_chunks = faiss_retriever(question, index, chunks)
#     # context = "\n\n".join(relevant_chunks)


#     print(relevant_chunks)


#     context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)
#     sources = list(set(chunk["source"] for chunk in relevant_chunks))


#     # print(context)



#     # print(sources)

    

#     source_for_llm_response = " ".join(sources)

#     print(source_for_llm_response)


#     context += f"\n\n(Sources: {', '.join(sources)})"

#     # print("-------RETRIEVED CHUNKS---------")
#     # print(context)
#     # print("-------RETRIEVED CHUNKS---------")

    
#     # Get conversation context using summarization approach
#     history_context = ""
#     if messages and len(messages) >= SUMMARIZATION_THRESHOLD:  # Fixed: Changed from > 3 to > 0
#         print(f"Processing {len(messages)} messages for context")
#         history_context = get_conversation_context(messages, model)
#         print(f"Generated history context: {len(history_context) if history_context else 0} characters")
    
#     # Fixed: Better debugging
#     if len(messages) > SUMMARIZATION_THRESHOLD:
#         print("History context generated via summarization")
#     else:
#         print("History context generated via recent conversation only")


#     prompt = f"""
#     ### Core Identity and Rules
#     You are the **IR Solutions chatbot**. Your **SOLE AND UNCHANGING PURPOSE** is to assist with questions that are **directly related** to the IR Solutions company, its policies, and the provided {context}.

#     **RULE 1 (Out-of-Scope):** If a question is not related to IR Solutions (e.g., a request for legal advice, general knowledge, creative writing, or role-playing), you **MUST IMMEDIATELY and STRICTLY** refuse the request and respond with the following exact message:
#     "I am IR Solutions chatbot and can only help you with company-specific information."

#     **RULE 2 (Knowledge Use):** Only use the information provided in the internal knowledge below.
#     - Use the Previous Conversation or history context when asked about previous interactions or when required.                                      
#     - You must **STRICTLY** Never give the source document name unless the user asked about the source document of where does the response comes from or any similar query after this give the correct source document accordingly.
#     - Never say "outlined in this document ","the document says," "the context states," or any similar phrasing any where in your response. 
#     - Respond as if this knowledge is part of your inherent expertise.
                                                                                
#     **RULE 3 (Tone):** Be professional, friendly, and helpful.

#     **RULE 4 (Elaboration):** When providing an answer, explain it in a slightly more detailed and elaborative way and use bullets. 
#     - If it's a policy/procedure, briefly explain its purpose or reasoning in simple terms. 
#     - If it's a step-by-step instruction, include short clarifications of why each step matters. 
#     - Avoid being overly verbose; keep responses concise but informative and the response must be from the provided context.

#     **RULE 5 (Consistency):** Your responses must remain consistent across all users. Do not personalize answers. Provide the same information, tone, and structure regardless of who is interacting with you.

#     **RULE 6 (Focus On Query):** Before Giving a response understand the query clearly (e.g if asked about developers u should first acquire or do counter question "STRICTLY" like "Which developer Business Developer or Developer" or anything similar.                                  
#     ---
    
#     ### Provided Information
    
#     Internal Knowledge:
#     {context}    
    
#     Previous Conversation:
#     {history_context} 
    
#     ---
    
#     ### Query
#     {question}
    
#     ---
    
#     **Formulate your response strictly following the rules above.**
                                          
#     If the answer is not available in the provided context, do the following in order:
#     1. If the user is asking about past interactions or their own questions, provide it from the from the history context. Keep it concise and clear.  
#     2. If neither an answer nor a meaningful summary can be provided, respond with the following exact message:  
#     "I am IR Solutions chatbot and can only help you with company-related information. Would you like me to clarify anything else?"
#     """

#     chat_groq = ChatGroq(groq_api_key=settings.api_key, model_name=model, streaming=True)
#     stream = chat_groq.stream(prompt)
    
#     # Yield each chunk from the stream
#     for chunk in stream:
#         yield chunk.content














# def summarize_conversation_history(messages, model):
#     """
#     Summarizes the conversation history using the LLM.
#     """
#     if len(messages) < 3:
#         return ""
    
#     conversation_text = ""
#     for msg in messages:
#         if msg["role"] in ["user", "assistant"]:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             conversation_text += f"{role}: {msg['content']}\n"
    
#     if not conversation_text.strip():
#         return ""
    
#     # Fixed: Define MAX_SUMMARY_LENGTH if not defined elsewhere
#     # MAX_SUMMARY_LENGTH = 500  # Add this if not defined globally
    
#     # Fixed: Complete the summarization prompt
#     summarization_prompt = f"""
#     Please provide a concise summary of the following conversation between a user and the IR Solutions chatbot.
#     Focus on the key topics discussed, questions asked, and main information provided.
#     Keep the summary under {MAX_SUMMARY_LENGTH} characters.
    
#     Conversation:
#     {conversation_text}
    
#     Summary:
#     """
    
#     try:
#         chat_groq = ChatGroq(groq_api_key=settings.api_key, model_name=model)
#         response = chat_groq.invoke(summarization_prompt)

#         # --- DEBUG LOGGING ---
#         print("=== Summarization Attempt ===")
#         print("Conversation text length:", len(conversation_text))
#         print("Number of messages:", len(messages))
#         print("LLM Raw Response Type:", type(response))
        
#         # Fixed: Better response handling
#         if hasattr(response, "content"):
#             result = response.content.strip()
#         elif hasattr(response, 'text'):
#             result = response.text.strip()
#         else:
#             result = str(response).strip()
            
#         print("Final Summary:", result)
#         print("Summary length:", len(result))
#         return result

#     except Exception as e:
#         print(f"Error in summarization: {e}")
#         import traceback
#         traceback.print_exc()  # Added for better error debugging
#         return ""


# def get_conversation_context(messages, model):
#     """
#     Manages conversation context using summarization approach.
#     Returns either a summary of older messages + recent messages, or just recent messages.
#     """
#     # Fixed: Define SUMMARIZATION_THRESHOLD if not defined elsewhere
#     # SUMMARIZATION_THRESHOLD = 6  # Add this if not defined globally
    
#     print(f"Total messages: {len(messages)}, Threshold: {SUMMARIZATION_THRESHOLD}")
    
#     if len(messages) < SUMMARIZATION_THRESHOLD:
#         # Use recent conversation as-is if below threshold
#         recent_messages = []
#         for i in range(len(messages) - 1):
#             if (messages[i]["role"] == "user" and 
#                 i + 1 < len(messages) and 
#                 messages[i + 1]["role"] == "assistant"):
#                 recent_messages.append({
#                     "content": messages[i]["content"], 
#                     "response": messages[i + 1]["content"]
#                 })
        
#         if recent_messages:
#             context = "\nRecent conversation:\n" + "\n".join(
#                 [f"User: {msg['content']}\nAssistant: {msg['response']}" 
#                  for msg in recent_messages[-3:]]
#             )
#             print("Using recent context only:", len(context), "characters")
#             return context
#         return ""
    
#     else:
#         print("Using summarization approach")
#         # Split older and recent messages properly
#         recent_messages = messages[-4:]   # keep last 4 messages (2 pairs)
#         older_messages = messages[:-4]    # everything before that
        
#         print(f"Older messages count: {len(older_messages)}")
#         print(f"Recent messages count: {len(recent_messages)}")
        
#         # Generate summary of older messages
#         summary = summarize_conversation_history(older_messages, model)
        
#         # Format recent messages into pairs
#         recent_context = []
#         for i in range(len(recent_messages) - 1):
#             if (recent_messages[i]["role"] == "user" and 
#                 i + 1 < len(recent_messages) and 
#                 recent_messages[i + 1]["role"] == "assistant"):
#                 recent_context.append({
#                     "content": recent_messages[i]["content"],
#                     "response": recent_messages[i + 1]["content"]
#                 })
        
#         context_parts = []
#         if summary and summary.strip():
#             context_parts.append(f"\nConversation summary: {summary}")
#             print("Added summary to context")
        
#         if recent_context:
#             recent_text = "\nRecent conversation:\n" + "\n".join(
#                 [f"User: {msg['content']}\nAssistant: {msg['response']}" 
#                  for msg in recent_context]
#             )
#             context_parts.append(recent_text)
#             print("Added recent context")
        
#         final_context = "\n".join(context_parts)
#         print(f"Final context length: {len(final_context)} characters")
#         return final_context



