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
import json
import hashlib
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings






os.makedirs(settings.processed_data_path, exist_ok=True)


os.makedirs(settings.combined_processed_data_path, exist_ok=True)


os.makedirs(settings.separate_processed_data_path, exist_ok=True)



def get_embeddings_model():
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder="hf_cache", 
        model_kwargs={"use_auth_token": settings.hugging_face_token}
    )


embedding_model = get_embeddings_model()


def get_pdf_hash(file_bytes):
    """Generates a SHA256 hash for the given file"""
    return hashlib.sha256(file_bytes).hexdigest()


def get_combined_pdfs_hash(pdf_files):
    """Generate a hash representing the combined set of PDFs"""
    hasher = hashlib.sha256()
    
    # Process files in consistent order
    for file_path in sorted(pdf_files):
        with open(file_path, 'rb') as f:
            # Hash the file content
            hasher.update(f.read())
        # Hash the filename to distinguish same content with different names
        hasher.update(os.path.basename(file_path).encode('utf-8'))
    
    return hasher.hexdigest()


    
def save_processed_data(pdf_hash,index,chunks):
    """Saves the FAISS index and text chunks to disk."""
    faiss.write_index(index, os.path.join(settings.separate_processed_data_path, f"{pdf_hash}.faiss"))
    with open(os.path.join(settings.separate_processed_data_path, f"{pdf_hash}.json"), 'w') as f:
        json.dump(chunks, f)


def save_combined_processed_data(pdf_hash,index,chunks):
    """Saves the FAISS index and text chunks to disk."""
    faiss.write_index(index, os.path.join(settings.combined_processed_data_path, f"{pdf_hash}.faiss"))
    with open(os.path.join(settings.combined_processed_data_path, f"{pdf_hash}.json"), 'w') as f:
        json.dump(chunks, f)



def load_processed_data(pdf_hash):
    """Loads the FAISS index and text chunks from disk if they exist."""
    idx_file = os.path.join(settings.separate_processed_data_path, f"{pdf_hash}.faiss")
    chk_file = os.path.join(settings.separate_processed_data_path, f"{pdf_hash}.json")
    if os.path.exists(idx_file) and os.path.exists(chk_file):
        index = faiss.read_index(idx_file)
        with open(chk_file, 'r') as f:
            chunks = json.load(f)
        return index, chunks
    return None, None




def load_combined_processed_data(pdf_hash):
    """Loads the FAISS index and text chunks from disk if they exist."""
    idx_file = os.path.join(settings.combined_processed_data_path, f"{pdf_hash}.faiss")
    chk_file = os.path.join(settings.combined_processed_data_path, f"{pdf_hash}.json")
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
    embeddings =  embedding_model.embed_documents(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index,chunks


def faiss_retriever(question, index, chunks):
    """Retrieves the most relevant chunks from the FAISS index based on a question."""
    q_emb = np.array([embedding_model.embed_query(question)]).astype("float32")
    _, idxs = index.search(q_emb, k=5)

    results = []

    for i in idxs[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return results



def merge_faiss_indexes(index_list):
    """
    Merge multiple FAISS indexes into a single index.
    All indexes must have the same dimension.
    """
    if not index_list:
        return None
    
    if len(index_list) == 1:
        return index_list[0]
    
    # Check dimensions consistency
    dim = index_list[0].d
    for idx in index_list:
        if idx.d != dim:
            raise ValueError("All FAISS indexes must have the same dimension to merge")

    # Merge into the first index
    final_index = faiss.IndexFlatL2(dim)  # or use the same type as your indexes
    for idx in index_list:
        xb = idx.reconstruct_n(0, idx.ntotal)  # get vectors from the index
        final_index.add(xb)

    return final_index


def merge_chunks(chunks_list):
    """
    Merge multiple chunks into a single list of chunks.
    """
    merged_chunks = []
    for chunks in chunks_list:
        merged_chunks.extend(chunks)
    return merged_chunks
