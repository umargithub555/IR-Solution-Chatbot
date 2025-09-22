**IR Solution Chatbot**

A chatbot designed for information retrieval (IR), enabling instant and accurate answers from a knowledge base of documents.

**Features**

**Intelligent Q&A** – Context-aware answers to natural language queries

**Knowledge Base Integration** – Retrieves grounded responses from specified documents

**Scalable & Modular** – Easy to extend with new data sources or APIs

**Vector Search** – Fast and efficient document lookup using FAISS

**User-Friendly Interface** – Simple and intuitive chat experience

**Technologies**

**Backend** – FastAPI

**Core AI** – Hugging Face Transformers (embeddings) + Groq API (generation)

**Vector Database** – Faiss

Dependencies – Managed with requirements.txt

**Gettings Started**

# Clone repository

git clone https://github.com/HighQDesk-Products/ir-document-chat-bot.git
cd ir-solution-chatbot

# Create virtual environment

python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

# Install dependencies

pip install -r requirements.txt

# Run server

uvicorn app.main:app --reload

API available at: http://127.0.0.1:8000
