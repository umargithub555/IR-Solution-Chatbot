 # 💬 IR Solution Chatbot (FastAPI)

This is an Information Retrieval (IR) Chatbot API built with FastAPI, FAISS, and Groq API.
It enables instant and accurate answers from a knowledge base of documents using vector search + AI generation.

Query Understanding → Natural language input with context

Knowledge Base → Retrieves grounded responses from documents

AI Generation → Uses Groq API for fluent and accurate answers

## 🚀 Features

Intelligent Q&A with context-aware answers

Knowledge Base Integration for document-based retrieval

Vector Search (FAISS) for fast and efficient document lookup

Scalable & Modular architecture for easy extensions

User-Friendly Interface for smooth chat experience

## 📦 Tech Stack

**Backend:** FastAPI

**Core AI:** Hugging Face Transformers (embeddings) + Groq API (generation)

**Vector Database:** FAISS

**Dependencies:** Managed with requirements.txt

**Environment:** Python 3.12


## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/HighQDesk-Products/ir-document-chat-bot.git
cd ir-solution-chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
```bash
uvicorn app.main:app --reload
```

### 🌐 API Access
Once the server is running, the API will be available at:

👉 http://127.0.0.1:8000