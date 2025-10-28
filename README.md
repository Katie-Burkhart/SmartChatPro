# 🤖 Smart Chat Pro (RAG Edition)

### A Secure, Retrieval-Augmented Python Tutoring Assistant

Smart Chat Pro is a **Retrieval-Augmented Generation (RAG)**–based chatbot built with **Streamlit** that helps students learn Python through course-aligned materials.  
It retrieves accurate, cited explanations from a **synthetic dataset of educational PDFs**, ensuring responses are **context-grounded**, **on-topic**, and **academically safe**.

---

## 🧭 Project Overview

This project was developed as part of **ISM 6564 – Large Language Models & AI Agents** under the guidance of **Dr. Tim Smith**.  
It demonstrates how a retrieval-augmented chatbot can combine **LLM reasoning**, **vector search**, and **responsible AI guardrails** to create an intelligent, trustworthy educational assistant.

### ✨ Key Features

- 🔍 **Hybrid Retrieval (Dense + BM25)** using Reciprocal Rank Fusion (RRF)
- 💬 **Context-Aware Explanations** grounded in authoritative PDF materials
- 🧠 **Reranking & Query Rewriting** with `gpt-4o-mini`
- 🧾 **Citations & Source Attribution** for every response
- 🛡️ **Robust Guardrails** against prompt injection, off-topic drift, and assignment misuse
- 📊 **Token Usage Tracking** with a visual progress bar
- 🔒 **User Authentication & Session History** via SQLite
- 🧱 **Persistent ChromaDB Vector Store** (local, reproducible setup)

---

## 🗂️ Repository Structure

```
SmartChatPro/
│
├── final_chatbot.py              # Main Streamlit app (UI + logic)
├── data_ingestion.py             # PDF ingestion and embedding pipeline
├── .env                          # Environment variables (API keys, model names)
│
├── utils/
│   ├── rag.py                    # RAG retrieval, reranking, and answer generation
│   ├── security.py               # Prompt injection & topic guardrails
│   └── text.py                   # Cleaning and chunking utilities
│
├── data/
│   ├── raw/                      # Synthetic course PDFs
│   └── processed/                # Cleaned and chunked JSONL data
│
├── vectorstore/                  # Persistent ChromaDB collection (auto-created)
├── database.py                   # Session and message storage
├── auth.py                       # User registration & authentication
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/SmartChatPro.git
cd SmartChatPro
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # (macOS/Linux)
# or
.venv\Scripts\activate           # (Windows)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
EMBED_MODEL=text-embedding-3-small
RAG_REWRITE_MODEL=gpt-4o-mini
RAG_RERANK_MODEL=gpt-4o-mini
RAG_ANSWER_MODEL=gpt-4o-mini
```

_(Make sure not to commit your API key to GitHub!)_

---

## 🧮 Build the Vector Database

Before running the chatbot, you must generate embeddings from the course PDFs:

```bash
python data_ingestion.py
```

Expected output:

```
✅ Loaded EMBED_MODEL = text-embedding-3-small
Ingested XXXX chunks of 384 dimensions into collection 'python_course_docs' ✓
Persistent vector store saved to: /path/to/vectorstore
```

---

## 💻 Run the Chatbot

```bash
streamlit run final_chatbot.py
```

The app will open in your browser at:

```
http://localhost:8501
```

### 🧑‍💻 Login or Register

- Create an account (username/password) on first launch.
- Each user’s chat sessions and message history are stored locally for persistence.

---

## 🧠 How It Works

1. **User Query → Security Filters:**  
   Input is sanitized and checked for prompt injection, assignment intent, and off-topic requests.

2. **Query Rewrite:**  
   `gpt-4o-mini` reformulates the question into a focused search query.

3. **Hybrid Retrieval:**  
   Combines semantic (vector) search and BM25 lexical search from ChromaDB using RRF.

4. **Reranking:**  
   The LLM selects the top chunks most educationally relevant to the query.

5. **Answer Generation:**  
   The system prompt enforces educational rules, and `gpt-4o-mini` generates a grounded, step-by-step explanation with citations.

---

## 🧰 Tech Stack

| Layer                          | Technology                                       |
| ------------------------------ | ------------------------------------------------ |
| **Frontend**                   | Streamlit                                        |
| **Backend / LLM API**          | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| **Database (Auth & Sessions)** | SQLite                                           |
| **Vector Store**               | ChromaDB (PersistentClient)                      |
| **Language**                   | Python 3.10+                                     |

---

## 🧪 Security & Responsible AI

Smart Chat Pro includes multiple built-in defenses:

- 🚫 **Prompt-Injection Detection**
- 🎯 **On-Topic Enforcement (Python-only)**
- 📚 **Assignment Guardrails (no full solutions)**
- 🧹 **Chunk Sanitization (detects unsafe text)**
- ⚠️ **Graceful Fallbacks when no results found**

Together, these ensure that all outputs remain **safe, ethical, and educationally aligned**.

---

## 🔮 Future Enhancements

- Instructor analytics dashboard
- Confidence scoring for retrievals
- Personalized module-based tutoring
- PDF hyperlink citations
- Migration to a managed vector DB (e.g., Pinecone or Weaviate)

---

## 🏁 Acknowledgments

Developed by **Team Smart Chat Pro**  
Course: _ISM 6564 – Large Language Models & AI Agents_  
Instructor: _Dr. Tim Smith, University of South Florida – Muma College of Business_

---

## 📜 License

This project is released for educational and academic purposes only.  
All synthetic dataset PDFs are AI-generated and do not contain copyrighted content.
