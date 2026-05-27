# 📄 RAG PDF Chatbot

An intelligent PDF chatbot built using **Retrieval Augmented Generation (RAG)** — ask any question about your PDF and get accurate, context-grounded answers. No hallucination.

---

## 🎯 What it does

Upload any PDF → Ask any question → Get accurate answers from your document

Built to solve the core problem of LLM hallucination — the model answers **only from your document**, not from its training memory.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free, local) |
| Vector Database | ChromaDB |
| Orchestration | LangChain |
| UI | Streamlit |
| Backend | Python |

---

## 🏗️ Architecture

```
Your PDF
    ↓
PyPDFLoader → extract text
    ↓
RecursiveCharacterTextSplitter → split into 500 char chunks
    ↓
HuggingFace Embeddings → convert chunks to vectors (384 dims)
    ↓
ChromaDB → store all vectors
    ↓
User asks question
    ↓
Question → embedding → similarity search → top 3 chunks
    ↓
Chunks + Question → Groq (Llama 3.3) → accurate answer ✅
```

---

## ✨ Features

- 📤 Upload any PDF directly in the browser
- 🔍 Semantic similarity search — finds meaning, not just keywords
- 🤖 Llama 3.3 70B for high quality answers
- 🆓 Free embeddings — runs locally, no API key needed
- 📚 Source chunks visible — see exactly where answer came from
- 🚫 No hallucination — model answers only from your document

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SnehaAngad/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📁 Project Structure

```
rag-pdf-chatbot/
│
├── app.py              ← Streamlit UI + RAG pipeline
├── main.py             ← Core RAG logic (testing)
├── requirements.txt    ← Dependencies
├── .env                ← API keys (not pushed to GitHub)
└── .gitignore          ← Protects sensitive files
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-text-splitters
sentence-transformers
chromadb
groq
python-dotenv
pypdf
```

---

## 💡 How RAG Works

**Without RAG:**
```
User asks → LLM guesses → might hallucinate ❌
```

**With RAG:**
```
User asks → fetch real chunks from PDF → LLM answers from facts ✅
```

RAG solves hallucination by grounding LLM responses in your actual documents.

---

## 🔑 Key Concepts Used

- **Chunking** — split large PDFs into 500 char pieces with 50 char overlap
- **Embeddings** — convert text to 384-dimensional vectors preserving meaning
- **Similarity Search** — find top 3 most relevant chunks for any question
- **Prompt Engineering** — instruct LLM to answer only from provided context

---

## 👩‍💻 Author

**Sneha Angadi**
- LinkedIn: [linkedin.com/in/snehangad](https://linkedin.com/in/snehangad)
- GitHub: [github.com/SnehaAngad](https://github.com/SnehaAngad)
- Email: angadisneha30@gmail.com

---

## 📄 License

MIT License — feel free to use and modify.