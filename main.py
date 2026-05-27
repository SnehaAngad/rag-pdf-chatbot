from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
import os
load_dotenv()

# ── PHASE 1: INDEXING ──────────────────────────────

# Step 1 — Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Step 2 — Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Step 3 — Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 4 — Store in ChromaDB
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)
print(f"Total vectors stored: {vectordb._collection.count()}")

# ── PHASE 2: RETRIEVAL + GENERATION ────────────────

question = "What is COVID-19?"

# Step 5 — Similarity search
relevant_chunks = vectordb.similarity_search(question, k=3)

# Step 6 — Build context
context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

# Step 7 — Send to LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": """You are a helpful assistant.
Answer only from the context provided.
If answer is not in context say 'I don't know'."""
        },
        {
            "role": "user",
            "content": f"""Context:
{context}

Question: {question}"""
        }
    ],
    temperature=0.1
)

print("\n--- LLM Answer ---")
print(response.choices[0].message.content)