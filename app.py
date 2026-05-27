import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
import tempfile

load_dotenv()

# ── PAGE SETUP ──────────────────────────────────────
st.title("📄 PDF Chatbot")
st.write("Upload a PDF and ask anything about it!")

# ── FILE UPLOAD ─────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── INDEXING ────────────────────────────────────
    with st.spinner("Reading your PDF..."):

        # Step 1 — Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Step 2 — Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Step 3 — Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Step 4 — Store in ChromaDB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

    st.success(f"PDF loaded! {len(chunks)} chunks ready.")

    # ── CHAT ────────────────────────────────────────
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Finding answer..."):

            # Step 5 — Similarity search
            relevant_chunks = vectordb.similarity_search(question, k=3)
            context = "\n\n".join([c.page_content for c in relevant_chunks])

            # Step 6 — Send to LLM
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant.
Answer only from the context provided.
If answer not in context say 'I don't know'."""
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

            answer = response.choices[0].message.content

        # Show answer
        st.write("### Answer:")
        st.write(answer)

        # Show sources
        with st.expander("See source chunks"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk.page_content)
                st.write("---")