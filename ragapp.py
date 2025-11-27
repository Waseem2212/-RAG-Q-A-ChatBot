import asyncio
import inspect
import os
import streamlit as st
from dotenv import load_dotenv
from typing import List

# load env
load_dotenv()

# --- Google GenAI / LangChain imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------
# LLM and embeddings (Gemini)
# ------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_DOCUMENT"
)

# ------------------------------
# Helper: safe batching for embeddings
# ------------------------------
def embed_texts_in_batches(embeddings_obj, texts: List[str], batch_size: int = 1) -> List[List[float]]:
    """
    Some Gemini embedding endpoints expect small batches or don't accept large batches.
    This helper calls embeddings.embed_documents on small batches and returns a list of vectors.
    """
    all_vectors: List[List[float]] = []
    # choose small batch by default to avoid "unexpected model name format" triggered by wrapper batching
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # embed_documents should return List[List[float]]
        vectors = embeddings_obj.embed_documents(batch)
        all_vectors.extend(vectors)
    return all_vectors

# ------------------------------
# Robust retriever call helpers
# ------------------------------
def _is_coroutine_callable(obj, name: str) -> bool:
    attr = getattr(obj, name, None)
    return callable(attr) and inspect.iscoroutinefunction(attr)

def _is_sync_callable(obj, name: str) -> bool:
    attr = getattr(obj, name, None)
    return callable(attr) and not inspect.iscoroutinefunction(attr)

def _maybe_run_async_fn(fn, *args, **kwargs):
    """
    If fn is coroutine function, run it with asyncio.run (nest_asyncio already applied).
    Otherwise call it synchronously.
    """
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))
    else:
        return fn(*args, **kwargs)

# ------------------------------
# RAG function (robust)
# ------------------------------
def run_rag(llm, retriever, question: str, top_k: int = 4):
    """
    Robust RAG retrieval that supports different Retriever APIs:
      - retriever.invoke(query)
      - retriever.get_relevant_documents(query)
      - retriever.ainvoke(query)  (async)
      - retriever.aget_relevant_documents(query) (async older style)
    Falls back gracefully and returns top_k docs.
    """
    docs = None

    # 1) Prefer synchronous invoke()
    if _is_sync_callable(retriever, "invoke"):
        try:
            docs = retriever.invoke(question)
        except Exception:
            docs = None

    # 2) If invoke is async coroutine
    if docs is None and _is_coroutine_callable(retriever, "invoke"):
        try:
            docs = _maybe_run_async_fn(retriever.invoke, question)
        except Exception:
            docs = None

    # 3) Try old-fashioned get_relevant_documents (sync)
    if docs is None and _is_sync_callable(retriever, "get_relevant_documents"):
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            docs = None

    # 4) Try async variant aget_relevant_documents or ainvoke or aget
    if docs is None:
        for async_name in ("aget_relevant_documents", "ainvoke", "aget", "aget_documents"):
            if _is_coroutine_callable(retriever, async_name):
                try:
                    docs = _maybe_run_async_fn(getattr(retriever, async_name), question)
                    break
                except Exception:
                    docs = None

    # 5) Try synchronous alternative names (some wrappers use different naming)
    if docs is None:
        for sync_name in ("get_relevant_documents", "get_documents", "retrieve", "search", "similarity_search"):
            if _is_sync_callable(retriever, sync_name):
                try:
                    docs = getattr(retriever, sync_name)(question)
                    break
                except Exception:
                    docs = None

    # 6) Final fallback: try attribute access and call if callable
    if docs is None:
        maybe_call = getattr(retriever, "get_relevant_documents", None)
        if callable(maybe_call):
            try:
                docs = maybe_call(question)
            except Exception:
                docs = None

    # If still nothing, raise clear error
    if docs is None:
        raise AttributeError(
            "Retriever object does not expose a compatible retrieval method. "
            "Tried .invoke(), .ainvoke(), .get_relevant_documents(), .aget_relevant_documents(), etc."
        )

    # Ensure docs is a list and slice to top_k
    try:
        docs = list(docs)[:top_k]
    except Exception:
        raise ValueError("Retriever returned non-iterable or malformed docs. Got: %r" % (docs,))

    # Build context and prompt as before
    context = " ".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert assistant.
        Answer the question using the context below. Be concise.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Answer:
        """
    )

    # call llm (your existing llm.invoke usage)
    answer = llm.invoke(prompt.format(context=context, question=question))

    return {
        "answer": answer,
        "sources": [d.page_content[:400] for d in docs]
    }

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="RAG Q&A ChatBot", layout="wide")
st.title("RAG Q&A ChatBot")

# Sidebar: file uploader
upload_file = st.sidebar.file_uploader("Choose file", type=["pdf", "txt", "docx"])
upload_button = st.sidebar.button("Upload & Process")

# Init session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------
# File processing
# -------------------
if upload_file and upload_button:
    with st.spinner("Processing file and building embeddings..."):
        # save temp file
        file_path = f"temp_{upload_file.name}"
        with open(file_path, "wb") as f:
            f.write(upload_file.read())

        # choose loader
        ext = upload_file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "docx":
            loader = Docx2txtLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path)
        else:
            st.error("Unsupported file format!")
            st.stop()

        # load and split
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Prepare texts and metadatas
        texts = [d.page_content for d in chunks]
        try:
            metadatas = [d.metadata if hasattr(d, "metadata") else {} for d in chunks]
        except Exception:
            metadatas = [{} for _ in chunks]

        # --- Embed in safe small batches ---
        # If Gemini rejects large batches, set batch_size=1. If your wrapper supports small batches,
        # you can increase batch_size to e.g. 8 or 16 for speed.
        batch_size = 1
        st.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
        vectors = embed_texts_in_batches(embeddings, texts, batch_size=batch_size)

        if len(vectors) != len(texts):
            st.error("Embedding count does not match text count — aborting.")
            st.stop()

        # Build list of (text, vector) pairs or (vector, metadata) depending on FAISS.from_embeddings signature.
        # We'll use FAISS.from_embeddings which accepts a list of (text, embedding) tuples and optional metadatas.
        text_embedding_pairs = list(zip(texts, vectors))

        # Build FAISS index from precomputed embeddings (this avoids the wrapper re-calling embeddings in a large batch)
        try:
            vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings)
            # optional: attach metadatas (if FAISS.from_embeddings supports a metadata argument)
            # Some langchain versions accept metadatas=metadatas in from_embeddings; if not supported, it's okay.
        except TypeError:
            # fallback if signature is different: try to pass metadatas
            try:
                vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metadatas)
            except Exception as e:
                st.error(f"Failed to create FAISS vectorstore from embeddings: {e}")
                st.stop()

        st.session_state.vectorstore = vectorstore
        st.success("File processed and vectorstore built successfully!")

# -------------------
# Question answering
# -------------------
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()
    user_query = st.text_input("Ask any question from your file:")

    if st.button("See Answer") and user_query.strip() != "":
        with st.spinner("Fetching answer..."):
            response = run_rag(llm, retriever, user_query)

        # Show answer
        st.subheader("Answer")
        st.write(response["answer"])

        # Show sources
        with st.expander("Sources / Retrieved Chunks"):
            for i, s in enumerate(response["sources"]):
                st.text(f"Chunk {i+1} preview:\n{s}")
