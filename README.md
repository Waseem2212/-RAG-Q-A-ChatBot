# RAG Q&A ChatBot

This project is a **Retrieval-Augmented Generation (RAG) ChatBot** built with **LangChain 2025**, **Google Gemini embeddings**, and **Streamlit** for a modern AI-powered question-answering interface. The bot can process PDFs, DOCX, and TXT files, create embeddings, and answer questions using relevant document chunks.

## Features

- Upload PDF, DOCX, or TXT files via a Streamlit interface.
- Chunk documents and create embeddings using **Google Gemini embeddings**.
- Build a **FAISS vector store** for efficient document retrieval.
- Perform **RAG-based question answering** using LangChain with Gemini LLMs.
- Display concise answers along with source document chunks.
- Robust handling of synchronous and asynchronous retrievers.

## Technologies Used

- Python 3.10+
- Streamlit
- LangChain 2025
- Google Generative AI (Gemini embeddings)
- FAISS vector store
- PyPDFLoader, Docx2txtLoader, TextLoader
- dotenv for environment variable management


