# Nepal Docs Q&A App

A Streamlit-based Retrieval-Augmented Generation (RAG) application for asking questions about uploaded Nepali government documents such as citizenship, tax, and company registration papers.

## What the app does

- Uploads PDF documents
- Splits them into searchable chunks
- Creates embeddings with Hugging Face
- Stores them in ChromaDB for semantic retrieval
- Answers questions using a Groq-backed language model
- Shows source citations for retrieved passages

## Project structure

- app.py — Streamlit UI and orchestration
- src/pdf_loader.py — PDF loading and chunking
- src/embeddings.py — ChromaDB vector store management
- src/retriever.py — prompt and retrieval pipeline
- src/citations.py — source citation formatting
- chroma_db/ — persistent vector database
- temp/ — temporary upload processing files

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirement.txt
```

## Environment setup

Create a .env file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the app

```bash
streamlit run app.py
```

## Workflow

1. Upload one or more PDF files.
2. The app automatically processes and embeds them.
3. Ask questions in the chat interface.
4. The system retrieves relevant chunks and generates a grounded answer with citations.

## Notes

- The current ingestion path is optimized for PDF files.
- The app uses ChromaDB for persistence across restarts.
- The answer language can be set to Auto, English, or Nepali from the sidebar.
