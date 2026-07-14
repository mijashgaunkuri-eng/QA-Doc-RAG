# Docs Q&A

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.39%2B-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-0.3%2B-green.svg)](https://python.langchain.com/)

A Streamlit-based **Retrieval-Augmented Generation (RAG)** application for uploading documents and asking natural-language questions grounded in their content. The system ingests PDF files, chunks and embeds them locally, stores vectors in ChromaDB, and generates answers with source citations using a Groq-hosted language model.

---

## Project Title and Description

**Docs Q&A** is a document intelligence web app that lets users upload PDF documents, automatically index them for semantic search, and chat with an AI assistant that answers strictly from retrieved context. It is designed for knowledge workers, researchers, and teams who need quick, cited answers from private document collections without manual preprocessing steps.

The application runs entirely through a browser-based Streamlit interface. Upload triggers an automatic pipeline—chunking, embedding, vector indexing, and QA chain construction—so users can start asking questions as soon as processing completes.

---

## Features

- **Automatic document ingestion** — Upload one or more PDFs; the app detects new files, chunks them, embeds them, and rebuilds the QA chain without a manual “Process” button.
- **Semantic retrieval** — Uses Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings) with ChromaDB for similarity search (`k=10` by default).
- **Grounded answers** — LangChain LCEL pipeline retrieves relevant chunks, formats context, and prompts a Groq LLM (`openai/gpt-oss-120b`, temperature `0`) to answer only from provided context.
- **Source citations** — Each response includes document name, page number, and excerpt cards for retrieved passages.
- **Per-document filtering** — Sidebar selector to search across all uploaded documents or restrict retrieval to a single file.
- **Multilingual responses** — Answer language can be set to Auto-detect, English, or Nepali (नेपाली) from the sidebar.
- **Persistent vector store** — Embeddings and metadata persist in `chroma_db/` across app restarts.
- **Incremental indexing** — New uploads are appended to the existing Chroma collection; duplicate chunks are skipped via `file_hash` and `chunk_id` metadata.
- **Document management** — View chunk counts, processing status (Queued → Processing → Ready), and remove individual documents without rebuilding the entire index.
- **Visual pipeline stepper** — Four-step UI indicator: Upload → Process → Embed → Chat.
- **Session chat history** — Conversational interface with optional chat history clearing.

> **Note:** The file uploader accepts `.pdf`, `.docx`, and `.txt` extensions, but the current ingestion path in `src/pdf_loader.py` processes **PDF files only**. DOCX and TXT support would require additional loaders.

---

## Technologies Used

| Category | Technology | Version (constraint) |
|----------|------------|----------------------|
| Language | Python | 3.10+ recommended |
| Web UI | [Streamlit](https://streamlit.io/) | `>=1.39.0, <2.0.0` |
| RAG framework | [LangChain](https://python.langchain.com/) | `>=0.3.0, <1.0.0` |
| LLM provider | [langchain-groq](https://python.langchain.com/docs/integrations/chat/groq/) | `>=0.3.0, <1.0.0` |
| Embeddings | [langchain-huggingface](https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub/) + [sentence-transformers](https://www.sbert.net/) | `>=0.3.0` / `>=3.0.0, <4.0.0` |
| Vector database | [ChromaDB](https://www.trychroma.com/) | `>=0.5.0, <1.0.0` |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io/) via LangChain `PyPDFLoader` | `>=4.0.0, <5.0.0` |
| Text splitting | [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) | `>=0.3.0, <1.0.0` |
| Configuration | [python-dotenv](https://github.com/theskumar/python-dotenv) | `>=1.0.0, <2.0.0` |
| Validation | [Pydantic](https://docs.pydantic.dev/) | `>=2.0.0, <3.0.0` |
| Optional embeddings | OpenAI (`text-embedding-3-small`) via `langchain-openai` | Supported in code, not used by default |

**Default models and settings**

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- LLM: `openai/gpt-oss-120b` on Groq
- Chunk size / overlap: `1000` / `200` characters
- Chroma collection name: `documents`
- Persist directory: `chroma_db/`

---

## Installation

### Prerequisites

- **Python 3.10 or newer**
- **pip** package manager
- A **[Groq API key](https://console.groq.com/)** for chat completions
- Sufficient disk space for embedding models (downloaded on first run) and the ChromaDB store

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd QA-Doc-RAG
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirement.txt
```

On first run, Hugging Face will download the sentence-transformers model (~90 MB).

### Step 4: Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional (only if switching the embedding provider to OpenAI in code):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is listed in `.gitignore` and must not be committed.

---

## Usage

### Start the application

```bash
streamlit run app.py
```

Streamlit opens the app in your default browser (typically `http://localhost:8501`).

### Workflow

1. **Upload** — Drag and drop or select PDF files in the main upload area.
2. **Automatic processing** — The app reads documents, splits them into chunks, builds embeddings, and indexes them in ChromaDB (typically 30–60 seconds depending on file size).
3. **Configure (optional)** — Use the sidebar to set answer language, filter by document, or remove uploaded files.
4. **Chat** — Type a question in the chat input. The assistant retrieves relevant chunks and returns an answer with citations.

### Sidebar options

| Option | Values | Description |
|--------|--------|-------------|
| Answer language | Auto-detect, English, नेपाली (Nepali) | Controls response language |
| Search within | All documents, or a specific filename | Limits retrieval to one document |
| Clear chat history | Button | Resets the conversation |

### Expected behavior

- **Before upload:** Chat is disabled; an onboarding message explains the workflow.
- **During processing:** Status badges show `Queued` → `Processing` → `Ready`; spinners indicate read, embed, and chain-building stages.
- **After processing:** Chat input is enabled; responses include inline citation cards with source file, page, and excerpt.
- **Out-of-context questions:** The model is instructed to respond with *"I couldn't find this information in the uploaded documents."*

### Runtime directories

| Path | Purpose |
|------|---------|
| `chroma_db/` | Persistent Chroma vector store (created at runtime) |
| `temp/` | Temporary PDF files during loading (cleaned per file) |

Both directories are gitignored.

---

## Project Structure

```
QA-Doc-RAG/
├── app.py                  # Streamlit UI, session state, orchestration, auto-processing
├── requirement.txt         # Python dependencies
├── README.md               # Project overview (this file)
├── DOCUMENTATION.md        # In-depth technical architecture reference
├── .gitignore              # Ignores .env, venv, chroma_db/, temp/, caches
└── src/
    ├── pdf_loader.py       # PDF loading, chunking, file hashing
    ├── embeddings.py       # ChromaDB create/load/delete, embedding providers
    ├── retriever.py        # LCEL QA chain (retrieve → prompt → Groq LLM)
    └── citations.py        # Citation formatting and deduplication
```

### Key modules

| File | Responsibility |
|------|----------------|
| `app.py` | Page layout, upload handling, automatic processing engine, chat UI, document CRUD |
| `src/pdf_loader.py` | Writes uploads to `temp/`, loads via `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter` |
| `src/embeddings.py` | Hugging Face / OpenAI embedding factories; incremental Chroma indexing and deletion by `file_hash` |
| `src/retriever.py` | Builds retriever + LCEL chain; `ask_question()` returns answer and source documents |
| `src/citations.py` | Formats unique `(source, page)` citation lists from retrieved chunks |

For architecture diagrams, session-state reference, and data-flow details, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## API Documentation

This project **does not expose a REST or GraphQL API**. All interaction occurs through the Streamlit web UI.

### Internal module interfaces

Developers extending the app can use these primary functions:

**`src/pdf_loader.py`**

```python
load_and_split_pdf(uploaded_file, chunk_size=1000, chunk_overlap=200) -> list[Document]
load_multiple_pdfs(uploaded_files, chunk_size=1000, chunk_overlap=200) -> list[Document]
get_file_hash(uploaded_file) -> str  # "{filename}_{size}"
```

**`src/embeddings.py`**

```python
create_vector_store(chunks, provider="huggingface", persist_dir="chroma_db") -> Chroma
load_vector_store(provider="huggingface", persist_dir="chroma_db") -> Chroma
delete_document_by_hash(file_hash, provider="huggingface", persist_dir="chroma_db") -> bool
get_documents_summary(provider="huggingface", persist_dir="chroma_db") -> list[dict]
search_similar(vector_store, query, k=3) -> list[Document]
```

**`src/retriever.py`**

```python
build_qa_chain(vector_store, k=10, source_filter=None, language="auto") -> tuple[chain, retriever]
ask_question(chain, retriever, question) -> tuple[str, list[Document]]
```

**`src/citations.py`**

```python
format_citations(source_docs) -> list[dict]  # [{"source": str, "page": int}, ...]
citations_to_text(citations) -> str
```

### Authentication

- **Groq API:** Authenticated via `GROQ_API_KEY` in `.env`.
- **Application:** No end-user authentication; intended for local or trusted deployment.

---

## Testing

The repository **does not currently include automated tests** (no `tests/` directory, pytest configuration, or CI workflow).

To validate manually after setup:

1. Run `streamlit run app.py`.
2. Upload a small PDF and confirm status reaches **Ready**.
3. Ask a question whose answer appears in the document; verify citations match source pages.
4. Ask an unrelated question; confirm the out-of-context fallback message.
5. Remove a document from the sidebar and confirm it no longer appears in retrieval.

If you add tests, a typical layout would be:

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

---

## Contributing

Contributions are welcome. Suggested workflow:

1. **Fork** the repository and create a feature branch from `main`.
2. **Follow existing conventions** — Match naming, module layout, and Streamlit session-state patterns in `app.py`.
3. **Keep changes focused** — Prefer minimal diffs; avoid unrelated refactors.
4. **Document behavior** — Update `README.md` and `DOCUMENTATION.md` when changing user-facing or architectural behavior.
5. **Do not commit secrets** — Never include `.env` files or API keys.
6. **Open a pull request** — Describe the problem, solution, and manual test steps.

### Coding standards

- Python 3.10+ type hints where practical
- Docstrings on public functions in `src/`
- Dependencies pinned with compatible ranges in `requirement.txt`
- Runtime artifacts (`chroma_db/`, `temp/`) remain gitignored

### Reporting issues

When filing an issue, include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or screenshots (redact API keys)

---

## License

No license file is present in this repository. Usage, modification, and distribution terms are undefined until a `LICENSE` file is added. Contact the repository owner for clarification.

---

## Acknowledgments

- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[LangChain](https://python.langchain.com/)** — RAG orchestration and LCEL chains
- **[ChromaDB](https://www.trychroma.com/)** — Embedded vector database
- **[Hugging Face](https://huggingface.co/)** — Local embedding models (`all-MiniLM-L6-v2`)
- **[Groq](https://groq.com/)** — Fast LLM inference API
- **[Sentence Transformers](https://www.sbert.net/)** — Semantic embedding models
- **[PyPDF](https://pypdf.readthedocs.io/)** — PDF text extraction

For detailed system design, memory model, and known limitations, see [DOCUMENTATION.md](DOCUMENTATION.md).
