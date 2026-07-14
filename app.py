"""
Docs QA AI - Automatic Processing Version

This application enables automatic document processing workflow:
- Upload documents → System automatically chunks, embeds, and makes chat available
- No manual "Process & Embed" button needed
- All processing happens with visual progress indicators
- Chat becomes available immediately after documents are ready
"""

import io
import os
import streamlit as st
from collections import Counter

from src.pdf_loader import load_multiple_pdfs, get_file_hash
from src.embeddings import (
    create_vector_store,
    load_vector_store,
    get_documents_summary,
    delete_document_by_hash,
    list_indexed_file_hashes,
)
from src.retriever import build_qa_chain, ask_question
from src.citations import format_citations

# --- Page config ---
st.set_page_config(
    page_title="Docs QA AI",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session-state defaults ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "auto"

if "selected_source" not in st.session_state:
    st.session_state.selected_source = "All documents"

if "document_info" not in st.session_state:
    st.session_state.document_info = []

if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []

if "error" not in st.session_state:
    st.session_state.error = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_settings" not in st.session_state:
    st.session_state.current_settings = None

# Load an existing ChromaDB store on app startup if it exists.
if st.session_state.vector_store is None and os.path.exists("chroma_db"):
    try:
        st.session_state.vector_store = load_vector_store(
            provider="huggingface",
            persist_dir="chroma_db",
        )
        # Populate document info and processed_files from the existing store
        try:
            existing_docs = get_documents_summary(provider="huggingface", persist_dir="chroma_db")
            if existing_docs and not st.session_state.document_info:
                st.session_state.document_info.extend(existing_docs)
            st.session_state.processed_files.update({d["hash"] for d in existing_docs})
        except Exception:
            # best-effort sync; do not fail app startup
            pass
    except Exception as exc:
        st.session_state.error = str(exc)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_stepper(current_step):
    """Display 4-step pipeline with visual indicators."""
    step_titles = ["Upload", "Process", "Embed", "Chat"]
    cols = st.columns(4)
    for index, title in enumerate(step_titles, start=1):
        if current_step > index:
            cols[index - 1].markdown(f"*✓ {title}*")
        elif current_step == index:
            cols[index - 1].markdown(f"*▶ {title}*")
        else:
            cols[index - 1].markdown(
                f"<span style='color: #6c757d'>{title}</span>",
                unsafe_allow_html=True
            )


def format_size(bytes_size):
    """Convert bytes to human-readable KB format."""
    return f"{round(bytes_size / 1024, 1)} KB"


def make_uploaded_file_holder(name, data, size):
    """Create a mock uploaded file object from binary data."""
    class UploadedFileHolder:
        def __init__(self, name, data, size):
            self.name = name
            self._data = data
            self.size = size

        def getbuffer(self):
            return self._data

    return UploadedFileHolder(name, data, size)


def ensure_doc_info():
    """Sync pending uploads with document_info list."""
    existing_hashes = {doc["hash"] for doc in st.session_state.document_info}
    for pending in st.session_state.pending_uploads:
        if pending["hash"] not in existing_hashes:
            st.session_state.document_info.append({
                "name": pending["name"],
                "size": pending["size"],
                "hash": pending["hash"],
                "status": "Queued",
                "chunk_count": 0,
            })
            existing_hashes.add(pending["hash"])


def refresh_doc_counts():
    """Update chunk counts and status for each document."""
    counts = Counter(c.metadata.get("source", "unknown") for c in st.session_state.chunks)
    for doc in st.session_state.document_info:
        # Prefer counts from in-memory chunks; fall back to existing stored value.
        doc["chunk_count"] = counts.get(doc["name"], doc.get("chunk_count", 0))
        if doc["hash"] in st.session_state.processed_files:
            doc["status"] = "Ready"
        elif doc["status"] not in ["Processing", "Queued"]:
            doc["status"] = "New"


def remove_document(doc_hash):
    """Delete a document and its chunks from Chroma by `file_hash` without full rebuild."""
    # Remove local session references
    st.session_state.document_info = [
        doc for doc in st.session_state.document_info
        if doc["hash"] != doc_hash
    ]
    st.session_state.processed_files.discard(doc_hash)
    st.session_state.pending_uploads = [
        pending for pending in st.session_state.pending_uploads
        if pending["hash"] != doc_hash
    ]
    st.session_state.chunks = [
        chunk for chunk in st.session_state.chunks
        if chunk.metadata.get("file_hash") != doc_hash
    ]

    # Delete from Chroma collection (incremental delete)
    try:
        delete_document_by_hash(doc_hash, provider="huggingface", persist_dir="chroma_db")
    except Exception as exc:
        st.session_state.error = str(exc)

    # Reload vector store so retriever/chain reflect the deletion
    try:
        if os.path.exists("chroma_db"):
            st.session_state.vector_store = load_vector_store(provider="huggingface", persist_dir="chroma_db")
        else:
            st.session_state.vector_store = None
    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.vector_store = None

    st.session_state.qa_chain = None
    st.session_state.retriever = None


def build_file_status_row(doc):
    """Return emoji badge for document status."""
    status_icons = {
        "New": "📄 New",
        "Queued": "⏰ Queued",
        "Processing": "⏳ Processing",
        "Ready": "✅ Ready",
    }
    return status_icons.get(doc["status"], doc["status"])


def process_new_uploads_automatically():
    """
    AUTOMATIC PROCESSING ENGINE
    
    This function detects new files and processes them without user action:
    1. Detect files in pending_uploads not in processed_files
    2. Load and chunk documents
    3. Build vector store
    4. Rebuild QA chain
    5. Update session state with progress feedback
    
    Returns: True if processing succeeded, False otherwise
    """
    new_pending = [
        pending for pending in st.session_state.pending_uploads
        if pending["hash"] not in st.session_state.processed_files
    ]
    
    if not new_pending:
        return False  # No new files to process
    
    # Mark documents as processing
    for doc in st.session_state.document_info:
        if doc["hash"] in {p["hash"] for p in new_pending}:
            doc["status"] = "Processing"
    
    st.session_state.processing = True
    st.session_state.error = None
    
    try:
        # Create a container for progress updates
        progress_container = st.container()
        
        # --- Stage 1: Load and chunk documents ---
        with progress_container.spinner("📖 Reading documents…"):
            loader_files = [
                make_uploaded_file_holder(p["name"], p["data"], p["size"])
                for p in new_pending
            ]
            new_chunks = load_multiple_pdfs(loader_files)
        
# Attach file_hash and stable chunk IDs to each new chunk.
        file_chunk_counters = {}
        for chunk in new_chunks:
            file_hash = next(
                (pending["hash"] for pending in new_pending
                 if pending["name"] == chunk.metadata.get("source")),
                None,
            )
            chunk.metadata["file_hash"] = file_hash

            counter = file_chunk_counters.get(file_hash, 0)
            chunk_id = f"{file_hash}_{counter}"
            chunk.metadata["chunk_id"] = chunk_id
            file_chunk_counters[file_hash] = counter + 1
        
        # Add chunks to session state
        st.session_state.chunks.extend(new_chunks)
        
        # Mark files as processed
        for pending in new_pending:
            st.session_state.processed_files.add(pending["hash"])
        
        # Remove from pending
        st.session_state.pending_uploads = [
            pending for pending in st.session_state.pending_uploads
            if pending["hash"] not in {p["hash"] for p in new_pending}
        ]
        
        refresh_doc_counts()

        # Automatically focus on the newest uploaded document
        st.session_state.selected_source = new_pending[-1]["name"]
        
        # --- Stage 2: Build embeddings and vector store ---
        with progress_container.spinner("✨ Building embeddings…"):
            chunks_to_index = new_chunks
            vector_store = create_vector_store(
                chunks_to_index,
                provider="huggingface"
            )
            st.session_state.vector_store = vector_store
        
        # --- Stage 3: Rebuild QA chain ---
        if st.session_state.vector_store:
            with progress_container.spinner("🔗 Building QA chain…"):
                chain, retriever = build_qa_chain(
                    st.session_state.vector_store,
                    source_filter=st.session_state.selected_source,
                    language=st.session_state.language,
                )
                st.session_state.qa_chain = chain
                st.session_state.retriever = retriever
                st.session_state.current_settings = (
                    st.session_state.selected_source,
                    st.session_state.language
                )
        
        st.session_state.processing = False
        progress_container.success("✅ Documents ready! You can now ask questions.")
        return True  # Processing succeeded
        
    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.processing = False
        progress_container.error(f"⚠️ Processing failed: {exc}")
        
        # Mark files as failed
        for doc in st.session_state.document_info:
            if doc["hash"] in {p["hash"] for p in new_pending}:
                if doc["status"] == "Processing":
                    doc["status"] = "New"
        
        return False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("Docs QA AI")
    st.write("Document manager, language settings and status live here.")
    st.divider()

    # Language selection with pill-style buttons
    st.write("*Answer language*")
    lang_cols = st.columns(3)
    lang_options = [("Auto-detect", "auto"), ("English", "English"), ("नेपाली", "Nepali")]
    for col_idx, (label, value) in enumerate(lang_options):
        active = st.session_state.language == value
        button_label = f"*{label}*" if active else label
        if lang_cols[col_idx].button(button_label, use_container_width=True):
            st.session_state.language = value
            st.session_state.current_settings = None
            st.session_state.qa_chain = None
            st.session_state.retriever = None
    st.divider()

    # Source filter
    if st.session_state.document_info:
        doc_options = ["All documents"] + [doc["name"] for doc in st.session_state.document_info]
        current_idx = doc_options.index(st.session_state.selected_source) if st.session_state.selected_source in doc_options else 0
        st.session_state.selected_source = st.selectbox(
            "Search within",
            options=doc_options,
            index=current_idx,
        )
    else:
        st.session_state.selected_source = "All documents"

    st.divider()

    # Document list with delete buttons
    if st.session_state.document_info:
        st.write("*Uploaded documents*")
        for doc in st.session_state.document_info:
            row_cols = st.columns([0.6, 2.5, 1, 1])
            row_cols[0].write("📄")
            row_cols[1].write(f"*{doc['name']}*")
            row_cols[2].write(f"{doc['chunk_count']} chunks")
            row_cols[3].write(build_file_status_row(doc))
            if row_cols[3].button("🗑", key=f"remove_{doc['hash']}", help="Remove document"):
                remove_document(doc["hash"])
                st.rerun()

        st.divider()
        if st.session_state.vector_store and st.session_state.chunks:
            st.success(
                f"✓ Vector store ready — {len(st.session_state.chunks)} chunks "
                f"across {len(st.session_state.document_info)} document(s)"
            )

    # Document statistics
    with st.expander("📊 Document stats"):
        if st.session_state.chunks:
            total_chunks = len(st.session_state.chunks)
            avg_chunk = round(sum(len(c.page_content) for c in st.session_state.chunks) / total_chunks)
            sources = sorted(set(c.metadata.get("source", "unknown") for c in st.session_state.chunks))
            st.metric("Total chunks", total_chunks)
            st.metric("Avg chunk size", f"{avg_chunk} chars")
            st.metric("Documents", len(sources))

            source_counts = Counter(c.metadata.get("source", "unknown") for c in st.session_state.chunks)
            for source, count in source_counts.items():
                st.write(f"📄 *{source}* — {count} chunks")
        else:
            st.write("Upload documents to see statistics.")

    st.divider()
    if st.button("🗑 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ============================================================================
# MAIN CONTENT - UPLOAD SECTION
# ============================================================================

st.title("Docs QA AI")
st.markdown("### Upload, process, embed and chat with your documents")

# Determine current pipeline step
if st.session_state.document_info:
    if st.session_state.vector_store:
        current_step = 4
    elif st.session_state.chunks:
        current_step = 3
    else:
        current_step = 2 if st.session_state.processing else 1
else:
    current_step = 1

build_stepper(current_step)
st.divider()

# Upload interface
st.subheader("Step 1: Upload your documents")
uploaded_files = st.file_uploader(
    "Drag and drop or select PDF, DOCX, or TXT files",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt"],
)

# Process newly uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)
        # Check if this file is already in pending or processed
        if file_hash not in {pending["hash"] for pending in st.session_state.pending_uploads} and \
           file_hash not in st.session_state.processed_files:
            st.session_state.pending_uploads.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "hash": file_hash,
                "data": uploaded_file.getbuffer(),
            })
    ensure_doc_info()
    refresh_doc_counts()

# Show onboarding or document list
if not st.session_state.document_info:
    st.info(
        "📚 *Workflow*: Upload → Automatic Processing → Chat\n\n"
        "After you upload documents, the system will automatically chunk, embed, and prepare them for questions. "
        "No manual steps needed!"
    )
else:
    st.write("### Uploaded files")
    for doc in st.session_state.document_info:
        cols = st.columns([0.5, 3, 1, 1, 1])
        cols[0].write("📄")
        cols[1].write(f"*{doc['name']}*")
        cols[2].write(format_size(doc["size"]))
        cols[3].write(f"{doc['chunk_count']} chunks")
        cols[4].write(build_file_status_row(doc))


# ============================================================================
# AUTOMATIC PROCESSING - NO BUTTON CLICK NEEDED
# ============================================================================

# Automatically process new uploads
if st.session_state.pending_uploads and not st.session_state.processing:
    st.divider()
    process_success = process_new_uploads_automatically()
    if process_success:
        st.rerun()  # Rerun to update UI


# ============================================================================
# QA CHAIN BUILD / REFRESH
# ============================================================================

# Rebuild QA chain if settings changed
if st.session_state.vector_store is not None:
    current_settings = (st.session_state.selected_source, st.session_state.language)
    if st.session_state.current_settings != current_settings or st.session_state.qa_chain is None:
        try:
            chain, retriever = build_qa_chain(
                st.session_state.vector_store,
                source_filter=st.session_state.selected_source,
                language=st.session_state.language,
            )
            st.session_state.qa_chain = chain
            st.session_state.retriever = retriever
            st.session_state.current_settings = current_settings
        except Exception as exc:
            st.session_state.error = str(exc)
            st.error(f"⚠️ Failed to build QA chain: {exc}")


# ============================================================================
# CHAT INTERFACE
# ============================================================================

st.divider()
st.subheader("💬 Chat")

if st.session_state.vector_store is None:
    st.info(
        "Your chat will appear here once documents are uploaded and automatically processed. "
        "This usually takes 30-60 seconds depending on document size."
    )

# Show greeting message if ready but no chat history
if st.session_state.vector_store and not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(
            f"🎉 Ready! Ask me anything about your {len(st.session_state.document_info)} "
            f"document(s) in English or Nepali."
        )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            for citation in msg["citations"]:
                st.caption(f"📄 {citation['source']} (p.{citation['page']})")

# Chat input
chat_prompt = "Ask about your documents... / आफ्नो कागजातहरू बारे सोध्नुहोस्..."
user_question = st.chat_input(
    chat_prompt,
    disabled=not (st.session_state.qa_chain and st.session_state.retriever),
)

if not (st.session_state.qa_chain and st.session_state.retriever):
    st.caption("💡 Chat will be enabled once your documents are processed.")
else:
    st.caption("💡 Tip: Press Enter to send your question")

# Process user question
if user_question and st.session_state.qa_chain and st.session_state.retriever:
    # Show user message
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            answer, source_docs = ask_question(
                st.session_state.qa_chain,
                st.session_state.retriever,
                user_question,
            )
            citations = format_citations(source_docs)

        st.write(answer)

        # Display citations as info cards
        if citations:
            unique_keys = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", 0) + 1
                key = (source, page)
                if key in unique_keys:
                    continue
                unique_keys.add(key)
                excerpt = doc.page_content.strip().replace("\n", " ")[:140]
                st.info(f"📄 {source} · p.{page} · {excerpt}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )