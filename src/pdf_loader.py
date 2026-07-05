from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_and_split_pdf(uploaded_file, chunk_size=1000, chunk_overlap=200):
    """Load a PDF file and split it into chunks."""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Add the original filename as metadata to each document
        for page in documents:
            page.metadata["source"] = uploaded_file.name

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        return chunks
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def load_multiple_pdfs(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """Load and split multiple PDF files."""
    all_chunks = []
    for uploaded_file in uploaded_files:
        chunks = load_and_split_pdf(uploaded_file, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"Loaded and split {uploaded_file.name} into {len(chunks)} chunks.")
    return all_chunks

def get_file_hash(uploaded_file):
    """
    Creates a unique fingerprint for a file based on its name + size.
    Used to detect if we've already processed this exact file.
    """
    return f"{uploaded_file.name}_{uploaded_file.size}"