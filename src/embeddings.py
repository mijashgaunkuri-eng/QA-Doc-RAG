import os
from collections import defaultdict
from typing import List, Dict, Set

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_embeddings_model(provider: str = "huggingface"):
    """Returns an embeddings model.
    - "huggingface" = free, runs locally
    - "openai" = paid, runs in the cloud
    """
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


DEFAULT_COLLECTION_NAME = "documents"


def _get_chroma_collection(vector_store):
    """Return the underlying chroma collection object for advanced ops.
    This uses a couple of fallback paths to remain compatible with
    different langchain/chroma wrapper versions."""
    if hasattr(vector_store, "_collection") and vector_store._collection is not None:
        return vector_store._collection
    client = getattr(vector_store, "client", None)
    col_name = getattr(vector_store, "collection_name", DEFAULT_COLLECTION_NAME)
    if client is not None and hasattr(client, "get_collection"):
        try:
            return client.get_collection(col_name)
        except Exception:
            return None
    return None


def load_vector_store(provider: str = "huggingface", persist_dir: str = "chroma_db") -> Chroma:
    """Load a Chroma vector store (does not alter data)."""
    embedding_model = get_embeddings_model(provider)

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    return vector_store


def list_indexed_file_hashes(provider: str = "huggingface", persist_dir: str = "chroma_db") -> Set[str]:
    """Return set of file_hash values already present in the collection.
    Useful to populate `processed_files` at startup and avoid re-indexing."""
    vs = load_vector_store(provider=provider, persist_dir=persist_dir)
    coll = _get_chroma_collection(vs)
    hashes = set()
    if coll is None:
        return hashes

    try:
        # Ask chroma for all metadatas; this may return large payloads for huge DBs.
        data = coll.get(include=["metadatas"]) or {}
        metadatas = data.get("metadatas", [])
        for meta in metadatas:
            if not meta:
                continue
            file_hash = meta.get("file_hash")
            if file_hash:
                hashes.add(file_hash)
    except Exception:
        # Best-effort: if the wrapper does not support `get`, return empty set.
        pass

    return hashes


def get_documents_summary(provider: str = "huggingface", persist_dir: str = "chroma_db") -> List[Dict]:
    """Return a summary list of documents already indexed in Chroma.
    Each item: {name, hash, chunk_count, status}
    """
    vs = load_vector_store(provider=provider, persist_dir=persist_dir)
    coll = _get_chroma_collection(vs)
    summary = []
    if coll is None:
        return summary

    try:
        data = coll.get(include=["metadatas", "ids"]) or {}
        metadatas = data.get("metadatas", [])
        # Group by file_hash
        groups = defaultdict(lambda: {"name": None, "hash": None, "count": 0})
        for meta in metadatas:
            if not meta:
                continue
            fh = meta.get("file_hash")
            src = meta.get("source") or meta.get("source_file") or "unknown"
            if not fh:
                # If no file_hash metadata was stored, skip it — we cannot map to files.
                continue
            g = groups[fh]
            g["name"] = src
            g["hash"] = fh
            g["count"] += 1

        for fh, info in groups.items():
            summary.append({
                "name": info["name"],
                "hash": info["hash"],
                "size": 0,
                "chunk_count": info["count"],
                "status": "Ready",
            })
    except Exception:
        pass

    return summary


def create_vector_store(chunks: List, provider: str = "huggingface", persist_dir: str = "chroma_db") -> Chroma:
    """Incrementally add `chunks` to an existing Chroma collection.

    Behavior:
    - If the collection doesn't exist, it will be created by the Chroma wrapper.
    - For each chunk we expect `chunk.metadata['file_hash']` and `chunk.metadata['chunk_id']`.
    - Already-indexed chunk ids are skipped to prevent duplicates.
    - If a whole file (file_hash) is already present, all its chunks are skipped.
    """
    if not chunks:
        return load_vector_store(provider=provider, persist_dir=persist_dir)

    vs = load_vector_store(provider=provider, persist_dir=persist_dir)
    coll = _get_chroma_collection(vs)

    # Build mapping: file_hash -> set(existing_chunk_ids)
    existing_ids_by_file = defaultdict(set)
    if coll is not None:
        try:
            # For each file_hash present, collect existing ids so we can avoid duplicates.
            all_data = coll.get(include=["metadatas", "ids"]) or {}
            metadatas = all_data.get("metadatas", [])
            ids = all_data.get("ids", [])
            for meta, _id in zip(metadatas, ids):
                if not meta:
                    continue
                fh = meta.get("file_hash")
                cid = meta.get("chunk_id") or _id
                if fh:
                    existing_ids_by_file[fh].add(cid)
        except Exception:
            # If the collection API is not available, fall back to optimistic add.
            existing_ids_by_file = defaultdict(set)

    # Prepare lists to add
    to_add = []
    to_add_ids = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        fh = meta.get("file_hash")
        cid = meta.get("chunk_id")
        if fh and cid and cid in existing_ids_by_file.get(fh, set()):
            # This chunk already indexed; skip it
            continue
        # If file_hash exists with any ids and this chunk has no id, we still attempt to add
        to_add.append(chunk)
        to_add_ids.append(cid if cid is not None else None)

    if to_add:
        # LangChain Chroma wrapper expects ids list or will generate them automatically.
        try:
            vs.add_documents(to_add, ids=to_add_ids)
        except TypeError:
            # Older/newer wrappers may use different method signatures
            vs.add_documents(to_add)

        # Persist changes if supported
        if hasattr(vs, "persist"):
            try:
                vs.persist()
            except Exception:
                pass

    return vs


def delete_document_by_hash(file_hash: str, provider: str = "huggingface", persist_dir: str = "chroma_db") -> bool:
    """Delete all chunks for a single document identified by `file_hash`.

    Returns True when deletion was attempted (even if nothing existed), False on fatal errors.
    """
    if not file_hash:
        return False
    vs = load_vector_store(provider=provider, persist_dir=persist_dir)
    coll = _get_chroma_collection(vs)
    if coll is None:
        return False

    try:
        data = coll.get(where={"file_hash": file_hash}, include=["ids"]) or {}
        ids = data.get("ids", [])
        if ids:
            # Try wrapper-level delete first
            try:
                if hasattr(vs, "delete"):
                    vs.delete(ids=ids)
                else:
                    coll.delete(ids=ids)
            except Exception:
                # Fallback to collection-level delete
                coll.delete(ids=ids)

            # Persist after deletion if supported
            if hasattr(vs, "persist"):
                try:
                    vs.persist()
                except Exception:
                    pass

        return True
    except Exception:
        return False


def search_similar(vector_store, query, k: int = 3):
    """Searches the vector store for chunks similar to the query."""
    return vector_store.similarity_search(query, k=k)