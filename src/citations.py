def format_citations(source_docs):
    """
    Takes source documents from the retriever and formats them
    into a clean, deduplicated citation list.

    Returns a list of dicts: [{"source": "citizenship.pdf", "page": 3}, ...]
    """
    citations = []
    seen = set()  # avoid duplicate (file, page) pairs

    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0) + 1  # pages are 0-indexed internally

        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "page": page})

    return citations


def citations_to_text(citations):
    """
    Converts citation list into a readable string.
    e.g. "citizenship.pdf (p.3), tax.pdf (p.7)"
    """
    if not citations:
        return "No sources found"

    parts = [f"{c['source']} (p.{c['page']})" for c in citations]
    return ", ".join(parts)