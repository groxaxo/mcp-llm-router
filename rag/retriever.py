"""
Query retrieval utilities for embedded documents stored in Chroma.

This module embeds the query string using the same embedding model, performs a
vector search to retrieve the most similar chunks, and returns their content
along with metadata and similarity scores.
"""

from typing import List, Dict, Any

from .ollama_embedder import embed_texts
from .chroma_store import get_collection

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the query and perform a vector search against the collection.

    Args:
        query: The user query string.
        top_k: The maximum number of results to return.

    Returns:
        A list of hits containing the document text, metadata, and distance score.
    """
    collection = get_collection()
    qvec = embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"],
    )
    hits: List[Dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        hits.append(
            {
                "id": res["ids"][0][i],
                "doc": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
        )
    return hits