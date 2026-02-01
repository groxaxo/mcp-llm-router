"""
Embedding wrapper for calling the Ollama embed API and enforcing normalization.

This module defines a simple function that takes a list of strings and returns a
corresponding list of embedding vectors. It always requests a fixed dimension
and ensures that the returned vectors are L2-normalized.  If the Ollama
endpoint is unavailable or returns malformed data, a ValueError is raised.
"""

import math
from typing import List

import requests

from .embedding_config import EMBED_MODEL, EMBED_DIMS, OLLAMA_EMBED_URL

def _l2_normalize(vec: List[float]) -> List[float]:
    """Return a new list with the vector scaled to unit length."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the Ollama API.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embedding vectors, one per input string.

    Raises:
        ValueError: If the Ollama response is invalid or dimensions mismatch.
    """
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "dimensions": EMBED_DIMS,
        "truncate": True,
    }
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        raise ValueError(f"Ollama embed request failed: {exc}") from exc

    data = resp.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("Ollama embed response missing 'embeddings' list")

    results: List[List[float]] = []
    for vec in embeddings:
        if not isinstance(vec, list):
            raise ValueError("Unexpected embedding format")
        if len(vec) != EMBED_DIMS:
            raise ValueError(
                f"Unexpected embedding dimension {len(vec)} != {EMBED_DIMS}"
            )
        results.append(_l2_normalize(vec))

    return results