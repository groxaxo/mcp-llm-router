"""
High‑level package for indexing, retrieving and reranking local files using
Qwen embeddings and a Qwen cross‑encoder reranker.

This package exposes convenience functions and classes to chunk text into
tokens, embed those chunks locally using the Ollama embedding API, store
embeddings in a persistent ChromaDB collection, retrieve relevant chunks
for a query, and rerank those chunks using a local cross‑encoder.  The
default configuration uses the Qwen3‑Embedding‑0.6B model for embeddings
and the Qwen3‑Reranker‑0.6B model (via a sequence classification
adapter) for reranking.  These defaults can be overridden via the
functions' parameters.

The key exported symbols are:

* ``chunk_text`` – break a string into overlapping token chunks.
* ``embed_texts`` – obtain L2‑normalized embeddings for a list of texts.
* ``get_client`` / ``get_collection`` – create or retrieve a ChromaDB client
  and collection.
* ``index_path`` – recursively index all files under a directory into
  ChromaDB.
* ``retrieve`` – perform a vector search over the indexed embeddings.
* ``Reranker`` – class encapsulating a cross‑encoder reranker model.
* ``rerank_passages`` – convenience function to rerank passages for a query.
"""

from .embedding_config import (
    EMBED_MODEL,
    EMBED_DIMS,
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    OLLAMA_EMBED_URL,
)
from .chunker import chunk_text
from .ollama_embedder import embed_texts
from .chroma_store import get_client, get_collection
from .indexer import index_path
from .retriever import retrieve
from .reranker import Reranker, rerank_passages

__all__ = [
    # Configuration constants
    "EMBED_MODEL",
    "EMBED_DIMS",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP",
    "COLLECTION_NAME",
    "OLLAMA_EMBED_URL",
    # Core functions
    "chunk_text",
    "embed_texts",
    "get_client",
    "get_collection",
    "index_path",
    "retrieve",
    # Reranker
    "Reranker",
    "rerank_passages",
]