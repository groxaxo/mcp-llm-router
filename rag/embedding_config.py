"""
Embedding configuration constants for local embeddings using Qwen3-Embedding-0.6B via Ollama.

This module centralizes all hyperparameters used by the indexing and retrieval
pipeline so they can be consistently imported across the codebase. If any of these
constants change (model, dimensions, chunk size, overlap, embed URL) then a new
Chroma collection name will automatically be generated to avoid mixing different
embedding configurations in the same vector store.
"""

# Name of the Ollama model to use for embeddings. See `ollama pull` for options.
EMBED_MODEL = "qwen3-embedding:0.6b"

# Dimensionality of the output vectors. Qwen3-Embedding-0.6B supports up to 1024.
EMBED_DIMS = 1024

# Number of tokens per chunk when breaking documents. See `chunker.py` for usage.
CHUNK_TOKENS = 400

# Number of tokens of overlap between consecutive chunks.
CHUNK_OVERLAP = 80

# URL for the Ollama embed endpoint. This should point to a running Ollama instance.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

# Generate a deterministic collection name based on the embed configuration. This
# prevents different embed sizes or models from being mixed into the same index.
COLLECTION_NAME = (
    f"code_docs_{EMBED_MODEL.replace(':','_')}_{EMBED_DIMS}"
    f"_{CHUNK_TOKENS}_{CHUNK_OVERLAP}_l2"
)