"""
Wrapper functions for working with Chroma vector stores.

This module centralizes the initialization of a persistent Chroma client and
collection. It hides the Chroma-specific configuration details and uses the
embedding configuration to derive the collection name.  Calling
``get_collection`` multiple times will return the same collection instance.
"""

import chromadb
from chromadb.config import Settings

from .embedding_config import COLLECTION_NAME

_client = None
_collection = None

def get_client(persist_dir: str = "data/chroma"):
    """Return a global Chroma PersistentClient, creating it if necessary."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
    return _client

def get_collection(client=None):
    """
    Return the Chroma collection used for document storage.  A new collection is
    created on demand if it does not already exist. The collection is configured
    for cosine distance, which is appropriate for L2-normalized vectors.
    """
    global _collection
    if client is None:
        client = get_client()
    if _collection is None:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection