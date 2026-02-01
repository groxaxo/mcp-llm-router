"""
Document indexing pipeline for local files.

This module walks a directory tree, reads files with specified extensions,
chunks them into token-based segments, embeds each chunk, and upserts the
results into a Chroma collection.  Metadata is attached to each document to
record its origin and positions.
"""

import hashlib
import os
from typing import Dict, List, Optional

from .chunker import chunk_text
from .embedding_config import (
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
)
from .ollama_embedder import embed_texts
from .chroma_store import get_collection

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _iter_files(root: str, exts: Optional[List[str]] = None):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if exts and not any(path.lower().endswith(e) for e in exts):
                continue
            yield path

def index_path(
    root: str,
    exts: Optional[List[str]] = None,
    namespace_prefix: str = "file",
) -> Dict[str, int]:
    """
    Index all files under ``root`` into the Chroma collection.

    Args:
        root: Root directory to scan.
        exts: Optional list of file extensions to include (e.g. [".py",".md"]).
        namespace_prefix: Prefix for document IDs; defaults to "file".

    Returns:
        A dictionary summarizing the number of files processed and chunks indexed.
    """
    collection = get_collection()
    total_files = 0
    total_chunks = 0

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, any]] = []

    for path in _iter_files(root, exts):
        total_files += 1
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        chunks = chunk_text(text)
        for idx, (chunk, t0, t1) in enumerate(chunks):
            chunk_id = f"{namespace_prefix}:{path}:{idx}:{_sha256(chunk)[:12]}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(
                {
                    "path": path,
                    "chunk_index": idx,
                    "token_start": t0,
                    "token_end": t1,
                    "content_hash": _sha256(chunk),
                    "chunk_tokens": CHUNK_TOKENS,
                    "chunk_overlap": CHUNK_OVERLAP,
                }
            )
            total_chunks += 1

        # Batch to avoid huge memory usage
        if len(docs) >= 128:
            _flush(collection, ids, docs, metadatas)
            ids, docs, metadatas = [], [], []

    # Flush remaining
    if docs:
        _flush(collection, ids, docs, metadatas)

    return {"files_indexed": total_files, "chunks_indexed": total_chunks}

def _flush(collection, ids: List[str], docs: List[str], metas: List[Dict[str, any]]):
    """
    Embed a batch of documents and upsert them into the Chroma collection.
    """
    vectors = embed_texts(docs)
    collection.add(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)