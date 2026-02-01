"""
Token-based chunking utility for splitting text into fixed-size overlapping segments.

This module uses the Hugging Face transformers tokenizer for the Qwen3 embedding
model to compute token boundaries. Chunking by tokens instead of characters helps
ensure that each chunk contains a consistent amount of semantic information and
prevents broken tokens at the boundaries. The overlap ensures that information near
the edges of chunks is not lost.
"""

from typing import List, Tuple

from transformers import AutoTokenizer

from .embedding_config import CHUNK_TOKENS, CHUNK_OVERLAP

# Load the tokenizer once at import time. This uses the same tokenizer as the
# embedding model.
_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", use_fast=True)

def chunk_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Split the input string into a list of (chunk_text, token_start, token_end) tuples.

    Each chunk will contain at most ``CHUNK_TOKENS`` tokens and will overlap the
    previous chunk by ``CHUNK_OVERLAP`` tokens. The returned token indices refer to
    positions in the original tokenized sequence.

    Args:
        text: The raw text to split.

    Returns:
        A list of tuples containing the decoded chunk text and the start and end token
        indices of the chunk relative to the original text.
    """
    # Encode to token IDs without adding special tokens so that token positions match
    # between chunk boundaries.
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    chunks: List[Tuple[str, int, int]] = []

    start = 0
    total_tokens = len(token_ids)
    while start < total_tokens:
        end = min(start + CHUNK_TOKENS, total_tokens)
        chunk_ids = token_ids[start:end]
        chunk_text = _tokenizer.decode(chunk_ids)
        chunks.append((chunk_text, start, end))

        if end == total_tokens:
            break
        # Slide the window forward with overlap
        start = end - CHUNK_OVERLAP

    return chunks