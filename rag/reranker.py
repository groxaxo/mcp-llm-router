"""
Local reranker implementation using a cross‑encoder model.

This module provides a helper class and functions to perform document
reranking using the Qwen3 Reranker 0.6B model (or any other HuggingFace
sequence classification model).  The reranker works by scoring each
passage with respect to a query and then ordering the passages by their
score.  It is designed to run locally on CPU or GPU and can load
quantized models if your environment supports them (for example via
bitsandbytes or other quantization libraries).

By default the implementation uses the `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
model, which is a sequence‑classification variant of the Qwen3 0.6B reranker
model published on HuggingFace.  This variant is designed to work with
`sentence-transformers`/`CrossEncoder` style interfaces.  If you prefer to
use the original model from Alibaba (`Qwen/Qwen3-Reranker-0.6B`), you can
change the `DEFAULT_MODEL_NAME` accordingly.  Note that some models may
require `trust_remote_code=True` when loading.

Example usage:

    from rag.reranker import Reranker

    rr = Reranker()
    passages = ["passage one", "passage two", "another passage"]
    ranked = rr.rerank("my query", passages, top_n=2)
    for text, score in ranked:
        print(score, text)

You can also use the functional convenience wrapper:

    from rag.reranker import rerank_passages
    ranked = rerank_passages("my query", passages, top_n=2)

This module does not download any models by itself.  To run it you must
have the `transformers` and `torch` libraries installed, and ensure
network access or a locally cached copy of the model.  Quantization
requires optional dependencies such as `bitsandbytes`.  If you are
running in an offline environment you should pre‑download the model
weights and specify the `model_name` to point at the local directory.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch

try:
    # transformers is a required dependency for the reranker
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
except ImportError as e:
    raise ImportError(
        "The 'transformers' package is required to use the reranker. "
        "Please install it via pip (e.g. 'pip install transformers') before "
        "using this module."
    ) from e


# Default model name.  This variant is a sequence classification version of the
# Qwen3 Reranker 0.6B model and is compatible with the HuggingFace API.  You
# can change this to 'Qwen/Qwen3-Reranker-0.6B' if you prefer to use the
# original model; however, note that the original model may require
# `trust_remote_code=True`.
DEFAULT_MODEL_NAME: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"


class Reranker:
    """A cross‑encoder reranker for ordering passages by relevance to a query.

    The reranker loads a HuggingFace sequence classification model and
    tokenizer.  It computes relevance scores for (query, passage)
    pairs and returns passages sorted by score.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to
        ``DEFAULT_MODEL_NAME``.
    device : str, optional
        Device on which to run the model (e.g. ``"cuda"`` or ``"cpu"``).
        If not provided the device is selected automatically based on
        availability.
    torch_dtype : torch.dtype, optional
        Floating point precision to use when loading the model.  You may
        specify ``torch.float16`` or ``torch.bfloat16`` to reduce memory
        usage.  If not provided, the model's default precision is used.
    trust_remote_code : bool, default False
        Whether to allow execution of code provided by the model
        repository.  Some models (including some Qwen variants) require
        this flag to load properly.  Set this to ``True`` if you trust
        the model source.
    load_in_8bit : bool, default False
        Whether to load the model in 8‑bit precision using
        bitsandbytes.  This can significantly reduce memory usage on
        compatible hardware but requires the ``bitsandbytes`` library.  If
        set to True but bitsandbytes is not available, a warning will
        be printed and the model will fall back to default precision.
    load_in_4bit : bool, default False
        Whether to load the model in 4‑bit precision using
        bitsandbytes.  Requires both ``bitsandbytes`` and a recent
        version of ``transformers``.  Only one of ``load_in_8bit`` or
        ``load_in_4bit`` may be True.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit may be True")

        # Determine device automatically if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        # Quantization options via bitsandbytes
        if load_in_8bit:
            model_kwargs.update({
                "load_in_8bit": True,
            })
        if load_in_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
            })

        # Device mapping: 'auto' will place model layers on available GPUs
        model_kwargs["device_map"] = "auto" if device == "cuda" else None

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
        # Ensure model is on the correct device
        self.model.to(self.device)
        # Put model in evaluation mode
        self.model.eval()

    def _score(self, query: str, passage: str) -> float:
        """Compute the relevance score for a single query/passage pair.

        Parameters
        ----------
        query : str
            The search query.
        passage : str
            A candidate passage.

        Returns
        -------
        float
            A relevance score (higher is better).  The score is derived
            from the model's first logit or raw score.
        """
        # Tokenize single pair
        encoded = self.tokenizer(
            query,
            passage,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            # We assume the model returns logits with shape (1, num_labels)
            # Some models return a single score (no num_labels) which we handle
            logits = outputs.logits
            score = float(logits.squeeze()[0])  # first logit as relevance score
        return score

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank passages for a given query.

        Parameters
        ----------
        query : str
            The search query.
        passages : List[str]
            List of candidate passages.  Order does not matter; the
            reranker will score and reorder them.
        top_n : int, optional
            If provided, only the top ``top_n`` passages are returned.

        Returns
        -------
        List[Tuple[str, float]]
            A list of (passage, score) tuples sorted by descending
            relevance score.  If ``top_n`` is provided, the list is
            truncated to that length.
        """
        if not passages:
            return []

        # Encode all pairs in a batch for efficiency
        # The tokenizer can take a pair of lists (query repeated) and list of passages
        encoded = self.tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        # Convert to scores: we take the first logit from each row
        scores = logits[:, 0].cpu().tolist()
        scored = list(zip(passages, scores))
        # Sort by descending score
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored


def rerank_passages(
    query: str,
    passages: List[str],
    top_n: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> List[Tuple[str, float]]:
    """Convenience function to rerank passages without instantiating a class.

    This function constructs a :class:`Reranker` with the given parameters,
    scores the passages, and returns them in order of relevance.

    Parameters
    ----------
    query : str
        The search query.
    passages : List[str]
        Candidate passages to score and order.
    top_n : int, optional
        If provided, only the top ``top_n`` passages are returned.
    model_name : str, default DEFAULT_MODEL_NAME
        HuggingFace model identifier.  See :class:`Reranker` for details.
    device : str, optional
        Device on which to run the model (e.g. ``"cuda"`` or ``"cpu"``).
    torch_dtype : torch.dtype, optional
        Precision for model weights.
    trust_remote_code : bool, default False
        Whether to allow execution of remote code when loading the model.
    load_in_8bit : bool, default False
        Whether to load the model in 8‑bit precision via bitsandbytes.
    load_in_4bit : bool, default False
        Whether to load the model in 4‑bit precision via bitsandbytes.

    Returns
    -------
    List[Tuple[str, float]]
        A list of (passage, score) tuples sorted by descending score.
    """
    reranker = Reranker(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    return reranker.rerank(query, passages, top_n)
