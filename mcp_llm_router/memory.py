"""Local memory index with embeddings and optional reranking."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import re

import httpx

DEFAULT_EMBEDDING_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
DEFAULT_EMBEDDING_PATH = os.getenv("EMBEDDINGS_PATH", "/embeddings")

DEFAULT_RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.openai.com/v1")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")
DEFAULT_RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "none")
DEFAULT_RERANK_PATH = os.getenv("RERANK_PATH", "/chat/completions")
DEFAULT_RERANK_MODE = os.getenv("RERANK_MODE", "llm")


@dataclass
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    api_key_env: Optional[str] = os.getenv("EMBEDDINGS_API_KEY_ENV", "OPENAI_API_KEY")
    model: str = DEFAULT_EMBEDDING_MODEL
    path: str = DEFAULT_EMBEDDING_PATH
    timeout_s: float = 60.0


@dataclass
class RerankConfig:
    provider: str = DEFAULT_RERANK_PROVIDER
    base_url: str = DEFAULT_RERANK_BASE_URL
    api_key_env: Optional[str] = os.getenv("RERANK_API_KEY_ENV", "OPENAI_API_KEY")
    model: str = DEFAULT_RERANK_MODEL
    path: str = DEFAULT_RERANK_PATH
    mode: str = DEFAULT_RERANK_MODE  # "llm" or "api"
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 60.0


@dataclass
class MemorySettings:
    embedding: EmbeddingConfig
    rerank: RerankConfig


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(namespace, doc_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_namespace ON memory_docs(namespace)"
            )
            conn.commit()

    async def upsert_documents(
        self,
        namespace: str,
        docs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _write() -> Dict[str, Any]:
            now = int(time.time())
            inserted = 0
            updated = 0
            with self._connect() as conn:
                for doc in docs:
                    doc_id = doc.get("doc_id") or str(uuid.uuid4())
                    content = doc["content"]
                    metadata = json.dumps(doc.get("metadata") or {})
                    embedding = json.dumps(doc["embedding"])
                    row = conn.execute(
                        "SELECT id FROM memory_docs WHERE namespace = ? AND doc_id = ?",
                        (namespace, doc_id),
                    ).fetchone()
                    if row:
                        conn.execute(
                            """
                            UPDATE memory_docs
                            SET content = ?, metadata = ?, embedding = ?, updated_at = ?
                            WHERE namespace = ? AND doc_id = ?
                            """,
                            (content, metadata, embedding, now, namespace, doc_id),
                        )
                        updated += 1
                    else:
                        conn.execute(
                            """
                            INSERT INTO memory_docs
                                (namespace, doc_id, content, metadata, embedding, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (namespace, doc_id, content, metadata, embedding, now, now),
                        )
                        inserted += 1
                conn.commit()
            return {"inserted": inserted, "updated": updated}

        return await asyncio.to_thread(_write)

    async def delete_namespace(self, namespace: str) -> int:
        def _delete() -> int:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM memory_docs WHERE namespace = ?",
                    (namespace,),
                )
                conn.commit()
                return cur.rowcount

        return await asyncio.to_thread(_delete)

    async def delete_document(self, namespace: str, doc_id: str) -> int:
        def _delete() -> int:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM memory_docs WHERE namespace = ? AND doc_id = ?",
                    (namespace, doc_id),
                )
                conn.commit()
                return cur.rowcount

        return await asyncio.to_thread(_delete)

    async def list_namespaces(self) -> List[str]:
        def _list() -> List[str]:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT namespace FROM memory_docs ORDER BY namespace"
                ).fetchall()
                return [row["namespace"] for row in rows]

        return await asyncio.to_thread(_list)

    async def stats(self) -> Dict[str, Any]:
        def _stats() -> Dict[str, Any]:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM memory_docs").fetchone()[0]
                by_ns = conn.execute(
                    "SELECT namespace, COUNT(*) as cnt FROM memory_docs GROUP BY namespace"
                ).fetchall()
                return {
                    "total": total,
                    "by_namespace": {row["namespace"]: row["cnt"] for row in by_ns},
                }

        return await asyncio.to_thread(_stats)

    async def search(
        self,
        namespace: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        def _search() -> List[Dict[str, Any]]:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT doc_id, content, metadata, embedding, updated_at FROM memory_docs WHERE namespace = ?",
                    (namespace,),
                ).fetchall()

            scored: List[Dict[str, Any]] = []
            for row in rows:
                embedding = json.loads(row["embedding"])
                score = _cosine_similarity(query_embedding, embedding)
                metadata = json.loads(row["metadata"] or "{}")
                scored.append(
                    {
                        "doc_id": row["doc_id"],
                        "content": row["content"],
                        "metadata": metadata,
                        "score": score,
                        "updated_at": row["updated_at"],
                    }
                )

            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[:top_k]

        return await asyncio.to_thread(_search)


async def embed_texts(texts: Sequence[str], config: EmbeddingConfig) -> List[List[float]]:
    if config.provider.lower() == "ollama":
        return await _embed_with_ollama(texts, config)
    return await _embed_with_openai_compatible(texts, config)


async def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    config: RerankConfig,
) -> List[Dict[str, Any]]:
    if config.provider.lower() == "none" or not documents:
        return documents

    if config.mode.lower() == "api":
        reranked = await _rerank_with_api(query, documents, config)
        if reranked:
            return reranked

    reranked = await _rerank_with_llm(query, documents, config)
    return reranked or documents


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


async def _embed_with_openai_compatible(
    texts: Sequence[str], config: EmbeddingConfig
) -> List[List[float]]:
    base_url = config.base_url.rstrip("/")
    path = config.path if config.path.startswith("/") else f"/{config.path}"
    url = base_url + path

    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key(config.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": config.model, "input": list(texts)}
    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return [item["embedding"] for item in data.get("data", [])]


async def _embed_with_ollama(
    texts: Sequence[str], config: EmbeddingConfig
) -> List[List[float]]:
    base_url = config.base_url.rstrip("/")
    path = config.path if config.path.startswith("/") else f"/{config.path}"
    url = base_url + path

    headers = {"Content-Type": "application/json"}
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        for text in texts:
            payload = {"model": config.model, "prompt": text}
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise ValueError("Ollama embedding response missing 'embedding' field")
            results.append(embedding)

    return results


async def _rerank_with_api(
    query: str, documents: List[Dict[str, Any]], config: RerankConfig
) -> List[Dict[str, Any]]:
    base_url = config.base_url.rstrip("/")
    path = config.path if config.path.startswith("/") else f"/{config.path}"
    url = base_url + path

    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key(config.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": config.model,
        "query": query,
        "documents": [doc["content"] for doc in documents],
        "top_n": len(documents),
    }

    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            return []
        data = response.json()

    results = data.get("results") or []
    if not results:
        return []

    # Typical rerank response uses index + relevance_score
    reranked: List[Dict[str, Any]] = []
    for item in results:
        index = item.get("index")
        if index is None:
            continue
        if index < 0 or index >= len(documents):
            continue
        doc = dict(documents[index])
        doc["rerank_score"] = item.get("relevance_score")
        reranked.append(doc)

    if reranked:
        reranked.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
    return reranked


async def _rerank_with_llm(
    query: str, documents: List[Dict[str, Any]], config: RerankConfig
) -> List[Dict[str, Any]]:
    base_url = config.base_url.rstrip("/")
    path = config.path if config.path.startswith("/") else f"/{config.path}"
    url = base_url + path

    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key(config.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    compact_docs = []
    for doc in documents:
        content = doc.get("content", "")
        compact_docs.append(
            {
                "id": doc.get("doc_id"),
                "content": content[:1200],
            }
        )

    system_prompt = (
        "You are a reranking engine. Respond ONLY with JSON in the format: "
        '{"results": [{"id": "...", "score": 0.0}, ...]}. '
        "Scores should reflect relevance to the query, higher is better."
    )
    user_prompt = {
        "query": query,
        "documents": compact_docs,
    }

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            return []
        data = response.json()

    content = _extract_chat_content(data)
    if not content:
        return []

    try:
        parsed = json.loads(_extract_json_object(content))
    except json.JSONDecodeError:
        return []

    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not results:
        return []

    score_map = {item.get("id"): item.get("score", 0.0) for item in results}

    reranked = []
    for doc in documents:
        doc_id = doc.get("doc_id")
        doc_copy = dict(doc)
        doc_copy["rerank_score"] = score_map.get(doc_id, 0.0)
        reranked.append(doc_copy)

    reranked.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
    return reranked


def _extract_chat_content(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def _extract_json_object(text: str) -> str:
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        return text
    return text[first_brace : last_brace + 1]


def _get_api_key(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    val = os.getenv(name)
    if val and val not in ("", "YOUR_DEEPSEEK_API_KEY", "YOUR_OPENAI_API_KEY"):
        return val

    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        try:
            with open(bashrc_path, "r", encoding="utf-8") as f:
                content = f.read()
                token = _parse_bashrc_export(content, name)
                if token:
                    return token
        except Exception:
            pass

    return None


def _parse_bashrc_export(content: str, name: str) -> Optional[str]:
    pattern = rf'^export\s+{name}=["\']?([^"\'\s#]+)["\']?'
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("export"):
            continue
        if not line.startswith(f"export {name}"):
            continue
        match = re.match(pattern, line)
        if match:
            return match.group(1)
    return None
