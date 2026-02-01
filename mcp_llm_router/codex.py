"""
Codex: Local Codebase Scanner for MCP LLM Router.

This module provides functionality to scan the local codebase (or specific directories),
extract semantic structures (classes, functions, docs) using AST parsing, and
feed them into the MemoryStore for RAG capabilities.
"""

import ast
import glob
import logging
import os
from typing import Any, Dict, List, Optional

from mcp_llm_router.memory import EmbeddingConfig, MemoryStore, embed_texts

logger = logging.getLogger(__name__)


class CodexScanner:
    """Scans local codebase and indexes it into MemoryStore."""

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_config: EmbeddingConfig,
        root_path: str = ".",
    ) -> None:
        self.memory_store = memory_store
        self.embedding_config = embedding_config
        self.root_path = os.path.abspath(root_path)
        self.namespace = "codex"

    async def scan_and_index(self, path_pattern: str = "**/*.py") -> Dict[str, Any]:
        """
        Scans files matching the pattern and indexes them.

        Args:
            path_pattern: Glob pattern relative to root_path (default recursive .py)

        Returns:
            Dict with stats (scanned_files, indexed_items)
        """
        search_path = os.path.join(self.root_path, path_pattern)
        files = glob.glob(search_path, recursive=True)

        logger.info(f"Codex: Found {len(files)} files to scan in {search_path}")

        all_docs: List[Dict[str, Any]] = []

        for file_path in files:
            if not os.path.isfile(file_path):
                continue

            # Skip hidden directories/files (like .venv, .git)
            if any(part.startswith(".") for part in file_path.split(os.sep)):
                continue

            try:
                docs = self._parse_file(file_path)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Codex: Failed to parse {file_path}: {e}")

        if all_docs:
            # Generate embeddings in batches
            texts_to_embed = [doc["content"] for doc in all_docs]
            logger.info(
                f"Codex: Generating embeddings for {len(texts_to_embed)} items..."
            )

            try:
                embeddings = await embed_texts(texts_to_embed, self.embedding_config)

                # Assign embeddings back to docs
                for doc, emb in zip(all_docs, embeddings):
                    doc["embedding"] = emb

                logger.info(f"Codex: Upserting {len(all_docs)} documents to memory...")
                result = await self.memory_store.upsert_documents(
                    self.namespace, all_docs
                )
                logger.info(f"Codex: Upsert result: {result}")
                return {
                    "scanned_files": len(files),
                    "indexed_items": len(all_docs),
                    "upsert_stats": result,
                }
            except Exception as e:
                logger.error(f"Codex: Embedding/Upsert failed: {e}")
                return {
                    "scanned_files": len(files),
                    "indexed_items": 0,
                    "error": str(e),
                }

        return {"scanned_files": len(files), "indexed_items": 0}

    def _parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses a single Python file into memory documents."""
        rel_path = os.path.relpath(file_path, self.root_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        docs = []

        # 1. Index the full file content (summarized/truncated if needed, but here full)
        # For very large files, chunking would be better, but we'll start simple.
        docs.append(
            {
                "doc_id": f"file:{rel_path}",
                "content": f"File: {rel_path}\n\n{content}",
                "metadata": {"type": "file", "path": rel_path},
                "embedding": [],  # Will need embedding logic, but memory.py expects embeddings?
                # WAIT: memory.py upsert_documents takes 'docs' which MUST have 'embedding'.
                # We need to generate embeddings first!
            }
        )

        # AST Parsing for top-level definitions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = self._process_function(node, rel_path)
                if doc:
                    docs.append(doc)
            elif isinstance(node, ast.ClassDef):
                doc = self._process_class(node, rel_path, content)
                if doc:
                    docs.append(doc)

        return docs

    def _process_function(self, node: Any, file_path: str) -> Optional[Dict[str, Any]]:
        """Extracts function metadata."""
        docstring = ast.get_docstring(node) or ""
        # Basic signature reconstruction (simplified)
        name = node.name
        lineno = node.lineno

        # We store the definition source if possible, or just the signature
        # Getting exact source from AST is hard without 'ast.unparse' (Py3.9+) or the original source.
        # For now, we'll create a rich description.

        content = (
            f"Function: {name}\nFile: {file_path}:{lineno}\n\nDocstring:\n{docstring}"
        )

        return {
            "doc_id": f"func:{file_path}:{name}",
            "content": content,
            "metadata": {
                "type": "function",
                "name": name,
                "file": file_path,
                "lineno": lineno,
            },
            "embedding": [],  # Placeholder
        }

    def _process_class(
        self, node: ast.ClassDef, file_path: str, full_source: str
    ) -> Optional[Dict[str, Any]]:
        """Extracts class metadata."""
        docstring = ast.get_docstring(node) or ""
        name = node.name
        lineno = node.lineno

        # Extract method names
        methods = [
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        content = f"Class: {name}\nFile: {file_path}:{lineno}\nMethods: {', '.join(methods)}\n\nDocstring:\n{docstring}"

        return {
            "doc_id": f"class:{file_path}:{name}",
            "content": content,
            "metadata": {
                "type": "class",
                "name": name,
                "file": file_path,
                "lineno": lineno,
            },
            "embedding": [],  # Placeholder
        }
