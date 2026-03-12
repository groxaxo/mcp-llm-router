"""Helpers for MCP roots-aware path validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, unquote

from mcp.server.fastmcp import Context


def _file_uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve(strict=False)


async def get_allowed_roots(ctx: Context | None) -> list[Path]:
    if ctx is None:
        return []

    try:
        result = await ctx.session.list_roots()
    except Exception:
        return []

    allowed_roots: list[Path] = []
    for root in getattr(result, "roots", []) or []:
        uri = getattr(root, "uri", "")
        path = _file_uri_to_path(uri)
        if path is not None:
            allowed_roots.append(path)
    return allowed_roots


async def find_paths_outside_roots(
    paths: Iterable[str],
    ctx: Context | None,
) -> list[str]:
    allowed_roots = await get_allowed_roots(ctx)
    if not allowed_roots:
        return []

    violations: list[str] = []
    for raw_path in paths:
        candidate = (raw_path or "").strip()
        if not candidate or candidate == "File path not specified":
            continue

        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve(strict=False)
            if not any(
                resolved == root or root in resolved.parents for root in allowed_roots
            ):
                violations.append(candidate)
            continue

        normalized_parts = candidate_path.parts
        if normalized_parts and normalized_parts[0] == "..":
            violations.append(candidate)

    return violations
