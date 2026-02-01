"""Bridge helpers for optional mcp-as-a-judge integration."""

from __future__ import annotations

from typing import Any, Optional

_JUDGE_AVAILABLE = False
_JUDGE_IMPORT_ERROR: Optional[str] = None


def load_judge() -> Any:
    global _JUDGE_AVAILABLE, _JUDGE_IMPORT_ERROR

    if _JUDGE_AVAILABLE:
        from mcp_llm_router.judge import server as judge_server

        return judge_server

    try:
        from mcp_llm_router.judge import server as judge_server

        _JUDGE_AVAILABLE = True
        _JUDGE_IMPORT_ERROR = None
        return judge_server
    except Exception as exc:  # pragma: no cover - defensive import guard
        _JUDGE_AVAILABLE = False
        _JUDGE_IMPORT_ERROR = str(exc)
        raise


def judge_available() -> bool:
    if _JUDGE_AVAILABLE:
        return True

    try:
        load_judge()
        return True
    except Exception:
        return False


def judge_import_error() -> Optional[str]:
    return _JUDGE_IMPORT_ERROR
