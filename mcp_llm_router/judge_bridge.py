"""Bridge helpers for optional mcp-as-a-judge integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_JUDGE_AVAILABLE = False
_JUDGE_IMPORT_ERROR: Optional[str] = None


def _ensure_judge_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    judge_src = repo_root / "mcp-as-a-judge" / "src"
    if judge_src.exists() and str(judge_src) not in sys.path:
        sys.path.insert(0, str(judge_src))


def load_judge() -> Any:
    global _JUDGE_AVAILABLE, _JUDGE_IMPORT_ERROR

    if _JUDGE_AVAILABLE:
        import mcp_as_a_judge.server as judge_server
        return judge_server

    _ensure_judge_on_path()

    try:
        import mcp_as_a_judge.server as judge_server

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
