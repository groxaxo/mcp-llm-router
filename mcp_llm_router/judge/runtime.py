"""Shared runtime state for the embedded judge server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_llm_router.judge.core.logging_config import (
    get_context_aware_logger,
    get_logger,
    log_startup_message,
    setup_logging,
)
from mcp_llm_router.judge.core.server_helpers import initialize_llm_configuration
from mcp_llm_router.judge.db.conversation_history_service import ConversationHistoryService
from mcp_llm_router.judge.db.db_config import load_config

setup_logging("INFO")

mcp: FastMCP | None = None

try:
    from mcp_llm_router.judge.models import rebuild_plan_approval_model
    from mcp_llm_router.judge.models.enhanced_responses import rebuild_models

    rebuild_models()
    rebuild_plan_approval_model()
except Exception as exc:  # pragma: no cover - non-critical startup compatibility
    import logging

    logging.debug(f"Server model rebuild failed (non-critical): {exc}")

initialize_llm_configuration()
config = load_config()
conversation_service = ConversationHistoryService(config)
log_startup_message(config)
logger = get_logger(__name__)
context_logger = get_context_aware_logger(__name__)


def set_registered_mcp(mcp_server: FastMCP) -> None:
    global mcp
    mcp = mcp_server


def get_registered_mcp() -> FastMCP | None:
    return mcp
