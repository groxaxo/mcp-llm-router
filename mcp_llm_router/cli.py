"""CLI entrypoint for the MCP LLM Router server."""

from __future__ import annotations

from mcp_llm_router.server import mcp


def main() -> None:
    mcp.run()
