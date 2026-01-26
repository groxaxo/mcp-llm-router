#!/usr/bin/env python3
"""Minimal test MCP server."""

from fastmcp import FastMCP

test_mcp = FastMCP('test-server')

@test_mcp.tool()
def test_tool(message: str) -> str:
    """A simple test tool that echoes the message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    test_mcp.run()
