"""Quick smoke tests for MCP tool registration."""

import pytest

from mcp_llm_router.server import mcp


@pytest.mark.anyio
async def test_mcp_server_lists_tools():
    tools = await mcp.list_tools()

    tool_names = {tool.name for tool in tools}

    assert "start_session" in tool_names
    assert "agent_llm_request" in tool_names
