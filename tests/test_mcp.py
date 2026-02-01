"""Quick test to verify MCP server tools are properly defined."""
import sys
sys.path.insert(0, '/home/op/mcp-llm-router')

from mcp_llm_router.server import mcp

print("✓ MCP Server initialized successfully")
print(f"\n📋 Available tools:")
tools = mcp.list_tools()
for tool in tools:
    print(f"  - {tool.name}")
    print(f"    {tool.description[:80]}...")
print(f"\nTotal: {len(tools)} tools")
