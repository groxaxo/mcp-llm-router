#!/usr/bin/env python3
"""MCP Client - Connect to and interact with MCP servers."""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def connect_to_server(server_config: Dict[str, Any]):
    """Connect to an MCP server and return session info."""
    server_params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        env=dict(os.environ) | server_config.get("env", {})
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get server info
            tools = await session.list_tools()

            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    for tool in tools
                ]
            }


async def call_tool(server_config: Dict[str, Any], tool_name: str, arguments: Dict[str, Any] = None):
    """Call a tool on an MCP server."""
    server_params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        env=dict(os.environ) | server_config.get("env", {})
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(tool_name, arguments=arguments or {})
            return result


def load_config(config_path: str) -> dict:
    """Load MCP configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


async def list_tools(server_name: str, config_path: str = None):
    """List tools available on a server."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "mcp-config.json")

    config = load_config(config_path)

    if "mcpServers" not in config or server_name not in config["mcpServers"]:
        print(f"Server '{server_name}' not found in config")
        return

    server_config = config["mcpServers"][server_name]

    try:
        result = await connect_to_server(server_config)
        print(f"🛠️  Tools available on '{server_name}':")
        for tool in result["tools"]:
            print(f"  • {tool['name']}: {tool['description'][:60]}...")
    except Exception as e:
        print(f"❌ Failed to connect to server '{server_name}': {e}")


async def run_tool(server_name: str, tool_name: str, args_json: str = None, config_path: str = None):
    """Run a tool on a server."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "mcp-config.json")

    config = load_config(config_path)

    if "mcpServers" not in config or server_name not in config["mcpServers"]:
        print(f"Server '{server_name}' not found in config")
        return

    server_config = config["mcpServers"][server_name]

    # Parse arguments
    arguments = {}
    if args_json:
        try:
            arguments = json.loads(args_json)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON arguments: {e}")
            return

    try:
        print(f"🔧 Calling tool '{tool_name}' on server '{server_name}'...")
        result = await call_tool(server_config, tool_name, arguments)

        print("✅ Tool result:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ Failed to call tool: {e}")


async def main():
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("action", choices=["list-tools", "call-tool"], help="Action to perform")
    parser.add_argument("server", help="Server name")
    parser.add_argument("tool", nargs="?", help="Tool name (for call-tool action)")
    parser.add_argument("--args", "-a", help="JSON arguments for tool call")
    parser.add_argument("--config", "-c", help="Path to config file")

    args = parser.parse_args()

    if args.action == "list-tools":
        await list_tools(args.server, args.config)
    elif args.action == "call-tool":
        if not args.tool:
            print("Tool name required for call-tool action")
            sys.exit(1)
        await run_tool(args.server, args.tool, args.args, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())