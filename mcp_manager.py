#!/usr/bin/env python3
"""MCP Server Manager - Manage multiple MCP servers and their connections."""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPServerManager:
    """Manager for MCP server connections and orchestration."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "mcp-config.json")
        self.config = self.load_config()
        self.connections: Dict[str, Dict[str, Any]] = {}

    def load_config(self) -> dict:
        """Load MCP configuration."""
        if not os.path.exists(self.config_path):
            return {"mcpServers": {}}

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_config(self):
        """Save current configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def add_server(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        """Add a new MCP server to the configuration."""
        if "mcpServers" not in self.config:
            self.config["mcpServers"] = {}

        self.config["mcpServers"][name] = {
            "command": command,
            "args": args or [],
            "env": env or {}
        }
        self.save_config()
        print(f"✅ Added MCP server: {name}")

    def remove_server(self, name: str):
        """Remove an MCP server from configuration."""
        if "mcpServers" in self.config and name in self.config["mcpServers"]:
            del self.config["mcpServers"][name]
            self.save_config()
            print(f"✅ Removed MCP server: {name}")
        else:
            print(f"❌ Server '{name}' not found")

    def list_servers(self):
        """List all configured servers."""
        if "mcpServers" not in self.config:
            print("No servers configured")
            return

        print("📋 Configured MCP Servers:")
        for name, config in self.config["mcpServers"].items():
            cmd = [config["command"]] + config.get("args", [])
            print(f"  • {name}: {' '.join(cmd)}")

    async def test_server(self, name: str):
        """Test connection to an MCP server."""
        if name not in self.config.get("mcpServers", {}):
            print(f"❌ Server '{name}' not found")
            return False

        server_config = self.config["mcpServers"][name]

        try:
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=dict(os.environ) | server_config.get("env", {})
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await session.list_tools()

                    print(f"✅ Server '{name}' connected successfully")
                    print(f"   Tools available: {len(tools)}")
                    for tool in tools[:5]:  # Show first 5 tools
                        print(f"     • {tool.name}")
                    if len(tools) > 5:
                        print(f"     ... and {len(tools) - 5} more")

                    return True

        except Exception as e:
            print(f"❌ Failed to connect to server '{name}': {e}")
            return False

    async def call_tool_across_servers(self, tool_name: str, arguments: Dict[str, Any] = None):
        """Call a tool across all configured servers and return results."""
        results = {}

        for server_name in self.config.get("mcpServers", {}):
            try:
                server_config = self.config["mcpServers"][server_name]

                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=dict(os.environ) | server_config.get("env", {})
                )

                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        # Check if tool exists
                        tools = await session.list_tools()
                        tool_names = [t.name for t in tools]

                        if tool_name in tool_names:
                            result = await session.call_tool(tool_name, arguments=arguments or {})
                            results[server_name] = result
                            print(f"✅ Called '{tool_name}' on '{server_name}'")
                        else:
                            print(f"⚠️  Tool '{tool_name}' not found on '{server_name}'")

            except Exception as e:
                print(f"❌ Failed to call tool on '{server_name}': {e}")
                results[server_name] = {"error": str(e)}

        return results


def main():
    parser = argparse.ArgumentParser(description="MCP Server Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add server
    add_parser = subparsers.add_parser("add", help="Add a new MCP server")
    add_parser.add_argument("name", help="Server name")
    add_parser.add_argument("command", help="Command to run the server")
    add_parser.add_argument("args", nargs="*", help="Command arguments")
    add_parser.add_argument("--env", help="Environment variables as JSON")

    # Remove server
    remove_parser = subparsers.add_parser("remove", help="Remove an MCP server")
    remove_parser.add_argument("name", help="Server name")

    # List servers
    subparsers.add_parser("list", help="List configured servers")

    # Test server
    test_parser = subparsers.add_parser("test", help="Test server connection")
    test_parser.add_argument("name", help="Server name")

    # Call tool
    call_parser = subparsers.add_parser("call", help="Call a tool across servers")
    call_parser.add_argument("tool", help="Tool name")
    call_parser.add_argument("--args", help="Tool arguments as JSON")

    # Config file
    parser.add_argument("--config", "-c", help="Path to config file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = MCPServerManager(args.config)

    if args.command == "add":
        env = json.loads(args.env) if args.env else {}
        manager.add_server(args.name, args.command, args.args, env)

    elif args.command == "remove":
        manager.remove_server(args.name)

    elif args.command == "list":
        manager.list_servers()

    elif args.command == "test":
        asyncio.run(manager.test_server(args.name))

    elif args.command == "call":
        arguments = json.loads(args.args) if args.args else {}
        results = asyncio.run(manager.call_tool_across_servers(args.tool, arguments))
        print("\n📊 Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()