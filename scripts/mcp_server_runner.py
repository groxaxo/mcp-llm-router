#!/usr/bin/env python3
"""MCP Server Runner - Run MCP servers with different configurations."""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load MCP configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


def run_server(server_name: str, config_path: str = None):
    """Run a specific MCP server from configuration."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "mcp-config.json")

    config = load_config(config_path)

    if "mcpServers" not in config:
        print("No mcpServers found in config")
        return

    if server_name not in config["mcpServers"]:
        print(f"Server '{server_name}' not found in config")
        available = list(config["mcpServers"].keys())
        print(f"Available servers: {available}")
        return

    server_config = config["mcpServers"][server_name]

    # Set environment variables
    env = os.environ.copy()
    if "env" in server_config:
        env.update(server_config["env"])

    # Run the server
    cmd = [server_config["command"]] + server_config.get("args", [])

    print(f"🚀 Starting MCP server: {server_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped MCP server: {server_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run MCP server: {e}")


def list_servers(config_path: str = None):
    """List all configured MCP servers."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "mcp-config.json")

    config = load_config(config_path)

    if "mcpServers" not in config:
        print("No mcpServers found in config")
        return

    print("📋 Configured MCP Servers:")
    for name, server_config in config["mcpServers"].items():
        cmd = [server_config["command"]] + server_config.get("args", [])
        print(f"  • {name}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="MCP Server Runner")
    parser.add_argument("action", choices=["run", "list"], help="Action to perform")
    parser.add_argument("server", nargs="?", help="Server name (for run action)")
    parser.add_argument("--config", "-c", help="Path to config file")

    args = parser.parse_args()

    if args.action == "list":
        list_servers(args.config)
    elif args.action == "run":
        if not args.server:
            print("Server name required for run action")
            sys.exit(1)
        run_server(args.server, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()