"""MCP LLM Router Server - Routes requests to multiple LLM providers and MCP servers."""

import os
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Initialize FastMCP server
mcp = FastMCP("llm-router")

# In-memory session storage (in production, use a real database)
sessions: Dict[str, Dict[str, Any]] = {}

# MCP server connections
mcp_connections: Dict[str, Dict[str, Any]] = {}

# Provider base URL mappings (can be overridden by env vars)
PROVIDER_BASE_URLS = {
    "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "deepinfra": os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    "anthropic": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
    "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
}


@mcp.tool()
def start_session(
    goal: str,
    constraints: Optional[str] = None,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start a new agent session with a goal and optional constraints.

    Args:
        goal: The primary objective of this session
        constraints: Optional constraints or requirements
        context: Optional additional context
        metadata: Optional metadata dictionary

    Returns:
        Session details including session_id
    """
    import uuid

    session_id = str(uuid.uuid4())

    session_data = {
        "session_id": session_id,
        "goal": goal,
        "constraints": constraints,
        "context": context,
        "metadata": metadata or {},
        "events": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    sessions[session_id] = session_data

    return {
        "session_id": session_id,
        "status": "active",
        "message": f"Session started with goal: {goal}",
    }


def _log_event(
    session_id: str, kind: str, message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal implementation of log_event."""
    if session_id not in sessions:
        return {"success": False, "error": f"Session {session_id} not found"}

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "kind": kind,
        "message": message,
        "details": details or {},
    }

    sessions[session_id]["events"].append(event)
    sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()

    return {"success": True, "event": event, "session_id": session_id}


@mcp.tool()
def log_event(
    session_id: str, kind: str, message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log an event to a session.

    Args:
        session_id: The session ID to log to
        kind: Event type (info, error, warning, success, etc.)
        message: Event message
        details: Optional event details

    Returns:
        Confirmation of logged event
    """
    return _log_event(session_id, kind, message, details)


def _get_api_key(name: str) -> Optional[str]:
    """Get API key from environment or fallback to .bashrc parsing."""
    # 1. Try environment
    val = os.getenv(name)
    if val and val not in ("", "YOUR_DEEPSEEK_API_KEY", "YOUR_OPENAI_API_KEY"):
        return val

    # 2. Try .bashrc fallback
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        try:
            with open(bashrc_path, "r") as f:
                import re

                content = f.read()
                # Match export NAME="VAL" or export NAME=VAL
                pattern = rf'^export\s+{name}=["\']?([^"\'\s#]+)["\']?'
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    return match.group(1)
        except Exception:
            pass

    return None


@mcp.tool()
def agent_llm_request(
    session_id: str,
    prompt: str,
    model: str,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    provider: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make a request to an LLM provider (OpenAI-compatible API).

    Args:
        session_id: Session ID for context tracking
        prompt: The prompt to send to the LLM
        model: Model identifier (e.g., "gpt-4", "anthropic/claude-3-opus")
        base_url: Base URL for the API (optional, overrides provider)
        api_key_env: Environment variable name containing the API key
        provider: Provider name (openai, openrouter, deepinfra, anthropic) for auto base_url
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt

    Returns:
        LLM response with content and metadata

    Examples:
        # OpenAI (default)
        agent_llm_request(session_id, "Hello", "gpt-4", api_key_env="OPENAI_API_KEY")

        # OpenRouter using provider
        agent_llm_request(session_id, "Hello", "anthropic/claude-3-opus",
                         provider="openrouter", api_key_env="OPENROUTER_API_KEY")

        # DeepInfra using provider
        agent_llm_request(session_id, "Hello", "meta-llama/Llama-2-70b-chat-hf",
                         provider="deepinfra", api_key_env="DEEPINFRA_API_KEY")
    """
    if session_id not in sessions:
        return {"success": False, "error": f"Session {session_id} not found"}

    # Automatic routing for DeepSeek models
    if (
        (model.startswith("deepseek-") or "deepseek" in model)
        and provider is None
        and base_url is None
    ):
        provider = "deepseek"
        if api_key_env == "OPENAI_API_KEY":  # Only switch if user hasn't overridden
            api_key_env = "DEEPSEEK_API_KEY"

    # Get API key (with bashrc fallback)
    api_key = _get_api_key(api_key_env)
    if not api_key:
        return {
            "success": False,
            "error": f"API key not found in environment or .bashrc: {api_key_env}",
        }

    # Determine base URL
    if base_url is None:
        if provider and provider.lower() in PROVIDER_BASE_URLS:
            base_url = PROVIDER_BASE_URLS[provider.lower()]
        else:
            # Default to OpenAI
            base_url = PROVIDER_BASE_URLS["openai"]

    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Make request
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]

        # Log this request in the session
        _log_event(
            session_id=session_id,
            kind="llm_request",
            message=f"LLM request to {model}",
            details={
                "model": model,
                "base_url": base_url,
                "prompt_length": len(prompt),
                "response_length": len(content),
            },
        )

        return {
            "success": True,
            "content": content,
            "model": model,
            "usage": result.get("usage", {}),
            "session_id": session_id,
        }

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
        _log_event(
            session_id=session_id,
            kind="error",
            message=f"LLM request failed: {error_msg}",
            details={"model": model, "base_url": base_url},
        )
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = str(e)
        _log_event(
            session_id=session_id,
            kind="error",
            message=f"LLM request failed: {error_msg}",
            details={"model": model, "base_url": base_url},
        )
        return {"success": False, "error": error_msg}


@mcp.tool()
def get_session_context(session_id: str) -> Dict[str, Any]:
    """
    Retrieve the full context of a session.

    Args:
        session_id: The session ID to retrieve

    Returns:
        Complete session data including goal, events, and metadata
    """
    if session_id not in sessions:
        return {"success": False, "error": f"Session {session_id} not found"}

    return {"success": True, "session": sessions[session_id]}


@mcp.tool()
def connect_mcp_server(
    server_name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Connect to another MCP server.

    Args:
        server_name: Name to identify this MCP server connection
        command: Command to run the MCP server
        args: Arguments for the command
        env: Environment variables for the server

    Returns:
        Connection status
    """
    try:
        server_params = StdioServerParameters(
            command=command, args=args or [], env=dict(os.environ) | (env or {})
        )

        # Store connection info
        mcp_connections[server_name] = {
            "command": command,
            "args": args or [],
            "env": env or {},
            "connected": False,
            "server_params": server_params,
        }

        return {
            "success": True,
            "server_name": server_name,
            "message": f"MCP server '{server_name}' configured for connection",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to configure MCP server connection: {str(e)}",
        }


@mcp.tool()
def list_mcp_servers() -> Dict[str, Any]:
    """
    List all configured MCP server connections.

    Returns:
        List of MCP server connections
    """
    return {
        "servers": [
            {
                "name": name,
                "command": info["command"],
                "args": info["args"],
                "connected": info["connected"],
            }
            for name, info in mcp_connections.items()
        ]
    }


async def call_mcp_tool(
    server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Call a tool on a connected MCP server.

    Args:
        server_name: Name of the MCP server to call
        tool_name: Name of the tool to call
        arguments: Arguments for the tool call

    Returns:
        Tool call result
    """
    if server_name not in mcp_connections:
        return {"success": False, "error": f"MCP server '{server_name}' not found"}

    server_info = mcp_connections[server_name]

    try:
        async with stdio_client(server_info["server_params"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Call the tool
                result = await session.call_tool(tool_name, arguments=arguments or {})

                return {
                    "success": True,
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": result,
                }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to call tool '{tool_name}' on server '{server_name}': {str(e)}",
        }


async def list_mcp_tools(server_name: str) -> Dict[str, Any]:
    """
    List available tools on a connected MCP server.

    Args:
        server_name: Name of the MCP server

    Returns:
        List of available tools
    """
    if server_name not in mcp_connections:
        return {"success": False, "error": f"MCP server '{server_name}' not found"}

    server_info = mcp_connections[server_name]

    try:
        async with stdio_client(server_info["server_params"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools = await session.list_tools()

                return {
                    "success": True,
                    "server_name": server_name,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tools
                    ],
                }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list tools on server '{server_name}': {str(e)}",
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
