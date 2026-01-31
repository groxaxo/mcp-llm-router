"""MCP LLM Router Server - routes requests to LLM providers and MCP servers."""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import Context, FastMCP

from mcp_llm_router.brain import BrainClient, BrainConfig, DEFAULT_PROVIDER_BASE_URLS
from mcp_llm_router.memory import (
    EmbeddingConfig,
    MemorySettings,
    MemoryStore,
    RerankConfig,
    embed_texts,
    rerank_documents,
)
from mcp_llm_router import judge_bridge

# Initialize FastMCP server
mcp = FastMCP("llm-router")

# In-memory session storage (in production, use a real database)
sessions: Dict[str, Dict[str, Any]] = {}

# MCP server connections configuration
mcp_server_configs: Dict[str, Dict[str, Any]] = {}

# Data directory + persistence settings
DATA_DIR = os.getenv(
    "MCP_ROUTER_DATA_DIR", os.path.join(os.getcwd(), ".mcp-llm-router")
)
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure judge DB persists if mcp-as-a-judge is used
os.environ.setdefault(
    "MCP_JUDGE_DATABASE_URL",
    f"sqlite:///{os.path.join(DATA_DIR, 'judge_history.db')}",
)

MEMORY_DB_PATH = os.getenv(
    "MCP_ROUTER_MEMORY_DB", os.path.join(DATA_DIR, "memory.db")
)

memory_store = MemoryStore(MEMORY_DB_PATH)
brain_client = BrainClient(DEFAULT_PROVIDER_BASE_URLS)


def _default_brain_config() -> BrainConfig:
    return BrainConfig(
        model=os.getenv("ROUTER_BRAIN_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("ROUTER_BRAIN_BASE_URL"),
        api_key_env=os.getenv("ROUTER_BRAIN_API_KEY_ENV", "OPENAI_API_KEY"),
        provider=os.getenv("ROUTER_BRAIN_PROVIDER"),
        max_tokens=int(os.getenv("ROUTER_BRAIN_MAX_TOKENS", "2000")),
        temperature=float(os.getenv("ROUTER_BRAIN_TEMPERATURE", "0.7")),
        system_prompt=os.getenv("ROUTER_BRAIN_SYSTEM_PROMPT"),
        reasoning_effort=os.getenv("ROUTER_BRAIN_REASONING_EFFORT"),
    )


def _default_memory_settings() -> MemorySettings:
    return MemorySettings(embedding=EmbeddingConfig(), rerank=RerankConfig())


def _brain_config_from_dict(
    data: Optional[Dict[str, Any]], fallback: BrainConfig
) -> BrainConfig:
    if not data:
        return fallback

    merged = {
        "model": data.get("model", fallback.model),
        "base_url": data.get("base_url", fallback.base_url),
        "api_key_env": data.get("api_key_env", fallback.api_key_env),
        "provider": data.get("provider", fallback.provider),
        "max_tokens": data.get("max_tokens", fallback.max_tokens),
        "temperature": data.get("temperature", fallback.temperature),
        "system_prompt": data.get("system_prompt", fallback.system_prompt),
        "reasoning_effort": data.get("reasoning_effort", fallback.reasoning_effort),
        "extra_headers": fallback.extra_headers,
        "extra_body": fallback.extra_body,
        "timeout_s": data.get("timeout_s", fallback.timeout_s),
    }

    if data.get("extra_headers"):
        merged["extra_headers"] = {
            **(fallback.extra_headers or {}),
            **data["extra_headers"],
        }
    if data.get("extra_body"):
        merged["extra_body"] = {**(fallback.extra_body or {}), **data["extra_body"]}

    return BrainConfig(**merged)


def _memory_settings_from_dict(
    data: Optional[Dict[str, Any]], fallback: MemorySettings
) -> MemorySettings:
    if not data:
        return fallback

    embedding_data = data.get("embedding", {}) if isinstance(data, dict) else {}
    rerank_data = data.get("rerank", {}) if isinstance(data, dict) else {}

    embedding = EmbeddingConfig(
        provider=embedding_data.get("provider", fallback.embedding.provider),
        base_url=embedding_data.get("base_url", fallback.embedding.base_url),
        api_key_env=embedding_data.get("api_key_env", fallback.embedding.api_key_env),
        model=embedding_data.get("model", fallback.embedding.model),
        path=embedding_data.get("path", fallback.embedding.path),
        timeout_s=embedding_data.get("timeout_s", fallback.embedding.timeout_s),
    )
    rerank = RerankConfig(
        provider=rerank_data.get("provider", fallback.rerank.provider),
        base_url=rerank_data.get("base_url", fallback.rerank.base_url),
        api_key_env=rerank_data.get("api_key_env", fallback.rerank.api_key_env),
        model=rerank_data.get("model", fallback.rerank.model),
        path=rerank_data.get("path", fallback.rerank.path),
        mode=rerank_data.get("mode", fallback.rerank.mode),
        temperature=rerank_data.get("temperature", fallback.rerank.temperature),
        max_tokens=rerank_data.get("max_tokens", fallback.rerank.max_tokens),
        timeout_s=rerank_data.get("timeout_s", fallback.rerank.timeout_s),
    )

    return MemorySettings(embedding=embedding, rerank=rerank)


def _brain_config_to_dict(config: BrainConfig) -> Dict[str, Any]:
    return {
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "provider": config.provider,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system_prompt": config.system_prompt,
        "reasoning_effort": config.reasoning_effort,
        "extra_headers": config.extra_headers,
        "extra_body": config.extra_body,
        "timeout_s": config.timeout_s,
    }


def _memory_settings_to_dict(settings: MemorySettings) -> Dict[str, Any]:
    return {
        "embedding": {
            "provider": settings.embedding.provider,
            "base_url": settings.embedding.base_url,
            "api_key_env": settings.embedding.api_key_env,
            "model": settings.embedding.model,
            "path": settings.embedding.path,
            "timeout_s": settings.embedding.timeout_s,
        },
        "rerank": {
            "provider": settings.rerank.provider,
            "base_url": settings.rerank.base_url,
            "api_key_env": settings.rerank.api_key_env,
            "model": settings.rerank.model,
            "path": settings.rerank.path,
            "mode": settings.rerank.mode,
            "temperature": settings.rerank.temperature,
            "max_tokens": settings.rerank.max_tokens,
            "timeout_s": settings.rerank.timeout_s,
        },
    }


def _get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return sessions.get(session_id)


def _get_session_brain_config(session_id: Optional[str]) -> BrainConfig:
    default = DEFAULT_BRAIN_CONFIG
    if not session_id:
        return default

    session = _get_session(session_id)
    if not session:
        return default

    return _brain_config_from_dict(session.get("brain_config"), default)


def _get_session_memory_settings(session_id: Optional[str]) -> MemorySettings:
    default = DEFAULT_MEMORY_SETTINGS
    if not session_id:
        return default

    session = _get_session(session_id)
    if not session:
        return default

    return _memory_settings_from_dict(session.get("memory_settings"), default)


class MCPConnectionManager:
    def __init__(self) -> None:
        # Maps server_name -> {'session': ClientSession, 'stack': AsyncExitStack}
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def get_session(
        self, server_name: str, server_params: StdioServerParameters
    ) -> ClientSession:
        """Get an existing session or create a new one."""
        if server_name in self.active_connections:
            return self.active_connections[server_name]["session"]

        stack = contextlib.AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self.active_connections[server_name] = {"session": session, "stack": stack}
            return session
        except Exception:
            await stack.aclose()
            raise

    async def close_connection(self, server_name: str) -> None:
        if server_name in self.active_connections:
            await self.active_connections[server_name]["stack"].aclose()
            del self.active_connections[server_name]


connection_manager = MCPConnectionManager()

# Global defaults
DEFAULT_BRAIN_CONFIG = _default_brain_config()
DEFAULT_MEMORY_SETTINGS = _default_memory_settings()


@mcp.tool()
def start_session(
    goal: str,
    constraints: Optional[str] = None,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    brain_config: Optional[Dict[str, Any]] = None,
    memory_settings: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start a new agent session with a goal and optional constraints.
    """
    import uuid

    session_id = str(uuid.uuid4())

    session_data: Dict[str, Any] = {
        "session_id": session_id,
        "goal": goal,
        "constraints": constraints,
        "context": context,
        "metadata": metadata or {},
        "events": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "brain_config": brain_config or {},
        "memory_settings": memory_settings or {},
        "task_id": task_id,
    }

    sessions[session_id] = session_data

    return {
        "session_id": session_id,
        "status": "active",
        "message": f"Session started with goal: {goal}",
    }


@mcp.tool()
def configure_brain(
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Configure the default or session-specific brain model."""
    global DEFAULT_BRAIN_CONFIG

    base_config = (
        _get_session_brain_config(session_id) if session_id else DEFAULT_BRAIN_CONFIG
    )

    overrides = {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "provider": provider,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "reasoning_effort": reasoning_effort,
        "timeout_s": timeout_s,
    }

    merged = _brain_config_from_dict(
        {k: v for k, v in overrides.items() if v is not None}, base_config
    )
    if extra_headers:
        merged.extra_headers = {**(merged.extra_headers or {}), **extra_headers}
    if extra_body:
        merged.extra_body = {**(merged.extra_body or {}), **extra_body}

    if session_id:
        session = _get_session(session_id)
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        session["brain_config"] = _brain_config_to_dict(merged)
        session["updated_at"] = datetime.utcnow().isoformat()
        scope = "session"
    else:
        DEFAULT_BRAIN_CONFIG = merged
        scope = "global"

    return {
        "success": True,
        "scope": scope,
        "brain_config": _brain_config_to_dict(merged),
    }


@mcp.tool()
def get_brain_config(session_id: Optional[str] = None) -> Dict[str, Any]:
    config = _get_session_brain_config(session_id)
    return {
        "success": True,
        "scope": "session" if session_id else "global",
        "brain_config": _brain_config_to_dict(config),
    }


@mcp.tool()
def configure_memory(
    session_id: Optional[str] = None,
    embedding: Optional[Dict[str, Any]] = None,
    rerank: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Configure memory embedding/rerank settings (global or per-session)."""
    global DEFAULT_MEMORY_SETTINGS

    base_settings = (
        _get_session_memory_settings(session_id)
        if session_id
        else DEFAULT_MEMORY_SETTINGS
    )
    merged = _memory_settings_from_dict(
        {
            "embedding": embedding or {},
            "rerank": rerank or {},
        },
        base_settings,
    )

    if session_id:
        session = _get_session(session_id)
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        session["memory_settings"] = _memory_settings_to_dict(merged)
        session["updated_at"] = datetime.utcnow().isoformat()
        scope = "session"
    else:
        DEFAULT_MEMORY_SETTINGS = merged
        scope = "global"

    return {
        "success": True,
        "scope": scope,
        "memory_settings": _memory_settings_to_dict(merged),
    }


@mcp.tool()
def link_task(session_id: str, task_id: str) -> Dict[str, Any]:
    session = _get_session(session_id)
    if not session:
        return {"success": False, "error": f"Session {session_id} not found"}

    session["task_id"] = task_id
    session["updated_at"] = datetime.utcnow().isoformat()

    return {"success": True, "session_id": session_id, "task_id": task_id}


def _log_event(
    session_id: str, kind: str, message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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
    """Log an event to a session."""
    return _log_event(session_id, kind, message, details)


@mcp.tool()
def get_session_context(session_id: str) -> Dict[str, Any]:
    """Retrieve the full context of a session."""
    if session_id not in sessions:
        return {"success": False, "error": f"Session {session_id} not found"}

    session = dict(sessions[session_id])
    if session.get("brain_config"):
        session["brain_config"] = _brain_config_from_dict(
            session["brain_config"], DEFAULT_BRAIN_CONFIG
        )
        session["brain_config"] = _brain_config_to_dict(session["brain_config"])
    if session.get("memory_settings"):
        session["memory_settings"] = _memory_settings_to_dict(
            _memory_settings_from_dict(session["memory_settings"], DEFAULT_MEMORY_SETTINGS)
        )

    return {"success": True, "session": session}


@mcp.tool()
async def agent_llm_request(
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
    """Make a request to an OpenAI-compatible LLM provider."""
    if session_id not in sessions:
        return {"success": False, "error": f"Session {session_id} not found"}

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    config = BrainConfig(
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    try:
        result = await brain_client.chat(messages, config)
        content = result["choices"][0]["message"]["content"]

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

    except Exception as exc:
        error_msg = str(exc)
        _log_event(
            session_id=session_id,
            kind="error",
            message=f"LLM request failed: {error_msg}",
            details={"model": model, "base_url": base_url},
        )
        return {"success": False, "error": error_msg}


@mcp.tool()
async def memory_index(
    namespace: str,
    texts: List[str],
    session_id: Optional[str] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Index texts into the local memory store using embeddings."""
    if not texts:
        return {"success": False, "error": "No texts provided"}

    settings = _get_session_memory_settings(session_id)
    embeddings = await embed_texts(texts, settings.embedding)
    if len(embeddings) != len(texts):
        return {
            "success": False,
            "error": "Embedding response count does not match input text count",
        }

    docs: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        docs.append(
            {
                "doc_id": doc_ids[idx] if doc_ids and idx < len(doc_ids) else None,
                "content": text,
                "metadata": metadatas[idx] if metadatas and idx < len(metadatas) else {},
                "embedding": embeddings[idx],
            }
        )

    result = await memory_store.upsert_documents(namespace, docs)
    return {
        "success": True,
        "namespace": namespace,
        **result,
    }


@mcp.tool()
async def memory_search(
    namespace: str,
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Search memory with embeddings and optional reranking."""
    settings = _get_session_memory_settings(session_id)
    query_embedding = await embed_texts([query], settings.embedding)
    if not query_embedding:
        return {"success": False, "error": "Embedding response was empty"}
    hits = await memory_store.search(namespace, query_embedding[0], top_k=top_k)

    if rerank and settings.rerank.provider.lower() != "none":
        hits = await rerank_documents(query, hits, settings.rerank)

    return {"success": True, "namespace": namespace, "results": hits}


@mcp.tool()
async def memory_delete(
    namespace: str,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete a document or entire namespace from memory."""
    if doc_id:
        deleted = await memory_store.delete_document(namespace, doc_id)
        return {"success": True, "deleted": deleted}

    deleted = await memory_store.delete_namespace(namespace)
    return {"success": True, "deleted": deleted}


@mcp.tool()
async def memory_list_namespaces() -> Dict[str, Any]:
    namespaces = await memory_store.list_namespaces()
    return {"success": True, "namespaces": namespaces}


@mcp.tool()
async def memory_stats() -> Dict[str, Any]:
    stats = await memory_store.stats()
    return {"success": True, **stats}


@mcp.tool()
async def router_chat(
    session_id: str,
    message: str,
    ctx: Context,
    task_id: Optional[str] = None,
    memory_namespace: Optional[str] = None,
    use_memory: bool = True,
    top_k: int = 5,
    rerank: bool = True,
    system_prompt: Optional[str] = None,
    brain_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main brain chat tool with optional memory + workflow guidance."""
    session = _get_session(session_id)
    if not session:
        return {"success": False, "error": f"Session {session_id} not found"}

    effective_task_id = task_id or session.get("task_id")
    brain_config = _get_session_brain_config(session_id)
    if brain_override:
        brain_config = _brain_config_from_dict(brain_override, brain_config)

    memory_hits: List[Dict[str, Any]] = []
    if use_memory and memory_namespace:
        settings = _get_session_memory_settings(session_id)
        query_embedding = await embed_texts([message], settings.embedding)
        if query_embedding:
            memory_hits = await memory_store.search(
                memory_namespace, query_embedding[0], top_k=top_k
            )
        if rerank and settings.rerank.provider.lower() != "none":
            memory_hits = await rerank_documents(message, memory_hits, settings.rerank)

    workflow_guidance = None
    task_metadata = None
    conversation_service = None
    history_payload = None

    if judge_bridge.judge_available():
        judge_server = judge_bridge.load_judge()
        conversation_service = judge_server.conversation_service
        if effective_task_id:
            try:
                from mcp_as_a_judge.tasks.manager import load_task_metadata_from_history
                from mcp_as_a_judge.workflow.workflow_guidance import (
                    calculate_next_stage,
                )

                task_metadata = await load_task_metadata_from_history(
                    effective_task_id, conversation_service
                )
                if task_metadata is not None:
                    workflow_guidance = await calculate_next_stage(
                        task_metadata=task_metadata,
                        current_operation="router_chat",
                        conversation_service=conversation_service,
                        ctx=ctx,
                    )
            except Exception:
                workflow_guidance = None

    if conversation_service is not None:
        with contextlib.suppress(Exception):
            history_records = await conversation_service.load_filtered_context_for_enrichment(
                session_id=session_id,
                current_prompt=message,
                ctx=ctx,
            )
            history_payload = conversation_service.format_conversation_history_as_json_array(
                history_records
            )

    # Build system prompt
    system_parts: List[str] = []
    if brain_config.system_prompt:
        system_parts.append(brain_config.system_prompt)
    if system_prompt:
        system_parts.append(system_prompt)

    system_parts.append("You are the MCP router brain. Provide clear next steps.")
    system_parts.append(f"Session goal: {session.get('goal')}")
    if session.get("constraints"):
        system_parts.append(f"Constraints: {session.get('constraints')}")
    if session.get("context"):
        system_parts.append(f"Context: {session.get('context')}")

    if task_metadata is not None:
        system_parts.append("Current task metadata (JSON):")
        system_parts.append(task_metadata.model_dump_json())

    if workflow_guidance is not None:
        system_parts.append("Workflow guidance:")
        system_parts.append(workflow_guidance.model_dump_json())

    if history_payload:
        system_parts.append("Conversation history (JSON list):")
        system_parts.append(json.dumps(history_payload))

    if memory_hits:
        system_parts.append("Relevant memory hits (JSON list):")
        system_parts.append(json.dumps(memory_hits))

    system_combined = "\n\n".join(system_parts)

    messages = [
        {"role": "system", "content": system_combined},
        {"role": "user", "content": message},
    ]

    try:
        result = await brain_client.chat(messages, brain_config)
        content = result["choices"][0]["message"]["content"]
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    # Save to conversation history if judge is available
    if conversation_service is not None:
        with contextlib.suppress(Exception):
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=session_id,
                tool_name="router_chat",
                tool_input=json.dumps(
                    {
                        "message": message,
                        "task_id": effective_task_id,
                        "memory_namespace": memory_namespace,
                    }
                ),
                tool_output=json.dumps({"response": content}),
            )

    _log_event(
        session_id=session_id,
        kind="router_chat",
        message="Router brain response",
        details={
            "task_id": effective_task_id,
            "memory_namespace": memory_namespace,
            "model": brain_config.model,
        },
    )

    return {
        "success": True,
        "session_id": session_id,
        "task_id": effective_task_id,
        "content": content,
        "model": brain_config.model,
        "memory_hits": memory_hits,
        "workflow_guidance": workflow_guidance.model_dump(
            mode="json",
            exclude_unset=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        if workflow_guidance is not None
        else None,
    }


@mcp.tool()
def connect_mcp_server(
    server_name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Connect to another MCP server."""
    try:
        server_params = StdioServerParameters(
            command=command, args=args or [], env=dict(os.environ) | (env or {})
        )

        mcp_server_configs[server_name] = {
            "command": command,
            "args": args or [],
            "env": env or {},
            "server_params": server_params,
        }

        return {
            "success": True,
            "server_name": server_name,
            "message": f"MCP server '{server_name}' configured for connection",
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to configure MCP server connection: {str(exc)}",
        }


@mcp.tool()
def list_mcp_servers() -> Dict[str, Any]:
    """List all configured MCP server connections."""
    return {
        "servers": [
            {
                "name": name,
                "command": info["command"],
                "args": info["args"],
                "connected": name in connection_manager.active_connections,
            }
            for name, info in mcp_server_configs.items()
        ]
    }


@mcp.tool()
async def call_mcp_tool(
    server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Call a tool on a connected MCP server."""
    if server_name not in mcp_server_configs:
        return {"success": False, "error": f"MCP server '{server_name}' not found"}

    server_info = mcp_server_configs[server_name]

    try:
        session = await connection_manager.get_session(
            server_name, server_info["server_params"]
        )
        result = await session.call_tool(tool_name, arguments=arguments or {})

        return {
            "success": True,
            "server_name": server_name,
            "tool_name": tool_name,
            "result": result,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to call tool '{tool_name}' on server '{server_name}': {str(exc)}",
        }


@mcp.tool()
async def list_mcp_tools(server_name: str) -> Dict[str, Any]:
    """List available tools on a connected MCP server."""
    if server_name not in mcp_server_configs:
        return {"success": False, "error": f"MCP server '{server_name}' not found"}

    server_info = mcp_server_configs[server_name]

    try:
        session = await connection_manager.get_session(
            server_name, server_info["server_params"]
        )
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

    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to list tools on server '{server_name}': {str(exc)}",
        }


# --- Optional judge tool wrappers (mcp-as-a-judge) ---

def _judge_unavailable(tool_name: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": f"Judge tool '{tool_name}' unavailable - mcp-as-a-judge import failed.",
        "details": judge_bridge.judge_import_error(),
    }


@mcp.tool()
async def set_coding_task(
    user_request: str,
    task_title: str,
    task_description: str,
    ctx: Context,
    task_size: Any = None,
    task_id: str = "",
    user_requirements: str = "",
    state: Any = None,
    tags: List[str] = [],  # noqa: B006
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("set_coding_task")
    judge_server = judge_bridge.load_judge()
    if task_size is None:
        with contextlib.suppress(Exception):
            from mcp_as_a_judge.models.task_metadata import TaskSize

            task_size = TaskSize.M
        if task_size is None:
            task_size = "m"
    if state is None:
        with contextlib.suppress(Exception):
            from mcp_as_a_judge.models.task_metadata import TaskState

            state = TaskState.CREATED
        if state is None:
            state = "created"
    return await judge_server.set_coding_task(
        user_request=user_request,
        task_title=task_title,
        task_description=task_description,
        ctx=ctx,
        task_size=task_size,
        task_id=task_id,
        user_requirements=user_requirements,
        state=state,
        tags=tags,
    )


@mcp.tool()
async def get_current_coding_task(ctx: Context) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("get_current_coding_task")
    judge_server = judge_bridge.load_judge()
    return await judge_server.get_current_coding_task(ctx=ctx)


@mcp.tool()
async def request_plan_approval(
    plan: str,
    design: str,
    research: str,
    task_id: str,
    ctx: Context,
    research_urls: List[str] = [],  # noqa: B006
    problem_domain: str = "",
    problem_non_goals: List[str] = [],  # noqa: B006
    library_plan: List[Dict[str, Any]] = [],  # noqa: B006
    internal_reuse_components: List[Dict[str, Any]] = [],  # noqa: B006
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("request_plan_approval")
    judge_server = judge_bridge.load_judge()
    return await judge_server.request_plan_approval(
        plan=plan,
        design=design,
        research=research,
        task_id=task_id,
        ctx=ctx,
        research_urls=research_urls,
        problem_domain=problem_domain,
        problem_non_goals=problem_non_goals,
        library_plan=library_plan,
        internal_reuse_components=internal_reuse_components,
    )


@mcp.tool()
async def raise_obstacle(
    problem: str,
    research: str,
    options: List[str],
    ctx: Context,
    task_id: str = "",
    decision_area: str = "",
    constraints: List[str] = [],  # noqa: B006
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("raise_obstacle")
    judge_server = judge_bridge.load_judge()
    return await judge_server.raise_obstacle(
        problem=problem,
        research=research,
        options=options,
        ctx=ctx,
        task_id=task_id,
        decision_area=decision_area,
        constraints=constraints,
    )


@mcp.tool()
async def raise_missing_requirements(
    current_request: str,
    identified_gaps: List[str],
    specific_questions: List[str],
    task_id: str,
    ctx: Context,
    decision_areas: List[str] = [],  # noqa: B006
    options: List[str] = [],  # noqa: B006
    constraints: List[str] = [],  # noqa: B006
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("raise_missing_requirements")
    judge_server = judge_bridge.load_judge()
    return await judge_server.raise_missing_requirements(
        current_request=current_request,
        identified_gaps=identified_gaps,
        specific_questions=specific_questions,
        task_id=task_id,
        ctx=ctx,
        decision_areas=decision_areas,
        options=options,
        constraints=constraints,
    )


@mcp.tool()
async def judge_coding_task_completion(
    task_id: str,
    completion_summary: str,
    requirements_met: List[str],
    implementation_details: str,
    ctx: Context,
    remaining_work: List[str] = [],  # noqa: B006
    quality_notes: str = "",
    testing_status: str = "",
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("judge_coding_task_completion")
    judge_server = judge_bridge.load_judge()
    return await judge_server.judge_coding_task_completion(
        task_id=task_id,
        completion_summary=completion_summary,
        requirements_met=requirements_met,
        implementation_details=implementation_details,
        ctx=ctx,
        remaining_work=remaining_work,
        quality_notes=quality_notes,
        testing_status=testing_status,
    )


@mcp.tool()
async def judge_coding_plan(
    plan: str,
    design: str,
    research: str,
    research_urls: List[str],
    ctx: Context,
    task_id: str = "",
    context: str = "",
    user_requirements: str = "",
    problem_domain: str = "",
    problem_non_goals: List[str] = [],  # noqa: B006
    library_plan: List[Dict[str, Any]] = [],  # noqa: B006
    internal_reuse_components: List[Dict[str, Any]] = [],  # noqa: B006
    design_patterns: List[Dict[str, Any]] = [],  # noqa: B006
    identified_risks: List[str] = [],  # noqa: B006
    risk_mitigation_strategies: List[str] = [],  # noqa: B006
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("judge_coding_plan")
    judge_server = judge_bridge.load_judge()
    return await judge_server.judge_coding_plan(
        plan=plan,
        design=design,
        research=research,
        research_urls=research_urls,
        ctx=ctx,
        task_id=task_id,
        context=context,
        user_requirements=user_requirements,
        problem_domain=problem_domain,
        problem_non_goals=problem_non_goals,
        library_plan=library_plan,
        internal_reuse_components=internal_reuse_components,
        design_patterns=design_patterns,
        identified_risks=identified_risks,
        risk_mitigation_strategies=risk_mitigation_strategies,
    )


@mcp.tool()
async def judge_code_change(
    code_change: str,
    ctx: Context,
    file_path: str = "File path not specified",
    change_description: str = "Change description not provided",
    task_id: str = "",
    user_requirements: str = "",
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("judge_code_change")
    judge_server = judge_bridge.load_judge()
    return await judge_server.judge_code_change(
        code_change=code_change,
        ctx=ctx,
        file_path=file_path,
        change_description=change_description,
        task_id=task_id,
        user_requirements=user_requirements,
    )


@mcp.tool()
async def judge_testing_implementation(
    task_id: str,
    test_summary: str,
    test_files: List[str],
    test_execution_results: str,
    ctx: Context,
    test_coverage_report: str = "",
    test_types_implemented: List[str] = [],  # noqa: B006
    testing_framework: str = "",
    performance_test_results: str = "",
    manual_test_notes: str = "",
) -> Any:
    if not judge_bridge.judge_available():
        return _judge_unavailable("judge_testing_implementation")
    judge_server = judge_bridge.load_judge()
    return await judge_server.judge_testing_implementation(
        task_id=task_id,
        test_summary=test_summary,
        test_files=test_files,
        test_execution_results=test_execution_results,
        ctx=ctx,
        test_coverage_report=test_coverage_report,
        test_types_implemented=test_types_implemented,
        testing_framework=testing_framework,
        performance_test_results=performance_test_results,
        manual_test_notes=manual_test_notes,
    )


if __name__ == "__main__":
    mcp.run()
