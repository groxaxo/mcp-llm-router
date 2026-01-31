# MCP LLM Router

A Model Context Protocol (MCP) server for routing LLM requests across multiple providers and connecting to other MCP servers.

## Features

- **Multi-Provider LLM Routing**: Route requests to OpenAI, OpenRouter, DeepInfra, and other OpenAI-compatible APIs
- **Configurable \"Brain\" Model**: Pick DeepSeek reasoning or any OpenAI-compatible model as the main router brain
- **Session Management**: Track agent sessions with goals, constraints, and event logging
- **Workflow Gating (Judge Integration)**: Optional in-process mcp-as-a-judge tools for plan -> code -> test -> completion gating
- **Local Memory Indexing**: Embeddings + optional reranking for retrieval (Ollama or OpenAI-compatible endpoints)
- **MCP Server Orchestration**: Connect to and orchestrate multiple MCP servers
- **Cross-Server Tool Calling**: Call tools across different MCP servers
- **Universal MCP Compatibility**: Works with any MCP-compatible client (not tied to specific IDEs)

## Installation

1. Clone or navigate to this directory:
```bash
cd ~/mcp-llm-router
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -U pip
pip install -e .
```

## Configuration

### MCP Server Configuration (`mcp-config.json`)

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "python",
      "args": ["-m", "mcp_llm_router.server"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "DEEPINFRA_API_KEY": "your-deepinfra-key",
        "OPENROUTER_API_KEY": "your-openrouter-key"
      }
    },
    "other-server": {
      "command": "python",
      "args": ["-m", "other_mcp_server"],
      "env": {}
    }
}
}
```

### Example Config + Demo

- `examples/mcp-config.deepseek-ollama.json` - DeepSeek brain + Ollama embeddings + judge history persistence.
- `examples/demo_judge_gating.py` - End-to-end demo that indexes memory and walks a task through judge gating via `router_chat`.

Run the demo:

```bash
python examples/demo_judge_gating.py --config examples/mcp-config.deepseek-ollama.json
```

Note: the demo skips `request_plan_approval` because it requires user elicitation. Ensure `DEEPSEEK_API_KEY` (or `LLM_API_KEY`) is set and Ollama is running for embeddings.

### Environment Variables

Set API keys in your environment or in the config:

```bash
export OPENAI_API_KEY="sk-proj-..."
export DEEPINFRA_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-..."
export DEEPSEEK_API_KEY="..."
```

### Brain Configuration (Router LLM)

```bash
# Core brain settings
export ROUTER_BRAIN_MODEL="deepseek-reasoner"
export ROUTER_BRAIN_PROVIDER="deepseek"
export ROUTER_BRAIN_API_KEY_ENV="DEEPSEEK_API_KEY"

# Optional overrides
export ROUTER_BRAIN_BASE_URL="https://api.deepseek.com"
export ROUTER_BRAIN_MAX_TOKENS="4000"
export ROUTER_BRAIN_TEMPERATURE="0.2"
```

You can also set the brain per session using the `configure_brain` tool.

### Memory Configuration (Embeddings + Rerank)

```bash
# Storage paths
export MCP_ROUTER_DATA_DIR="./.mcp-llm-router"
export MCP_ROUTER_MEMORY_DB="./.mcp-llm-router/memory.db"

# Embeddings (OpenAI-compatible)
export EMBEDDINGS_PROVIDER="openai"
export EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
export EMBEDDINGS_MODEL="text-embedding-3-small"
export EMBEDDINGS_API_KEY_ENV="OPENAI_API_KEY"
export EMBEDDINGS_PATH="/embeddings"

# Embeddings (Ollama example)
export EMBEDDINGS_PROVIDER="ollama"
export EMBEDDINGS_BASE_URL="http://localhost:11434"
export EMBEDDINGS_MODEL="nomic-embed-text"
export EMBEDDINGS_PATH="/api/embeddings"

# Rerank (OpenAI-compatible LLM rerank)
export RERANK_PROVIDER="openai"
export RERANK_BASE_URL="https://api.openai.com/v1"
export RERANK_MODEL="gpt-4o-mini"
export RERANK_API_KEY_ENV="OPENAI_API_KEY"
export RERANK_PATH="/chat/completions"
export RERANK_MODE="llm"
```

### Judge Persistence (mcp-as-a-judge)

```bash
# Persist judge conversation history + task metadata
export MCP_JUDGE_DATABASE_URL="sqlite:///./.mcp-llm-router/judge_history.db"
```

## Usage

### Running MCP Servers

#### Using the Server Runner
```bash
# List configured servers
python mcp_server_runner.py list

# Run a specific server
python mcp_server_runner.py run llm-router
```

#### Using the Server Manager
```bash
# Add a new server
python mcp_manager.py add my-server python -m my_mcp_server

# List servers
python mcp_manager.py list

# Test server connection
python mcp_manager.py test llm-router

# Remove a server
python mcp_manager.py remove my-server
```

### Connecting to MCP Servers

#### Using the MCP Client
```bash
# List tools on a server
python mcp_client.py list-tools llm-router

# Call a tool on a server
python mcp_client.py call-tool llm-router start_session '{"goal": "Test session"}'
```

#### Using the Server Manager for Cross-Server Operations
```bash
# Call a tool across all configured servers
python mcp_manager.py call start_session '{"goal": "Test all servers"}'
```

## MCP Tools Available

### Session Management
- `start_session(goal, constraints, context, metadata)` - Start a new agent session
- `log_event(session_id, kind, message, details)` - Log events to a session
- `get_session_context(session_id)` - Retrieve full session data

### LLM Routing
- `agent_llm_request(session_id, prompt, model, base_url, api_key_env, ...)` - Route to LLM providers
- `configure_brain(...)` - Set the global or per-session brain model/settings
- `get_brain_config(session_id)` - Read the active brain configuration
- `router_chat(session_id, message, ...)` - Main brain chat (memory + workflow guidance)

### Memory (Embeddings + Rerank)
- `configure_memory(...)` - Set embedding/rerank configuration globally or per-session
- `memory_index(namespace, texts, metadatas, doc_ids)` - Index texts into memory
- `memory_search(namespace, query, top_k, rerank)` - Retrieve relevant memory hits
- `memory_delete(namespace, doc_id)` - Delete one doc or a whole namespace
- `memory_list_namespaces()` - List namespaces
- `memory_stats()` - Show memory counts

### MCP Server Orchestration
- `connect_mcp_server(server_name, command, args, env)` - Configure connection to another MCP server
- `list_mcp_servers()` - List configured MCP server connections
- `call_mcp_tool(server_name, tool_name, arguments)` - Call tools on other MCP servers
- `list_mcp_tools(server_name)` - List tools available on another MCP server

### Judge Tools (Optional, from mcp-as-a-judge)
- `set_coding_task(...)`
- `get_current_coding_task()`
- `request_plan_approval(...)`
- `judge_coding_plan(...)`
- `judge_code_change(...)`
- `judge_testing_implementation(...)`
- `judge_coding_task_completion(...)`
- `raise_obstacle(...)`
- `raise_missing_requirements(...)`

## Integration with MCP Clients

### Any MCP-Compatible Client

The server works with any client that supports the MCP protocol:

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "python",
      "args": ["-m", "mcp_llm_router.server"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### Example: Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "python",
      "args": ["-m", "mcp_llm_router.server"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DEEPINFRA_API_KEY": "..."
      }
    }
  }
}
```

### Example: Custom MCP Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_llm_router.server"],
        env={"OPENAI_API_KEY": "your-key"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Start a session
            result = await session.call_tool("start_session", {
                "goal": "Test the MCP server"
            })
            print("Session started:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider Configuration

### OpenAI
```python
{
  "base_url": null,  # Uses default
  "api_key_env": "OPENAI_API_KEY"
}
```

### OpenRouter
```python
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY"
}
```

### DeepInfra
```python
{
  "base_url": "https://api.deepinfra.com/v1/openai",
  "api_key_env": "DEEPINFRA_API_KEY"
}
```

## CLI Tool

The `opencode` command provides direct CLI access:

```bash
# Basic usage
opencode run "What is Python"

# Use specific provider
opencode run "Explain Docker" --provider deepinfra --model meta-llama/Meta-Llama-3.1-70B-Instruct
```

## Development

### Running the Server Directly
```bash
cd ~/mcp-llm-router
source .venv/bin/activate
python -m mcp_llm_router.server
```

### Testing
```bash
# Test server startup
timeout 5 python -m mcp_llm_router.server

# Test CLI
opencode run "Hello world"

# Test MCP client
python mcp_client.py list-tools llm-router
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   MCP Client    │◄──►│  LLM Router MCP  │
│ (Claude, etc.)  │    │     Server       │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  LLM Providers   │
                       │ • OpenAI         │
                       │ • OpenRouter     │
                       │ • DeepInfra      │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Other MCP Servers│
                       │ • File system    │
                       │ • Database       │
                       │ • APIs           │
                       └──────────────────┘
```

## License

MIT License - see LICENSE file for details.

```bash
# Basic usage with OpenAI (default)
opencode run "Explain quantum computing"

# Use a specific provider
opencode run "Write a Python function" --provider openrouter --model anthropic/claude-3-opus

# Use DeepInfra
opencode run "Summarize this text" --provider deepinfra --model meta-llama/Llama-3.1-70B-Instruct
```

**Available providers:**
- `openai` (default) - Uses OPENAI_API_KEY
- `openrouter` - Uses OPENROUTER_API_KEY
- `deepinfra` - Uses DEEPINFRA_API_KEY

## MCP Tools

When used as an MCP server in Antigravity, the following tools are available:

### start_session
Start a new agent session with a goal and constraints.

```python
{
  "goal": "Implement user authentication",
  "constraints": "Use JWT tokens, no external dependencies",
  "context": "FastAPI application"
}
```

### log_event
Log events during an agent session (info, error, warning, success).

```python
{
  "session_id": "uuid-here",
  "kind": "error",
  "message": "Build failed",
  "details": {"exit_code": 1}
}
```

### agent_llm_request
Make a request to an LLM provider within a session.

```python
{
  "session_id": "uuid-here",
  "prompt": "How do I fix this error?",
  "model": "gpt-4",
  "base_url": "https://openrouter.ai/api/v1",  # optional
  "api_key_env": "OPENROUTER_API_KEY"
}
```

### get_session_context
Retrieve full session history and events.

```python
{
  "session_id": "uuid-here"
}
```

## Example Agent Workflow in Antigravity

1. **Start session:**
   ```
   Call start_session with goal="Build a REST API for task management"
   ```

2. **Work on task:**
   ```
   Create files, run commands, etc.
   ```

3. **Log progress:**
   ```
   Call log_event with kind="info", message="Created database schema"
   ```

4. **When stuck:**
   ```
   Call agent_llm_request with prompt="How do I handle authentication?"
   ```

5. **Review context:**
   ```
   Call get_session_context to see full history
   ```

## Development

Run the MCP server directly:
```bash
cd ~/mcp-llm-router
source .venv/bin/activate
python -m mcp_llm_router.server
```

## Environment Variables

Set these in your `~/.bashrc` or Antigravity config:

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
export DEEPINFRA_API_KEY="..."
```
