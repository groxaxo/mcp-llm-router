# MCP LLM Router

A Model Context Protocol (MCP) server for routing LLM requests across multiple providers and connecting to other MCP servers. **Designed with an "all-local except the brain" architecture** for privacy and control.

## Features (Unified Router + Judge)

- **One server, two roles**: `mcp_llm_router.server` now ships Judge tools in-process—no separate `mcp-as-a-judge` server required.
- **Multi-Provider LLM Routing**: Route requests to OpenAI, OpenRouter, DeepInfra, and other OpenAI-compatible APIs.
- **Configurable "Brain" Model**: Choose DeepSeek reasoning or any OpenAI-compatible model as the router brain.
- **Session Management**: Track agent sessions with goals, constraints, and event logging.
- **Quality Gating (Judge)**: Plan → code → test → completion validation using the embedded Judge toolset.
- **Local-First Memory**: **Default: Local embeddings via Ollama** with optional ChromaDB vector store for efficient semantic search. OpenAI-compatible endpoints supported as fallback.
- **Local Cross-Encoder Reranking**: Optional privacy-focused reranking using Qwen3-Reranker-0.6B for improved search relevance without external API calls.
- **MCP Server Orchestration**: Connect to and orchestrate multiple MCP servers.
- **Cross-Server Tool Calling**: Call tools across different MCP servers.
- **Universal MCP Compatibility**: Works with any MCP-compatible client (not tied to specific IDEs).

## Architecture: All-Local Except the Brain

This project follows an **"all-local except the brain"** design philosophy:

- ✅ **Embeddings**: Run locally via Ollama (default: `qwen3-embedding:0.6b`)
- ✅ **Vector Storage**: SQLite (default) or ChromaDB with HNSW indexing (optional RAG package)
- ✅ **Document Chunking**: Token-based chunking with overlap (optional RAG package)
- ✅ **Semantic Search**: Local cosine similarity with L2-normalized vectors
- ✅ **Reranking**: Optional local cross-encoder reranking with Qwen3-Reranker-0.6B
- 🌐 **LLM "Brain"**: Configurable external API (DeepSeek, OpenAI, etc.) for reasoning and generation

**Why?** This architecture keeps your data and semantic search private and fast, while leveraging powerful external LLMs only for high-level reasoning tasks.

## Installation

### 1. Install Python Dependencies

1. Clone or navigate to this directory:
```bash
cd ~/mcp-llm-router
```

2. Create and activate a Conda environment (Python 3.12+):
```bash
conda create -n mcp-router python=3.12 -y
conda activate mcp-router
```

3. Install dependencies:
```bash
pip install -U pip
pip install -e .
```

If editable install fails or you only need dependencies, use the autoinstaller:
```bash
python3 scripts/auto_install.py --upgrade
```

### 2. Install and Setup Ollama (for Local Embeddings)

**Required for default local embeddings functionality.**

1. Install Ollama from [https://ollama.ai](https://ollama.ai) or:
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull the default embedding model:
   ```bash
   ollama pull qwen3-embedding:0.6b
   ```

3. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/version
   ```

   Ollama should start automatically. If not, run:
   ```bash
   ollama serve
   ```

**Alternative Embedding Models:**
- `nomic-embed-text` - General-purpose embeddings
- `mxbai-embed-large` - Larger model for better quality
- Any other Ollama-compatible embedding model

To use a different model, set the environment variable:
```bash
export EMBEDDINGS_MODEL="nomic-embed-text"
```

## Conda Quickstart

```bash
conda create -n mcp-router python=3.12 -y
conda activate mcp-router
python -m pip install -U pip
python -m pip install -e .

# Install Ollama and pull embedding model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3-embedding:0.6b

# Verify everything works
python scripts/verify_server.py
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
- `examples/mcp-config.local-reranker.json` - DeepSeek brain + Ollama embeddings + local cross-encoder reranking.
- `examples/demo_judge_gating.py` - End-to-end demo that indexes memory and walks a task through judge gating via `router_chat`.
- `examples/local_reranker_example.py` - Example of using local cross-encoder reranking to improve search relevance.

Run the demo:

```bash
python examples/demo_judge_gating.py --config examples/mcp-config.deepseek-ollama.json
```

Run the local reranker example:

```bash
python examples/local_reranker_example.py
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

#### Default: Local Ollama Embeddings (Recommended)

**No API keys required!** The default configuration uses local Ollama embeddings:

```bash
# Storage paths
export MCP_ROUTER_DATA_DIR="./.mcp-llm-router"
export MCP_ROUTER_MEMORY_DB="./.mcp-llm-router/memory.db"

# Local embeddings via Ollama (DEFAULT - no API key needed)
export EMBEDDINGS_PROVIDER="ollama"
export EMBEDDINGS_BASE_URL="http://localhost:11434"
export EMBEDDINGS_MODEL="qwen3-embedding:0.6b"
export EMBEDDINGS_PATH="/api/embed"
# No EMBEDDINGS_API_KEY_ENV needed for local Ollama!
```

#### Alternative: OpenAI-Compatible Embeddings

If you prefer cloud-based embeddings:

```bash
# Embeddings via OpenAI
export EMBEDDINGS_PROVIDER="openai"
export EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
export EMBEDDINGS_MODEL="text-embedding-3-small"
export EMBEDDINGS_API_KEY_ENV="OPENAI_API_KEY"
export EMBEDDINGS_PATH="/embeddings"
```

#### Reranking (Optional)

Reranking is optional and defaults to "none". Three modes are available:

##### 1. Local Cross-Encoder Reranking (Recommended for Privacy)

Uses the local Qwen3-Reranker-0.6B model for reranking without external API calls:

```bash
# Local cross-encoder reranking (requires transformers and torch)
export RERANK_PROVIDER="local"
export RERANK_MODE="local"
export RERANK_MODEL="tomaarsen/Qwen3-Reranker-0.6B-seq-cls"  # Default model
```

**Requirements**: 
- Install PyTorch: `pip install torch`
- Install Transformers: `pip install transformers`
- The model will be automatically downloaded on first use (~1.2GB)

##### 2. LLM-Based Reranking

Uses an external LLM API for reranking:

```bash
# Rerank using OpenAI-compatible LLM (optional)
export RERANK_PROVIDER="openai"
export RERANK_BASE_URL="https://api.openai.com/v1"
export RERANK_MODEL="gpt-4o-mini"
export RERANK_API_KEY_ENV="OPENAI_API_KEY"
export RERANK_PATH="/chat/completions"
export RERANK_MODE="llm"
```

##### 3. Disable Reranking

```bash
# Or disable reranking entirely (default)
export RERANK_PROVIDER="none"
```

### Judge Persistence (embedded Judge)

```bash
# Persist judge conversation history + task metadata
export MCP_JUDGE_DATABASE_URL="sqlite:///./.mcp-llm-router/judge_history.db"
```

### Advanced: ChromaDB + Token Chunking (RAG Package)

For enhanced semantic search with vector indexing and intelligent chunking, this repository includes an optional `rag` package that provides:

- **Token-based chunking** with overlap for consistent semantic granularity
- **ChromaDB vector store** with HNSW indexing for fast similarity search
- **L2-normalized embeddings** for consistent cosine similarity
- **Batch embedding** and efficient upserts

#### Using the RAG Package

1. **Install additional dependencies** (already included in `pyproject.toml`):
   ```bash
   pip install -e .  # chromadb, transformers are now included
   ```

2. **Index your codebase**:
   ```bash
   python -m rag.main --path . --exts .py,.md --interactive
   ```

   This will:
   - Scan the current directory for `.py` and `.md` files
   - Chunk them into 400-token segments with 80-token overlap
   - Embed using Ollama (`qwen3-embedding:0.6b`)
   - Store in ChromaDB at `data/chroma/`
   - Enter interactive mode for testing queries

3. **Use in your code**:
   ```python
   from rag.retriever import retrieve
   from rag.indexer import index_path
   
   # Index documents
   stats = index_path("/path/to/docs", exts=[".py", ".md"])
   print(f"Indexed {stats['files_indexed']} files")
   
   # Retrieve relevant chunks
   results = retrieve("How does authentication work?", top_k=5)
   for hit in results:
       print(f"Score: {hit['distance']:.4f}")
       print(f"File: {hit['meta']['path']}")
       print(f"Content: {hit['doc']}\n")
   ```

**RAG Package Components:**
- `rag/embedding_config.py` - Configuration constants
- `rag/chunker.py` - Token-based text chunking
- `rag/ollama_embedder.py` - Ollama embedding with normalization
- `rag/chroma_store.py` - ChromaDB initialization and management
- `rag/indexer.py` - Document indexing pipeline
- `rag/retriever.py` - Vector search and retrieval
- `rag/main.py` - CLI for indexing and queries

**Note:** The RAG package is a self-contained enhancement. The core MCP server works with its built-in SQLite memory store without requiring ChromaDB.

## Usage

### Running MCP Servers

#### Using the Server Runner
```bash
# List configured servers
python scripts/mcp_server_runner.py list

# Run a specific server
python scripts/mcp_server_runner.py run llm-router
```

#### Using the Server Manager
```bash
# Add a new server
python scripts/mcp_manager.py add my-server python -m my_mcp_server

# List servers
python scripts/mcp_manager.py list

# Test server connection
python scripts/mcp_manager.py test llm-router

# Remove a server
python scripts/mcp_manager.py remove my-server
```

### Connecting to MCP Servers

#### Using the MCP Client
```bash
# List tools on a server
python scripts/mcp_client.py list-tools llm-router

# Call a tool on a server
python scripts/mcp_client.py call-tool llm-router start_session '{"goal": "Test session"}'
```

#### Using the Server Manager for Cross-Server Operations
```bash
# Call a tool across all configured servers
python scripts/mcp_manager.py call start_session '{"goal": "Test all servers"}'
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

### Judge Tools (built-in)
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
scripts/opencode run "What is Python"

# Use specific provider
scripts/opencode run "Explain Docker" --provider deepinfra --model meta-llama/Meta-Llama-3.1-70B-Instruct
```

## Development

### Running the Server Directly
```bash
cd ~/mcp-llm-router
conda activate mcp-router
python -m mcp_llm_router.server
```

### Testing
```bash
# Test server startup
timeout 5 python -m mcp_llm_router.server

# Test CLI
scripts/opencode run "Hello world"

# Test MCP client
python scripts/mcp_client.py list-tools llm-router
```

## Architecture

```
┌─────────────────┐    ┌──────────────────────────────────────┐
│   MCP Client    │◄──►│     LLM Router MCP Server            │
│ (Claude, etc.)  │    │  ┌────────────────────────────────┐  │
└─────────────────┘    │  │  Session & Memory Management   │  │
                       │  │  • SQLite/ChromaDB (local)     │  │
                       │  │  • Ollama Embeddings (local)   │  │
                       │  │  • L2-normalized vectors       │  │
                       │  └────────────────────────────────┘  │
                       │                │                     │
                       │                ▼                     │
                       │  ┌────────────────────────────────┐  │
                       │  │  Brain (External LLM API)      │  │
                       │  │  • DeepSeek / OpenAI / etc.    │  │
                       │  │  • Reasoning & Generation      │  │
                       │  └────────────────────────────────┘  │
                       └──────────────────────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────┐
                              │ Other MCP Servers│
                              │ • File system    │
                              │ • Database       │
                              │ • APIs           │
                              └──────────────────┘

All-Local Except the Brain:
  ✅ Embeddings: Ollama (local, no API key)
  ✅ Vector Store: SQLite or ChromaDB (local)
  ✅ Semantic Search: Local cosine similarity
  🌐 LLM Brain: External API (configurable)
```

## License

MIT License - see LICENSE file for details.

```bash
# Basic usage with OpenAI (default)
scripts/opencode run "Explain quantum computing"

# Use a specific provider
scripts/opencode run "Write a Python function" --provider openrouter --model anthropic/claude-3-opus

# Use DeepInfra
scripts/opencode run "Summarize this text" --provider deepinfra --model meta-llama/Llama-3.1-70B-Instruct
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
conda activate mcp-router
python -m mcp_llm_router.server
```

## Environment Variables

Set these in your `~/.bashrc` or Antigravity config:

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
export DEEPINFRA_API_KEY="..."
```
