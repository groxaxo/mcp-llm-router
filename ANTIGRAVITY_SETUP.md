# Add MCP LLM Router to Antigravity

This guide shows how to integrate the MCP LLM Router server with Google's Antigravity agent framework.

## 1. Install the Server Locally

From the folder containing your `mcp-llm-router/` project:

```bash
cd mcp-llm-router
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install "fastmcp<3" openai httpx
```

## 2. Configure Antigravity

### Locate Config File

Find your Antigravity MCP config file:

- **Linux/macOS:** `~/.gemini/antigravity/mcp_config.json`
- **Windows:** `C:\Users\<USER>\.gemini\antigravity\mcp_config.json`

### Access in Antigravity UI

**Agent panel** → **…** → **MCP Servers** → **Manage MCP Servers** → **View raw config**

### Add Server Configuration

Edit `mcp_config.json` and add the `llm-router` server. **Use absolute paths:**

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "/ABSOLUTE/PATH/TO/mcp-llm-router/.venv/bin/python",
      "args": ["-m", "mcp_llm_router.server"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "DEEPINFRA_API_KEY": "..."
      }
    }
  }
}
```

**Replace `/ABSOLUTE/PATH/TO/` with your actual project path!**

On Windows, use forward slashes or escaped backslashes:
```json
"command": "C:/Users/YourName/mcp-llm-router/.venv/Scripts/python.exe"
```

### Refresh Antigravity

After saving the config:
1. Go back to the MCP Servers screen in Antigravity
2. Click **Refresh** to load the tools

## 3. Using the Router in Antigravity

Once installed, your agent will have access to these tools:

### Session Management
- `start_session(goal, constraints, context, metadata)` - Start a new agent session
- `log_event(session_id, kind, message, details)` - Log events (info, error, warning, success)
- `get_session_context(session_id)` - Retrieve full session context

### LLM Routing
- `agent_llm_request(session_id, prompt, model, ...)` - Route requests to different LLMs

### MCP Integration
- `connect_mcp_server(server_name, command, args, env)` - Connect to other MCP servers
- `list_mcp_servers()` - List configured MCP servers

## 4. Example Agent Workflow

```python
# 1. Start a session
session = start_session(
    goal="Build a REST API for user management",
    constraints="Use FastAPI, SQLite, and follow REST best practices"
)
session_id = session["session_id"]

# 2. Log progress
log_event(
    session_id=session_id,
    kind="info",
    message="Created project structure"
)

# 3. When stuck, consult an LLM
response = agent_llm_request(
    session_id=session_id,
    prompt="How should I structure SQLAlchemy models for users with roles?",
    model="gpt-4",
    provider="openai",  # Uses OPENAI_API_KEY from config
    api_key_env="OPENAI_API_KEY"
)

# 4. Or use OpenRouter for specialized models
response = agent_llm_request(
    session_id=session_id,
    prompt="Generate comprehensive test cases",
    model="anthropic/claude-3-opus-20240229",
    provider="openrouter",  # Automatically uses https://openrouter.ai/api/v1
    api_key_env="OPENROUTER_API_KEY"
)

# 5. Or use DeepInfra for open models
response = agent_llm_request(
    session_id=session_id,
    prompt="Explain this code pattern",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="deepinfra",  # Automatically uses https://api.deepinfra.com/v1/openai
    api_key_env="DEEPINFRA_API_KEY"
)
```

## 5. Provider Configuration

### Simplified Usage (Recommended)

Just specify the `provider` parameter - no need to pass `base_url`:

| Provider | Usage Example |
|----------|--------------|
| **OpenAI** | `provider="openai", api_key_env="OPENAI_API_KEY"` |
| **OpenRouter** | `provider="openrouter", api_key_env="OPENROUTER_API_KEY"` |
| **DeepInfra** | `provider="deepinfra", api_key_env="DEEPINFRA_API_KEY"` |
| **Anthropic** | `provider="anthropic", api_key_env="ANTHROPIC_API_KEY"` |

### Advanced: Custom Base URLs

You can override default base URLs via environment variables in `mcp_config.json`:

```json
{
  "mcpServers": {
    "llm-router": {
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_BASE_URL": "https://custom.openai.endpoint/v1",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "DEEPINFRA_BASE_URL": "https://api.deepinfra.com/v1/openai"
      }
    }
  }
}
```

Or pass `base_url` directly in the tool call:

```python
agent_llm_request(
    session_id=session_id,
    prompt="...",
    model="custom-model",
    base_url="https://my-custom-endpoint.com/v1",
    api_key_env="MY_CUSTOM_KEY"
)
```

## 6. Best Practices for Autonomous Agents

### Session-Driven Development
1. Always start with `start_session()` to establish context
2. Log all significant events: file edits, command runs, decisions
3. Use `kind="error"` for failures to help debugging

### Strategic LLM Usage
- Use cheaper models (gpt-3.5-turbo, claude-haiku) for routine tasks
- Use powerful models (gpt-4, claude-opus) for complex reasoning
- Use specialized models (codegen, deepseek-coder) via OpenRouter/DeepInfra

### Error Recovery
```python
# Log errors with details
log_event(
    session_id=session_id,
    kind="error",
    message="Build failed",
    details={"command": "npm build", "exit_code": 1}
)

# Consult LLM for solutions
response = agent_llm_request(
    session_id=session_id,
    prompt=f"Build failed with: {error_output}. How to fix?",
    model="gpt-4",
    provider="openai"
)
```

## 7. Troubleshooting

### Server Not Appearing in Antigravity

1. Check that the path in `mcp_config.json` is **absolute**
2. Verify Python environment has all dependencies installed
3. Click **Refresh** in the MCP Servers screen
4. Check Antigravity logs for connection errors

### API Key Errors

Ensure environment variables are set in the config:
```json
"env": {
  "OPENAI_API_KEY": "your-actual-key-here"
}
```

### Test the Server Manually

Run the server standalone to verify it works:

```bash
cd mcp-llm-router
source .venv/bin/activate
python -m mcp_llm_router.server
```

It should start without errors.

## 8. Next Steps

- Explore other MCP servers to integrate via `connect_mcp_server()`
- Build custom agent workflows leveraging multiple LLM providers
- Extend the server with your own custom tools and capabilities

---

**Need help?** Check the [FastMCP documentation](https://github.com/jlowin/fastmcp) or review the server source code.
