# Quick Start: Antigravity Integration

## One-Command Setup

### Linux/macOS
```bash
cd /path/to/mcp-llm-router
./setup_antigravity.sh
```

### Windows
```cmd
cd C:\path\to\mcp-llm-router
setup_antigravity.bat
```

The script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Print your configuration snippet with the correct path

## Manual Configuration

After running the setup script, add the configuration to Antigravity:

**Location:** `~/.gemini/antigravity/mcp_config.json` (Linux/macOS) or `C:\Users\<USER>\.gemini\antigravity\mcp_config.json` (Windows)

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "/absolute/path/to/mcp-llm-router/.venv/bin/python",
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

**Don't forget to:**
1. Replace the command path with the output from the setup script
2. Add your actual API keys
3. Click **Refresh** in Antigravity's MCP Servers panel

## Usage Examples

### Simple OpenAI Request
```python
# Agent automatically has access to these tools
session = start_session(goal="Debug authentication flow")
sid = session["session_id"]

response = agent_llm_request(
    session_id=sid,
    prompt="Explain OAuth2 authorization code flow",
    model="gpt-4",
    provider="openai",  # 🎯 New: simplified provider selection
    api_key_env="OPENAI_API_KEY"
)
```

### OpenRouter (Anthropic, DeepSeek, etc.)
```python
response = agent_llm_request(
    session_id=sid,
    prompt="Refactor this code to use async/await",
    model="anthropic/claude-3-opus-20240229",
    provider="openrouter",  # 🎯 Automatically uses correct base URL
    api_key_env="OPENROUTER_API_KEY"
)
```

### DeepInfra (Open Source Models)
```python
response = agent_llm_request(
    session_id=sid,
    prompt="Generate unit tests for this function",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="deepinfra",  # 🎯 Automatically uses correct base URL
    api_key_env="DEEPINFRA_API_KEY"
)
```

## What Changed?

**Before (required manual base_url):**
```python
agent_llm_request(
    session_id=sid,
    prompt="...",
    model="anthropic/claude-3-opus",
    base_url="https://openrouter.ai/api/v1",  # ❌ Had to specify manually
    api_key_env="OPENROUTER_API_KEY"
)
```

**After (automatic routing):**
```python
agent_llm_request(
    session_id=sid,
    prompt="...",
    model="anthropic/claude-3-opus",
    provider="openrouter",  # ✅ Automatically resolves base URL
    api_key_env="OPENROUTER_API_KEY"
)
```

## Provider Base URLs (Configured Automatically)

| Provider | Base URL | Can Override With |
|----------|----------|-------------------|
| `openai` | `https://api.openai.com/v1` | `OPENAI_BASE_URL` env var |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_BASE_URL` env var |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_BASE_URL` env var |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_BASE_URL` env var |

## Full Documentation

- **Setup Guide:** [ANTIGRAVITY_SETUP.md](ANTIGRAVITY_SETUP.md)
- **Examples:** [examples/antigravity_example.md](examples/antigravity_example.md)

## Troubleshooting

**Server not showing in Antigravity?**
- Verify the `command` path is absolute
- Check that `.venv/bin/python` exists
- Click **Refresh** in MCP Servers panel

**API key errors?**
- Ensure keys are set in the `env` section of config
- Don't use placeholder values like "your-key-here"

**Test manually:**
```bash
cd /path/to/mcp-llm-router
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m mcp_llm_router.server
# Should start without errors
```
