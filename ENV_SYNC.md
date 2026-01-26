# Environment Variable Auto-Sync

The MCP LLM Router can automatically pull API keys from your system environment variables.

## Quick Sync

Run this anytime you update your environment variables:

```bash
./sync_antigravity_env.sh
```

This script:
- ✅ Reads `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPINFRA_API_KEY`, `ANTHROPIC_API_KEY` from your environment
- ✅ Updates `~/.gemini/antigravity/mcp_config.json` automatically
- ✅ No need to manually edit JSON files

## Auto-Sync on Shell Startup (Optional)

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
# Auto-sync MCP LLM Router environment variables
if [ -f "$HOME/mcp-llm-router/sync_antigravity_env.sh" ]; then
    "$HOME/mcp-llm-router/sync_antigravity_env.sh" > /dev/null 2>&1
fi
```

Now your Antigravity config stays in sync with your environment automatically!

## Setting Environment Variables

### Option 1: Add to ~/.bashrc (Persistent)

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
echo 'export DEEPINFRA_API_KEY="..."' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Use a .env file

Create `.env` in your project:

```bash
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
DEEPINFRA_API_KEY=...
```

Load it before syncing:

```bash
source .env
./sync_antigravity_env.sh
```

## Manual Configuration (Not Recommended)

If you prefer, you can manually edit `~/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "llm-router": {
      "command": "/home/op/mcp-llm-router/.venv/bin/python",
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

But using `sync_antigravity_env.sh` is much easier! 🚀
