#!/bin/bash
# Auto-sync script: Updates Antigravity config with current environment variables

CONFIG_FILE="$HOME/.gemini/antigravity/mcp_config.json"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Syncing environment variables to Antigravity MCP config..."

# Get current environment variables
OPENAI_KEY="${OPENAI_API_KEY:-}"
OPENROUTER_KEY="${OPENROUTER_API_KEY:-}"
DEEPINFRA_KEY="${DEEPINFRA_API_KEY:-}"
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"

# Build env object
ENV_JSON="{"
FIRST=true

if [ -n "$OPENAI_KEY" ]; then
    ENV_JSON="$ENV_JSON\"OPENAI_API_KEY\": \"$OPENAI_KEY\""
    FIRST=false
    echo "  ✓ Found OPENAI_API_KEY"
fi

if [ -n "$OPENROUTER_KEY" ]; then
    [ "$FIRST" = false ] && ENV_JSON="$ENV_JSON,"
    ENV_JSON="$ENV_JSON\"OPENROUTER_API_KEY\": \"$OPENROUTER_KEY\""
    FIRST=false
    echo "  ✓ Found OPENROUTER_API_KEY"
fi

if [ -n "$DEEPINFRA_KEY" ]; then
    [ "$FIRST" = false ] && ENV_JSON="$ENV_JSON,"
    ENV_JSON="$ENV_JSON\"DEEPINFRA_API_KEY\": \"$DEEPINFRA_KEY\""
    FIRST=false
    echo "  ✓ Found DEEPINFRA_API_KEY"
fi

if [ -n "$ANTHROPIC_KEY" ]; then
    [ "$FIRST" = false ] && ENV_JSON="$ENV_JSON,"
    ENV_JSON="$ENV_JSON\"ANTHROPIC_API_KEY\": \"$ANTHROPIC_KEY\""
    FIRST=false
    echo "  ✓ Found ANTHROPIC_API_KEY"
fi

ENV_JSON="$ENV_JSON}"

# Create config directory if it doesn't exist
mkdir -p "$(dirname "$CONFIG_FILE")"

# Write config
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "llm-router": {
      "command": "$PROJECT_DIR/.venv/bin/python",
      "args": ["-m", "mcp_llm_router.server"],
      "env": $ENV_JSON
    }
  }
}
EOF

echo ""
echo "✅ Config updated: $CONFIG_FILE"
echo ""
echo "📋 Next step: Refresh Antigravity MCP Servers"
echo ""
echo "💡 Tip: Add this to your ~/.bashrc to auto-sync on shell startup:"
echo "   $PROJECT_DIR/sync_antigravity_env.sh"
