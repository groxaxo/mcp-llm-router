#!/bin/bash
# Quick setup script for MCP LLM Router

set -e

echo "🚀 MCP LLM Router Setup for Antigravity"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install -U pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install "fastmcp<3" openai httpx

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get your absolute path:"
echo "   pwd"
echo ""
echo "2. Open Antigravity MCP config:"
echo "   - On Linux/macOS: ~/.gemini/antigravity/mcp_config.json"
echo "   - On Windows: C:\\Users\\<USER>\\.gemini\\antigravity\\mcp_config.json"
echo ""
echo "3. Add this configuration:"
echo "{"
echo "  \"mcpServers\": {"
echo "    \"llm-router\": {"
echo "      \"command\": \"$SCRIPT_DIR/.venv/bin/python\","
echo "      \"args\": [\"-m\", \"mcp_llm_router.server\"],"
echo "      \"env\": {"
echo "        \"OPENAI_API_KEY\": \"your-key-here\","
echo "        \"OPENROUTER_API_KEY\": \"your-key-here\","
echo "        \"DEEPINFRA_API_KEY\": \"your-key-here\""
echo "      }"
echo "    }"
echo "  }"
echo "}"
echo ""
echo "4. Refresh Antigravity MCP Servers"
echo ""
echo "📖 For detailed instructions, see ANTIGRAVITY_SETUP.md"
