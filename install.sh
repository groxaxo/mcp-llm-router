#!/bin/bash
# MCP LLM Router - Automated Installation Script
# Supports Linux, macOS, and WSL

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   ███╗   ███╗ ██████╗██████╗     ██╗     ██╗     ███╗   ███╗║
║   ████╗ ████║██╔════╝██╔══██╗    ██║     ██║     ████╗ ████║║
║   ██╔████╔██║██║     ██████╔╝    ██║     ██║     ██╔████╔██║║
║   ██║╚██╔╝██║██║     ██╔═══╝     ██║     ██║     ██║╚██╔╝██║║
║   ██║ ╚═╝ ██║╚██████╗██║         ███████╗███████╗██║ ╚═╝ ██║║
║   ╚═╝     ╚═╝ ╚═════╝╚═╝         ╚══════╝╚══════╝╚═╝     ╚═╝║
║                                                            ║
║              R O U T E R   I N S T A L L E R               ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}MCP LLM Router - Automated Installation${NC}"
echo "=================================================="
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo -e "${BLUE}🖥️  Detected OS: ${OS}${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}📋 Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.12 or later.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment
echo ""
echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}🔌 Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo ""
echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
python -m pip install -U pip --quiet

# Install dependencies
echo ""
echo -e "${YELLOW}📥 Installing dependencies...${NC}"
if [ -f "pyproject.toml" ]; then
    pip install -e . --quiet
    echo -e "${GREEN}✓ Installed from pyproject.toml${NC}"
else
    echo -e "${RED}❌ pyproject.toml not found${NC}"
    exit 1
fi

# Check Ollama
echo ""
echo -e "${YELLOW}🔍 Checking for Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/version &> /dev/null; then
        echo -e "${GREEN}✓ Ollama is running${NC}"
        
        # Check for embedding model
        if ollama list | grep -q "qwen3-embedding"; then
            echo -e "${GREEN}✓ Embedding model (qwen3-embedding) is installed${NC}"
        else
            echo -e "${YELLOW}⚠️  Embedding model not found${NC}"
            echo -e "${BLUE}   To install: ollama pull qwen3-embedding:0.6b${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Ollama is not running${NC}"
        echo -e "${BLUE}   To start: ollama serve${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Ollama not found${NC}"
    echo ""
    echo -e "${BLUE}Ollama is required for local embeddings (recommended).${NC}"
    echo -e "${BLUE}To install Ollama:${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${BLUE}  curl -fsSL https://ollama.ai/install.sh | sh${NC}"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "${BLUE}  brew install ollama${NC}"
        echo -e "${BLUE}  or download from: https://ollama.ai${NC}"
    else
        echo -e "${BLUE}  Download from: https://ollama.ai${NC}"
    fi
    echo ""
    echo -e "${BLUE}After installing Ollama:${NC}"
    echo -e "${BLUE}  1. Start Ollama: ollama serve${NC}"
    echo -e "${BLUE}  2. Pull embedding model: ollama pull qwen3-embedding:0.6b${NC}"
fi

# Verify installation
echo ""
echo -e "${YELLOW}🧪 Verifying installation...${NC}"
if python -c "import mcp_llm_router.server" 2>/dev/null; then
    echo -e "${GREEN}✓ MCP LLM Router module loads successfully${NC}"
else
    echo -e "${RED}❌ Failed to import mcp_llm_router.server${NC}"
    exit 1
fi

# Success
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}║  ✅  Installation Complete!                          ║${NC}"
echo -e "${GREEN}║                                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Print next steps
echo -e "${BLUE}📋 Next Steps:${NC}"
echo ""
echo -e "${YELLOW}1. Set up your API keys (in ~/.bashrc or ~/.zshrc):${NC}"
echo "   export OPENAI_API_KEY=\"sk-...\""
echo "   export DEEPSEEK_API_KEY=\"...\""
echo "   export OPENROUTER_API_KEY=\"sk-or-...\""
echo ""

echo -e "${YELLOW}2. Configure your MCP client:${NC}"
echo "   Add this to your MCP config file:"
echo ""

# Determine Python path based on OS
if [[ "$OS" == "windows" ]]; then
    PYTHON_PATH="${SCRIPT_DIR}/.venv/Scripts/python.exe"
else
    PYTHON_PATH="${SCRIPT_DIR}/.venv/bin/python"
fi

echo "   {"
echo "     \"mcpServers\": {"
echo "       \"llm-router\": {"
echo "         \"command\": \"${PYTHON_PATH}\","
echo "         \"args\": [\"-m\", \"mcp_llm_router.server\"],"
echo "         \"env\": {"
echo "           \"OPENAI_API_KEY\": \"your-key-here\","
echo "           \"DEEPSEEK_API_KEY\": \"your-key-here\""
echo "         }"
echo "       }"
echo "     }"
echo "   }"
echo ""

echo -e "${YELLOW}3. Test the installation:${NC}"
echo "   python scripts/verify_server.py"
echo ""

echo -e "${YELLOW}4. Run examples:${NC}"
echo "   python examples/demo_judge_gating.py --config examples/mcp-config.deepseek-ollama.json"
echo ""

echo -e "${BLUE}📖 Documentation: README.md${NC}"
echo ""
