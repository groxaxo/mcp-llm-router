"""
Elicitation provider package.

This package provides a factory pattern for creating elicitation providers
that handle user input elicitation through various methods (MCP elicitation,
fallback prompts, etc.).
"""

from mcp_llm_router.judge.elicitation.factory import (
    ElicitationProviderFactory,
    elicitation_provider,
)
from mcp_llm_router.judge.elicitation.fallback_provider import FallbackElicitationProvider
from mcp_llm_router.judge.elicitation.interface import ElicitationProvider, ElicitationResult
from mcp_llm_router.judge.elicitation.mcp_provider import MCPElicitationProvider

__all__ = [
    "ElicitationProvider",
    "ElicitationProviderFactory",
    "ElicitationResult",
    "FallbackElicitationProvider",
    "MCPElicitationProvider",
    "elicitation_provider",
]
