"""Brain client for OpenAI-compatible chat completions used by the router."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_PROVIDER_BASE_URLS = {
    "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "deepinfra": os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    "anthropic": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
    "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
}


@dataclass
class BrainConfig:
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    provider: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 60.0


class BrainClient:
    def __init__(self, provider_base_urls: Optional[Dict[str, str]] = None) -> None:
        self.provider_base_urls = provider_base_urls or DEFAULT_PROVIDER_BASE_URLS

    async def chat(self, messages: List[Dict[str, str]], config: BrainConfig) -> Dict[str, Any]:
        base_url, provider, api_key_env = _resolve_brain_endpoint(config, self.provider_base_urls)
        api_key = _get_api_key(api_key_env) if api_key_env else None

        headers = {
            "Content-Type": "application/json",
            **(config.extra_headers or {}),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: Dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if config.reasoning_effort:
            body["reasoning_effort"] = config.reasoning_effort
        if config.extra_body:
            body.update(config.extra_body)

        url = _join_url(base_url, "/chat/completions")

        async with httpx.AsyncClient(timeout=config.timeout_s) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _resolve_brain_endpoint(
    config: BrainConfig, provider_base_urls: Dict[str, str]
) -> tuple[str, Optional[str], Optional[str]]:
    provider = config.provider
    base_url = config.base_url or None
    api_key_env = config.api_key_env

    if _looks_like_deepseek(config.model) and provider is None and base_url is None:
        provider = "deepseek"
        if api_key_env == "OPENAI_API_KEY":
            api_key_env = "DEEPSEEK_API_KEY"

    if base_url is None:
        if provider and provider.lower() in provider_base_urls:
            base_url = provider_base_urls[provider.lower()]
        else:
            base_url = provider_base_urls.get("openai", "https://api.openai.com/v1")

    return base_url, provider, api_key_env


def _looks_like_deepseek(model: str) -> bool:
    return model.startswith("deepseek-") or "deepseek" in model


def _get_api_key(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    val = os.getenv(name)
    if val and val not in ("", "YOUR_DEEPSEEK_API_KEY", "YOUR_OPENAI_API_KEY"):
        return val

    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        try:
            with open(bashrc_path, "r", encoding="utf-8") as f:
                content = f.read()
                pattern = rf'^export\s+{re.escape(name)}=["\']?([^"\'\s#]+)["\']?'
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    return match.group(1)
        except Exception:
            pass

    return None
