#!/usr/bin/env python3
"""Lightweight MCP inspector smoke check for the embedded router + judge server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp_llm_router.server import mcp


async def _collect_capabilities() -> dict[str, object]:
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    resource_templates = await mcp.list_resource_templates()
    prompts = await mcp.list_prompts()

    return {
        "server": "mcp-llm-router",
        "tool_count": len(tools),
        "resource_count": len(resources),
        "resource_template_count": len(resource_templates),
        "prompt_count": len(prompts),
        "tools": sorted(tool.name for tool in tools),
        "resources": sorted(str(resource.uri) for resource in resources),
        "resource_templates": sorted(
            template.uriTemplate for template in resource_templates
        ),
        "prompts": sorted(prompt.name for prompt in prompts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-stdio",
        action="store_true",
        help="Run the stdio server instead of printing the inspector summary.",
    )
    args = parser.parse_args()

    if args.run_stdio:
        mcp.run()
        return 0

    summary = asyncio.run(_collect_capabilities())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
