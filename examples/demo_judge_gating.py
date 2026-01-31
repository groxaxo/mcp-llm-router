#!/usr/bin/env python3
"""End-to-end demo: memory indexing + router_chat + judge gating."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_config_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path)

    repo_root = Path(__file__).resolve().parents[1]
    sample = repo_root / "examples" / "mcp-config.deepseek-ollama.json"
    if sample.exists():
        return sample

    fallback = repo_root / "mcp-config.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError("No MCP config found. Use --config to provide one.")


def _extract_payload(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result

    if hasattr(result, "content"):
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                texts.append(str(item))

        raw = "\n".join(texts).strip()
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Best-effort extraction if the content includes JSON in text
                if "{" in raw and "}" in raw:
                    snippet = raw[raw.find("{") : raw.rfind("}") + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        pass
        return {"raw": raw}

    return {"raw": str(result)}


def _get_task_id(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    if "task_id" in payload:
        return payload.get("task_id")
    meta = payload.get("current_task_metadata") or {}
    if isinstance(meta, dict) and meta.get("task_id"):
        return meta.get("task_id")
    return None


def _get_session_id(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    return payload.get("session_id")


def _print_step(title: str) -> None:
    print("\n==>", title)


def _print_payload(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except TypeError:
        print(payload)


async def _call_tool(
    session: ClientSession, tool_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    result = await session.call_tool(tool_name, arguments=arguments)
    return _extract_payload(result)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo: memory indexing + router_chat + judge gating"
    )
    parser.add_argument(
        "--config",
        help="Path to MCP config JSON (defaults to examples/mcp-config.deepseek-ollama.json)",
    )
    parser.add_argument("--server", default="llm-router", help="Server name")
    parser.add_argument("--namespace", default="demo-memory", help="Memory namespace")
    parser.add_argument(
        "--message",
        default=(
            "Create a concise implementation plan for adding a /status endpoint to a "
            "FastAPI app. Use memory context and include tests."
        ),
        help="Prompt for router_chat",
    )

    args = parser.parse_args()

    config_path = _default_config_path(args.config)
    config = _load_config(config_path)

    server_cfg = config.get("mcpServers", {}).get(args.server)
    if not server_cfg:
        print(f"Server '{args.server}' not found in {config_path}")
        return 1

    env = dict(os.environ)
    env.update(server_cfg.get("env", {}))
    server_params = StdioServerParameters(
        command=server_cfg["command"],
        args=server_cfg.get("args", []),
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            _print_step("set_coding_task")
            task_payload = await _call_tool(
                session,
                "set_coding_task",
                {
                    "user_request": "Add a /status endpoint to the API",
                    "task_title": "Status endpoint",
                    "task_description": "Expose a /status endpoint that returns app health and version.",
                    "user_requirements": "Include a JSON response and basic tests.",
                    "tags": ["demo", "status"],
                },
            )
            _print_payload(task_payload)
            task_id = _get_task_id(task_payload)
            if not task_id:
                print("Failed to extract task_id from set_coding_task response.")
                return 1

            _print_step("start_session")
            session_payload = await _call_tool(
                session,
                "start_session",
                {
                    "goal": "Implement /status endpoint",
                    "constraints": "Keep changes minimal and add tests",
                    "context": "FastAPI app with routes in api/routes.py",
                    "task_id": task_id,
                },
            )
            _print_payload(session_payload)
            session_id = _get_session_id(session_payload)
            if not session_id:
                print("Failed to extract session_id from start_session response.")
                return 1

            _print_step("memory_index")
            memory_payload = await _call_tool(
                session,
                "memory_index",
                {
                    "namespace": args.namespace,
                    "session_id": session_id,
                    "texts": [
                        "The FastAPI app defines routes in api/routes.py and uses APIRouter.",
                        "Tests live in tests/test_api.py and use httpx AsyncClient.",
                        "The app config includes a __version__ string in app/config.py.",
                    ],
                    "metadatas": [
                        {"source": "repo"},
                        {"source": "tests"},
                        {"source": "config"},
                    ],
                    "doc_ids": ["routes", "tests", "config"],
                },
            )
            _print_payload(memory_payload)

            _print_step("router_chat")
            router_payload = await _call_tool(
                session,
                "router_chat",
                {
                    "session_id": session_id,
                    "message": args.message,
                    "memory_namespace": args.namespace,
                },
            )
            _print_payload(router_payload)

            plan_text = router_payload.get("content") if isinstance(router_payload, dict) else None
            if not plan_text:
                plan_text = (
                    "- Add /status route returning {status, version}.\n"
                    "- Wire route into the API router.\n"
                    "- Add tests in tests/test_api.py for status 200 + payload."
                )

            _print_step("request_plan_approval (skipped - interactive)")
            print("This demo skips request_plan_approval because it requires user elicitation.")

            _print_step("judge_coding_plan")
            plan_payload = await _call_tool(
                session,
                "judge_coding_plan",
                {
                    "plan": plan_text,
                    "design": "Add a small handler function and wire to APIRouter.",
                    "research": "No external research required.",
                    "research_urls": [],
                    "task_id": task_id,
                },
            )
            _print_payload(plan_payload)

            _print_step("judge_code_change")
            code_payload = await _call_tool(
                session,
                "judge_code_change",
                {
                    "code_change": (
                        "diff --git a/api/routes.py b/api/routes.py\n"
                        "@@\n"
                        "+@router.get('/status')\n"
                        "+async def status():\n"
                        "+    return {'status': 'ok', 'version': __version__}\n"
                    ),
                    "file_path": "api/routes.py",
                    "change_description": "Add /status endpoint that returns status and version.",
                    "task_id": task_id,
                },
            )
            _print_payload(code_payload)

            _print_step("judge_testing_implementation")
            test_payload = await _call_tool(
                session,
                "judge_testing_implementation",
                {
                    "task_id": task_id,
                    "test_summary": "Added tests for /status endpoint response.",
                    "test_files": ["tests/test_api.py"],
                    "test_execution_results": "pytest -q (simulated)",
                    "testing_framework": "pytest",
                    "test_types_implemented": ["integration"],
                },
            )
            _print_payload(test_payload)

            _print_step("judge_coding_task_completion")
            completion_payload = await _call_tool(
                session,
                "judge_coding_task_completion",
                {
                    "task_id": task_id,
                    "completion_summary": "Added /status endpoint + tests.",
                    "requirements_met": [
                        "Expose /status endpoint",
                        "Return JSON with status and version",
                        "Add tests",
                    ],
                    "implementation_details": "Route in api/routes.py, tests in tests/test_api.py.",
                    "testing_status": "tests added",
                },
            )
            _print_payload(completion_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
