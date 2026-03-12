"""Compatibility bootstrap for the embedded judge server."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from mcp_llm_router.judge.mcp_features import (
    register_judge_prompts,
    register_judge_resources,
)
from mcp_llm_router.judge.runtime import (
    get_registered_mcp as _get_registered_mcp,
    set_registered_mcp,
)
from mcp_llm_router.judge.tool_description.factory import tool_description_provider
from mcp_llm_router.judge.tools.code_tools import (
    judge_code_change,
    judge_coding_task_completion,
)
from mcp_llm_router.judge.tools.obstacle_tools import (
    raise_missing_requirements,
    raise_obstacle,
)
from mcp_llm_router.judge.tools.plan_tools import (
    judge_coding_plan,
    request_plan_approval,
)
from mcp_llm_router.judge.tools.task_tools import (
    get_current_coding_task,
    set_coding_task,
)
from mcp_llm_router.judge.tools.testing_tools import judge_testing_implementation

mcp = _get_registered_mcp()


def get_registered_mcp() -> FastMCP | None:
    """Return the FastMCP instance used to register judge tools, if available."""
    return _get_registered_mcp()


def main() -> None:
    """Compatibility shim: run the unified router server instead of a standalone judge."""
    from mcp_llm_router.server import mcp as router_mcp

    router_mcp.run()


def _tool_definitions() -> list[tuple[Callable[..., object], str]]:
    return [
        (set_coding_task, "set_coding_task"),
        (get_current_coding_task, "get_current_coding_task"),
        (request_plan_approval, "request_plan_approval"),
        (raise_obstacle, "raise_obstacle"),
        (raise_missing_requirements, "raise_missing_requirements"),
        (judge_coding_task_completion, "judge_coding_task_completion"),
        (judge_coding_plan, "judge_coding_plan"),
        (judge_code_change, "judge_code_change"),
        (judge_testing_implementation, "judge_testing_implementation"),
    ]


def register_judge_tools(mcp_server: FastMCP) -> None:
    """Register all judge tools with the provided FastMCP instance."""
    global mcp
    set_registered_mcp(mcp_server)
    mcp = mcp_server

    for tool_fn, tool_name in _tool_definitions():
        mcp_server.add_tool(
            tool_fn,
            description=tool_description_provider.get_description(tool_name),
        )

    register_judge_resources(mcp_server)
    register_judge_prompts(mcp_server)
