import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from mcp_llm_router.judge.mcp_features import (
    get_task_capability_status,
    register_judge_prompts,
    register_judge_resources,
)
from mcp_llm_router.judge.roots import find_paths_outside_roots


@pytest.mark.anyio
async def test_register_judge_resources_and_prompts():
    server = FastMCP("judge-test")
    register_judge_resources(server)
    register_judge_prompts(server)

    resources = await server.list_resources()
    prompts = await server.list_prompts()
    resource_templates = await server.list_resource_templates()

    resource_uris = {str(resource.uri) for resource in resources}
    prompt_names = {prompt.name for prompt in prompts}
    template_uris = {template.uriTemplate for template in resource_templates}

    assert "judge://current-task" in resource_uris
    assert "judge://policy/rubric" in resource_uris
    assert "judge://workflow/states" in resource_uris
    assert "judge://task/{task_id}" in template_uris
    assert "judge://task/{task_id}/history" in template_uris
    assert "start_judged_coding_task" in prompt_names
    assert "submit_implementation_for_review" in prompt_names
    assert "prepare_testing_evidence" in prompt_names


@pytest.mark.anyio
async def test_current_task_resource_is_graceful_without_history():
    server = FastMCP("judge-test")
    register_judge_resources(server)

    with patch(
        "mcp_llm_router.judge.mcp_features.conversation_service.db.get_recent_sessions",
        AsyncMock(return_value=[]),
    ):
        contents = list(await server.read_resource("judge://current-task"))

    assert json.loads(contents[0].content)["found"] is False


@pytest.mark.anyio
async def test_find_paths_outside_roots_rejects_parent_traversal_when_roots_exist():
    ctx = MagicMock()
    ctx.session.list_roots = AsyncMock(
        return_value=SimpleNamespace(roots=[SimpleNamespace(uri="file:///repo")])
    )

    violations = await find_paths_outside_roots(
        ["src/app.py", "../secrets.txt", "/tmp/outside.py"],
        ctx,
    )

    assert "../secrets.txt" in violations
    assert "/tmp/outside.py" in violations
    assert "src/app.py" not in violations


def test_task_capability_status_is_feature_gated():
    with patch.dict(os.environ, {"MCP_JUDGE_ENABLE_TASKS": "true"}, clear=False):
        status = json.loads(get_task_capability_status())

    assert status["requested"] is True
    assert status["active"] is False
