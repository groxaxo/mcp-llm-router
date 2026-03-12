import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mcp_llm_router.judge.server as judge_server
from mcp_llm_router.judge.tools import task_tools
from mcp_llm_router.judge.models.enhanced_responses import TaskAnalysisResult
from mcp_llm_router.judge.models.task_metadata import TaskMetadata, TaskSize, TaskState
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance


def test_tool_optional_collection_defaults_are_none():
    expected_defaults = {
        judge_server.set_coding_task: ["tags"],
        judge_server.request_plan_approval: [
            "research_urls",
            "problem_non_goals",
            "library_plan",
            "internal_reuse_components",
        ],
        judge_server.raise_obstacle: ["constraints"],
        judge_server.raise_missing_requirements: [
            "decision_areas",
            "options",
            "constraints",
        ],
        judge_server.judge_coding_task_completion: ["remaining_work"],
        judge_server.judge_coding_plan: [
            "problem_non_goals",
            "library_plan",
            "internal_reuse_components",
            "design_patterns",
            "identified_risks",
            "risk_mitigation_strategies",
        ],
        judge_server.judge_testing_implementation: ["test_types_implemented"],
    }

    for tool_fn, parameter_names in expected_defaults.items():
        signature = inspect.signature(tool_fn)
        for name in parameter_names:
            assert signature.parameters[name].default is None


@pytest.mark.anyio
async def test_set_coding_task_default_tags_are_isolated():
    observed_ids: list[int] = []
    observed_initial_values: list[list[str]] = []

    async def fake_create_new_coding_task(*, tags, **_kwargs):
        observed_ids.append(id(tags))
        observed_initial_values.append(list(tags))
        tags.append("mutated-in-fake")
        return TaskMetadata(
            title="Example task",
            description="Example description",
            user_requirements="Keep tests stable",
            state=TaskState.PLANNING,
            task_size=TaskSize.M,
            tags=list(tags),
        )

    fake_guidance = WorkflowGuidance(
        next_tool="judge_coding_plan",
        reasoning="Continue the normal workflow.",
        preparation_needed=["Review the generated task metadata."],
        guidance="Proceed to planning validation.",
    )

    ctx = MagicMock()

    with (
        patch.object(task_tools, "set_context_reference"),
        patch.object(task_tools, "context_logger", MagicMock(info=AsyncMock())),
        patch(
            "mcp_llm_router.judge.tools.task_tools.create_new_coding_task",
            side_effect=fake_create_new_coding_task,
        ),
        patch(
            "mcp_llm_router.judge.tools.task_tools.calculate_next_stage",
            AsyncMock(return_value=fake_guidance),
        ),
        patch(
            "mcp_llm_router.judge.tools.task_tools.save_task_metadata_to_history",
            AsyncMock(),
        ),
        patch.object(
            task_tools,
            "conversation_service",
            MagicMock(save_tool_interaction_and_cleanup=AsyncMock()),
        ),
    ):
        result_one = await judge_server.set_coding_task(
            user_request="Add a healthcheck endpoint",
            task_title="Healthcheck endpoint",
            task_description="Expose /healthz",
            ctx=ctx,
        )
        result_two = await judge_server.set_coding_task(
            user_request="Add a metrics endpoint",
            task_title="Metrics endpoint",
            task_description="Expose /metrics",
            ctx=ctx,
        )

    assert isinstance(result_one, TaskAnalysisResult)
    assert isinstance(result_two, TaskAnalysisResult)
    assert observed_initial_values == [[], []]
    assert observed_ids[0] != observed_ids[1]
