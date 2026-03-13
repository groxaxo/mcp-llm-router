from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_llm_router.judge.models.enhanced_responses import TaskCompletionResult
from mcp_llm_router.judge.models.task_metadata import TaskMetadata, TaskSize, TaskState
from mcp_llm_router.judge.tools import common
from mcp_llm_router.judge.workflow import calculate_next_stage
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance


@pytest.mark.anyio
async def test_load_task_metadata_with_history_debug_logs_recent_entries():
    fake_entry = SimpleNamespace(
        source="set_coding_task",
        timestamp=1234567890,
        input="{}",
        output="{}",
    )
    logger_mock = MagicMock()

    with (
        patch(
            "mcp_llm_router.judge.tasks.manager.load_task_metadata_from_history",
            AsyncMock(return_value=None),
        ),
        patch.object(
            common.conversation_service,
            "load_filtered_context_for_enrichment",
            AsyncMock(return_value=[fake_entry]),
        ),
        patch.object(common, "logger", logger_mock),
    ):
        result = await common.load_task_metadata_with_history_debug(
            task_id="task-123",
            ctx=MagicMock(),
            log_prefix="judge_code_change",
        )

    assert result is None
    assert logger_mock.info.call_count >= 3


@pytest.mark.anyio
async def test_save_tool_interaction_serializes_model_output():
    result = TaskCompletionResult(
        approved=True,
        feedback="done",
        required_improvements=[],
        current_task_metadata=TaskMetadata(
            title="Task",
            description="Desc",
            task_size=TaskSize.M,
        ),
        workflow_guidance=WorkflowGuidance(
            next_tool=None,
            reasoning="Complete",
            preparation_needed=[],
            guidance="No further action required.",
        ),
    )

    mock_save = AsyncMock()
    with patch.object(
        common.conversation_service,
        "save_tool_interaction_and_cleanup",
        mock_save,
    ):
        await common.save_tool_interaction(
            session_id="task-123",
            tool_name="judge_coding_task_completion",
            tool_input={"task_id": "task-123"},
            tool_output=result,
        )

    kwargs = mock_save.await_args.kwargs
    assert kwargs["session_id"] == "task-123"
    assert '"approved": true' in kwargs["tool_output"]


def test_task_metadata_approval_state_transitions():
    metadata = TaskMetadata(
        title="Task",
        description="Desc",
        task_size=TaskSize.M,
        state=TaskState.PLAN_APPROVED,
    )
    metadata.modified_files = ["src/app.py"]
    metadata.test_files = ["tests/test_app.py"]

    readiness = metadata.validate_completion_readiness()
    assert readiness["ready_for_completion"] is False

    metadata.mark_plan_approved()
    metadata.mark_code_approved("src/app.py")
    metadata.mark_testing_approved()

    readiness = metadata.validate_completion_readiness()
    assert readiness["ready_for_completion"] is True
    assert readiness["missing_approvals"] == []


@pytest.mark.anyio
async def test_calculate_next_stage_routes_review_ready_to_testing():
    metadata = TaskMetadata(
        title="Task",
        description="Desc",
        task_size=TaskSize.M,
        state=TaskState.REVIEW_READY,
    )

    guidance = await calculate_next_stage(
        task_metadata=metadata,
        current_operation="set_coding_task_updated",
        conversation_service=MagicMock(),
        ctx=None,
    )

    assert guidance.next_tool == "judge_testing_implementation"
