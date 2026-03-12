"""Shared helpers for extracted judge tool modules."""

from __future__ import annotations

import builtins
import contextlib
import json
from typing import Any

from mcp.server.fastmcp import Context

from mcp_llm_router.judge.models.task_metadata import TaskMetadata, TaskSize, TaskState
from mcp_llm_router.judge.runtime import conversation_service, logger
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance


def build_error_guidance(
    *,
    reasoning: str,
    guidance: str,
    next_tool: str | None = None,
    preparation_needed: list[str] | None = None,
) -> WorkflowGuidance:
    return WorkflowGuidance(
        next_tool=next_tool,
        reasoning=reasoning,
        preparation_needed=preparation_needed or [],
        guidance=guidance,
    )


def build_fallback_task_metadata(
    *,
    title: str,
    description: str,
    user_requirements: str = "",
    state: TaskState,
    tags: list[str] | None = None,
) -> TaskMetadata:
    return TaskMetadata(
        title=title,
        description=description,
        user_requirements=user_requirements,
        state=state,
        task_size=TaskSize.M,
        tags=tags or [],
    )


def serialize_tool_output(output: Any) -> str:
    if hasattr(output, "model_dump"):
        payload = output.model_dump(
            mode="json",
            exclude_unset=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        return json.dumps(payload)
    if isinstance(output, (dict, list)):
        return json.dumps(output)
    return json.dumps({"message": str(output)})


async def save_tool_interaction(
    *,
    session_id: str,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    suppress_errors: bool = False,
) -> None:
    async def _save() -> None:
        await conversation_service.save_tool_interaction_and_cleanup(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input),
            tool_output=serialize_tool_output(tool_output),
        )

    if suppress_errors:
        with contextlib.suppress(builtins.BaseException):
            await _save()
        return

    await _save()


async def load_task_metadata_with_history_debug(
    *,
    task_id: str,
    ctx: Context,
    log_prefix: str,
) -> TaskMetadata | None:
    from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

    logger.info(f"{log_prefix}: Loading task metadata for task_id: {task_id}")
    task_metadata = await load_task_metadata_from_history(
        task_id=task_id,
        conversation_service=conversation_service,
    )

    logger.info(f"{log_prefix}: Task metadata loaded: {task_metadata is not None}")
    if task_metadata:
        logger.info(
            f"{log_prefix}: Task state: {task_metadata.state}, title: {task_metadata.title}"
        )
        return task_metadata

    conversation_history = await conversation_service.load_filtered_context_for_enrichment(
        task_id, "", ctx
    )
    logger.info(f"{log_prefix}: Conversation history entries: {len(conversation_history)}")
    for entry in conversation_history[-5:]:
        logger.info(f"{log_prefix}: History entry: {entry.source} at {entry.timestamp}")
    return None


async def load_recent_history_as_json(
    *,
    task_id: str,
    ctx: Context,
    limit: int = 10,
) -> list[dict[str, Any]]:
    conversation_history = await conversation_service.load_filtered_context_for_enrichment(
        task_id, "", ctx
    )
    return [
        {
            "timestamp": entry.timestamp,
            "tool": entry.source,
            "input": entry.input,
            "output": entry.output,
        }
        for entry in conversation_history[-limit:]
    ]
