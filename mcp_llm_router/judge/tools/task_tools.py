"""Shared imports for extracted judge tool modules."""

from __future__ import annotations

import builtins
import contextlib
import json
import time

from mcp.server.fastmcp import Context
from pydantic import ValidationError

from mcp_llm_router.judge.core.constants import MAX_TOKENS
from mcp_llm_router.judge.core.logging_config import (
    log_tool_execution,
    set_context_reference,
)
from mcp_llm_router.judge.core.server_helpers import (
    evaluate_coding_plan,
    extract_changed_files,
    extract_json_from_response,
    generate_dynamic_elicitation_model,
    generate_validation_error_message,
    initialize_llm_configuration,
    looks_like_unified_diff,
    validate_research_quality,
    validate_test_output,
)
from mcp_llm_router.judge.db.conversation_history_service import ConversationHistoryService
from mcp_llm_router.judge.db.db_config import load_config
from mcp_llm_router.judge.elicitation import elicitation_provider
from mcp_llm_router.judge.messaging.llm_provider import llm_provider
from mcp_llm_router.judge.models import (
    JudgeCodeChangeUserVars,
    PlanApprovalResponse,
    PlanApprovalResult,
    SystemVars,
)
from mcp_llm_router.judge.models.enhanced_responses import (
    EnhancedResponseFactory,
    JudgeResponse,
    TaskAnalysisResult,
    TaskCompletionResult,
)
from mcp_llm_router.judge.models.task_metadata import (
    TaskMetadata,
    TaskSize,
    TaskState,
)
from mcp_llm_router.judge.prompting.loader import create_separate_messages
from mcp_llm_router.judge.runtime import conversation_service, context_logger, logger
from mcp_llm_router.judge.tasks.manager import (
    create_new_coding_task,
    save_task_metadata_to_history,
    update_existing_coding_task,
)
from mcp_llm_router.judge.tools.common import save_tool_interaction
from mcp_llm_router.judge.workflow import calculate_next_stage
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

async def set_coding_task(
    user_request: str,
    task_title: str,
    task_description: str,
    ctx: Context,
    task_size: TaskSize = TaskSize.M,  # Task size classification (xs, s, m, l, xl) - defaults to Medium for backward compatibility
    # FOR UPDATING EXISTING TASKS ONLY
    task_id: str = "",  # REQUIRED when updating existing task
    user_requirements: str = "",  # Updates current requirements
    state: TaskState = TaskState.CREATED,  # Optional: update task state with validation when updating existing task
    # OPTIONAL
    tags: list[str] | None = None,
) -> TaskAnalysisResult:
    """Create or update coding task metadata with enhanced workflow management."""
    task_id_for_logging = task_id if task_id else "new_task"
    tags = list(tags or [])

    # Set global context reference for system-wide logging
    set_context_reference(ctx)

    # Log tool execution start using context-aware logger
    await context_logger.info(f"set_coding_task called for task: {task_id_for_logging}")

    original_input = {
        "user_request": user_request,
        "task_title": task_title,
        "task_description": task_description,
        "task_id": task_id,
        "user_requirements": user_requirements,
        "tags": tags,
        "state": state.value if isinstance(state, TaskState) else state,
    }

    try:
        if task_id:
            task_metadata = await update_existing_coding_task(
                task_id=task_id,
                user_request=user_request,
                task_title=task_title,
                task_description=task_description,
                user_requirements=user_requirements,
                state=state,  # Allow optional state transition with validation
                tags=tags,
                conversation_service=conversation_service,
            )
            action = "updated"
            context_summary = f"Updated coding task '{task_metadata.title}' (ID: {task_metadata.task_id})"

        else:
            task_metadata = await create_new_coding_task(
                user_request=user_request,
                task_title=task_title,
                task_description=task_description,
                user_requirements=user_requirements if user_requirements else "",
                tags=tags,
                conversation_service=conversation_service,
                task_size=task_size,
            )
            action = "created"
            context_summary = f"Created new coding task '{task_metadata.title}' (ID: {task_metadata.task_id})"

        workflow_guidance = await calculate_next_stage(
            task_metadata=task_metadata,
            current_operation=f"set_coding_task_{action}",
            conversation_service=conversation_service,
            ctx=ctx,
        )

        initial_guidance = workflow_guidance

        # Apply research requirements determined by LLM workflow guidance (for new tasks)
        if action == "created" and initial_guidance.research_required is not None:
            from mcp_llm_router.judge.models.task_metadata import ResearchScope

            task_metadata.research_required = initial_guidance.research_required
            task_metadata.research_rationale = initial_guidance.research_rationale or ""

            # Map research scope string to enum
            if initial_guidance.research_scope:
                scope_mapping = {
                    "none": ResearchScope.NONE,
                    "light": ResearchScope.LIGHT,
                    "deep": ResearchScope.DEEP,
                }
                task_metadata.research_scope = scope_mapping.get(
                    initial_guidance.research_scope.lower(), ResearchScope.NONE
                )

            # Set internal research and risk assessment requirements
            if initial_guidance.internal_research_required is not None:
                task_metadata.internal_research_required = (
                    initial_guidance.internal_research_required
                )
            if initial_guidance.risk_assessment_required is not None:
                task_metadata.risk_assessment_required = (
                    initial_guidance.risk_assessment_required
                )
            if initial_guidance.design_patterns_enforcement is not None:
                task_metadata.design_patterns_enforcement = (
                    initial_guidance.design_patterns_enforcement
                )

            # Update timestamp to reflect changes
            task_metadata.updated_at = int(time.time())

            logger.info(
                f"Applied LLM-determined research requirements: required={task_metadata.research_required}, scope={task_metadata.research_scope}, rationale='{task_metadata.research_rationale}'"
            )

        # Auto-transition all freshly created tasks to planning (unified workflow)
        # so agents aren't forced to call set_coding_task twice in a row. Perform this
        # after applying research flags so we preserve initial guidance data.
        if action == "created" and task_metadata.state == TaskState.CREATED:
            task_metadata.update_state(TaskState.PLANNING)
            context_summary = f"{context_summary} Transitioned state to 'planning' for unified workflow."

            workflow_guidance = await calculate_next_stage(
                task_metadata=task_metadata,
                current_operation="set_coding_task_updated",
                conversation_service=conversation_service,
                ctx=ctx,
            )

        # Save task metadata to conversation history using task_id as primary key
        await save_task_metadata_to_history(
            task_metadata=task_metadata,
            user_request=user_request,
            action=action,
            conversation_service=conversation_service,
        )

        result = EnhancedResponseFactory.create_task_analysis_result(
            action=action,
            context_summary=context_summary,
            current_task_metadata=task_metadata,
            workflow_guidance=workflow_guidance,
        )

        await save_tool_interaction(
            session_id=task_metadata.task_id,
            tool_name="set_coding_task",
            tool_input=original_input,
            tool_output=result,
        )

        return result

    except Exception as e:
        # Create error response
        error_metadata = TaskMetadata(
            title=task_title,
            description=task_description,
            user_requirements=user_requirements if user_requirements else "",
            state=TaskState.CREATED,
            task_size=TaskSize.M,
            tags=tags,
        )

        error_guidance = WorkflowGuidance(
            next_tool="get_current_coding_task",
            reasoning="Task update failed or task_id not found; retrieve the latest valid task_id and metadata.",
            preparation_needed=[
                "Call get_current_coding_task to fetch active task_id",
                "Retry with the returned task_id if needed",
            ],
            guidance=(
                f"Error occurred: {e!s}. Use get_current_coding_task to retrieve the most recent task_id, then retry the operation with that ID."
            ),
        )

        error_result = EnhancedResponseFactory.create_task_analysis_result(
            action="error",
            context_summary=f"Error creating/updating task: {e!s}",
            current_task_metadata=error_metadata,
            workflow_guidance=error_guidance,
        )

        # Save error interaction (use task_id if available, otherwise generate one for logging)
        error_task_id = task_id if task_id else error_metadata.task_id
        await save_tool_interaction(
            session_id=error_task_id,
            tool_name="set_coding_task",
            tool_input=original_input,
            tool_output=error_result,
            suppress_errors=True,
        )

        return error_result


async def get_current_coding_task(ctx: Context) -> dict:
    """Return the most recently active coding task's task_id and metadata.

    Use this when you lost the UUID. If none exists, the response includes
    guidance to call set_coding_task to create a new task.
    """
    set_context_reference(ctx)
    log_tool_execution("get_current_coding_task", "unknown")

    try:
        recent = await conversation_service.db.get_recent_sessions(limit=1)
        if not recent:
            return {
                "found": False,
                "feedback": "No existing coding task sessions found. Call set_coding_task to create a task and obtain a task_id UUID.",
                "workflow_guidance": {
                    "next_tool": "set_coding_task",
                    "reasoning": "No recent sessions in conversation history",
                    "preparation_needed": [
                        "Provide user_request, task_title, task_description"
                    ],
                    "guidance": "Call set_coding_task to initialize a new task. Use the returned task_id UUID in all subsequent tool calls.",
                },
            }

        task_id, last_activity = recent[0]

        # Load task metadata from history if available
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_metadata = await load_task_metadata_from_history(
            task_id=task_id, conversation_service=conversation_service
        )

        response: dict = {
            "found": True,
            "task_id": task_id,
            "last_activity": last_activity,
        }

        if task_metadata is not None:
            response["current_task_metadata"] = task_metadata.model_dump(
                mode="json",
                exclude_unset=True,
                exclude_none=True,
                exclude_defaults=True,
            )

            # Generate workflow guidance for the current task state
            from mcp_llm_router.judge.workflow.workflow_guidance import (
                calculate_next_stage,
            )

            workflow_guidance = await calculate_next_stage(
                task_metadata=task_metadata,
                current_operation="get_current_coding_task_found",
                conversation_service=conversation_service,
                ctx=ctx,
            )

            response["workflow_guidance"] = workflow_guidance.model_dump(
                mode="json",
                exclude_unset=True,
                exclude_none=True,
                exclude_defaults=True,
            )
        else:
            response["note"] = (
                "Task metadata not found in history for this session, but a session exists. Use this task_id UUID and proceed; if validation fails, recreate with set_coding_task."
            )
            # Provide basic workflow guidance even without metadata
            response["workflow_guidance"] = {
                "next_tool": "set_coding_task",
                "reasoning": "Task metadata not found in history, may need to recreate task",
                "preparation_needed": [
                    "Verify task_id is correct",
                    "If validation fails, recreate with set_coding_task",
                ],
                "guidance": "Try using this task_id with other tools. If validation fails, call set_coding_task to recreate the task with proper metadata.",
            }

        return response
    except Exception as e:
        return {
            "found": False,
            "error": f"Failed to retrieve current task: {e!s}",
            "workflow_guidance": {
                "next_tool": "set_coding_task",
                "reasoning": "Error while retrieving recent sessions",
                "preparation_needed": [
                    "Provide user_request, task_title, task_description"
                ],
                "guidance": "Call set_coding_task to initialize a new task and use its task_id UUID going forward.",
            },
        }
