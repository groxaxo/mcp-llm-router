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
from mcp_llm_router.judge.workflow import calculate_next_stage
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

async def raise_obstacle(
    problem: str,
    research: str,
    options: list[str],
    ctx: Context,
    task_id: str = "",  # OPTIONAL: Task ID for context and memory
    # Optional HITL assistance inputs
    decision_area: str = "",
    constraints: list[str] | None = None,
) -> str:
    """Obstacle handling tool - description loaded from tool_description_provider."""
    # Log tool execution start
    log_tool_execution("raise_obstacle", task_id if task_id else "unknown")
    constraints = list(constraints or [])

    # Store original input for saving later
    original_input = {
        "problem": problem,
        "research": research,
        "options": options,
        "task_id": task_id,
        "decision_area": decision_area,
        "constraints": constraints,
    }

    try:
        # Load task metadata to get current context
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_metadata = await load_task_metadata_from_history(
            task_id=task_id if task_id else "test_task",
            conversation_service=conversation_service,
        )

        if not task_metadata:
            # Create minimal task metadata for obstacle handling
            task_metadata = TaskMetadata(
                title="Obstacle Resolution",
                description=f"Handling obstacle: {problem}",
                user_requirements="Resolve obstacle to continue task",
                state=TaskState.BLOCKED,
                task_size=TaskSize.M,
                tags=["obstacle"],
            )

        # Update task state to BLOCKED
        task_metadata.update_state(TaskState.BLOCKED)

        formatted_options = "\n".join(
            f"{i + 1}. {option}" for i, option in enumerate(options)
        )

        context_info = (
            "Agent encountered an obstacle and needs user decision on how to proceed"
        )
        info_extra = []
        if decision_area:
            info_extra.append(f"Decision area: {decision_area}")
        if constraints:
            info_extra.append("Constraints: " + ", ".join(constraints))
        information_needed = (
            "User needs to choose from available options and provide any additional context"
            + (". " + "; ".join(info_extra) if info_extra else "")
        )
        current_understanding = (
            f"Problem: {problem}. Research: {research}. Options: {formatted_options}"
        )

        dynamic_model = await generate_dynamic_elicitation_model(
            context_info, information_needed, current_understanding, ctx
        )

        # Use elicitation provider with capability checking
        elicit_result = await elicitation_provider.elicit_user_input(
            message=f"""OBSTACLE ENCOUNTERED

Problem: {problem}

Research Done: {research}

Available Options:
{formatted_options}

Decision Area: {decision_area if decision_area else "Not specified"}

Constraints:
{chr(10).join(f"- {c}" for c in constraints) if constraints else "None provided"}

Please choose an option (by number or description) and provide any additional context or modifications you'd like.""",
            schema=dynamic_model,
            ctx=ctx,
        )

        if elicit_result.success:
            # Handle successful elicitation response
            user_response = elicit_result.data

            # Ensure user_response is a dictionary
            if not isinstance(user_response, dict):
                user_response = {"user_input": str(user_response)}  # type: ignore[unreachable]

            # Format the response data for display
            response_summary = []
            for field_name, field_value in user_response.items():
                if field_value:  # Only include non-empty values
                    formatted_key = field_name.replace("_", " ").title()
                    response_summary.append(f"**{formatted_key}:** {field_value}")

            response_text = (
                "\n".join(response_summary)
                if response_summary
                else "User provided response"
            )

            # HITL tools should always direct to set_coding_task to update requirements
            workflow_guidance = WorkflowGuidance(
                next_tool="set_coding_task",
                reasoning="Obstacle resolved through user interaction. Task requirements may need updating based on the resolution.",
                preparation_needed=[
                    "Review the obstacle resolution",
                    "Update task requirements if needed",
                ],
                guidance="Call set_coding_task to update the task with any new requirements or clarifications from the obstacle resolution. Then continue with the workflow.",
            )

            # Create resolution text
            result_text = f"✅ OBSTACLE RESOLVED: {response_text}"

            # Save successful interaction as conversation record
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_metadata.task_id,  # Use task_id as primary key
                tool_name="raise_obstacle",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {"obstacle_acknowledged": True, "message": result_text}
                ),
            )

            return result_text

        else:
            # Elicitation failed or not available - return the fallback message
            workflow_guidance = WorkflowGuidance(
                next_tool=None,
                reasoning="Obstacle elicitation failed or unavailable",
                preparation_needed=["Manual intervention required"],
                guidance=f"Obstacle not resolved: {elicit_result.message}. Manual intervention required.",
            )

            # Save failed interaction
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_metadata.task_id,  # Use task_id as primary key
                tool_name="raise_obstacle",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {"obstacle_acknowledged": False, "message": elicit_result.message}
                ),
            )

            return (
                f"❌ ERROR: Failed to elicit user decision: {elicit_result.message}. "
                f"No messaging providers available"
            )

    except Exception as e:
        # Create error response
        error_guidance = WorkflowGuidance(
            next_tool=None,
            reasoning="Error occurred while handling obstacle",
            preparation_needed=["Review error details", "Manual intervention required"],
            guidance=f"Error handling obstacle: {e!s}. Manual intervention required.",
        )

        # Save error interaction
        with contextlib.suppress(builtins.BaseException):
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_metadata.task_id
                if "task_metadata" in locals() and task_metadata
                else (task_id if task_id else "unknown"),
                tool_name="raise_obstacle",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {
                        "obstacle_acknowledged": False,
                        "message": f"❌ ERROR: Failed to elicit user decision. Error: {e!s}. Cannot resolve obstacle without user input.",
                    }
                ),
            )

        return (
            f"❌ ERROR: Failed to elicit user decision. Error: {e!s}. "
            f"No messaging providers available"
        )


async def raise_missing_requirements(
    current_request: str,
    identified_gaps: list[str],
    specific_questions: list[str],
    task_id: str,  # REQUIRED: Task ID for context and memory
    ctx: Context,
    # Optional HITL assistance inputs
    decision_areas: list[str] | None = None,
    options: list[str] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """Requirements clarification tool - description loaded from tool_description_provider."""
    # Log tool execution start
    log_tool_execution("raise_missing_requirements", task_id)
    decision_areas = list(decision_areas or [])
    options = list(options or [])
    constraints = list(constraints or [])

    # Store original input for saving later
    original_input = {
        "current_request": current_request,
        "identified_gaps": identified_gaps,
        "specific_questions": specific_questions,
        "task_id": task_id,
        "decision_areas": decision_areas,
        "options": options,
        "constraints": constraints,
    }

    try:
        # Load task metadata to get current context
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_metadata = await load_task_metadata_from_history(
            task_id=task_id,
            conversation_service=conversation_service,
        )

        if not task_metadata:
            # Create minimal task metadata for requirements clarification
            task_metadata = TaskMetadata(
                title="Requirements Clarification",
                description=f"Clarifying requirements: {current_request}",
                user_requirements=current_request,
                state=TaskState.CREATED,
                task_size=TaskSize.M,
                tags=["requirements"],
            )

        # Format the gaps and questions for clarity
        formatted_gaps = "\n".join(f"• {gap}" for gap in identified_gaps)
        formatted_questions = "\n".join(
            f"{i + 1}. {question}" for i, question in enumerate(specific_questions)
        )

        context_info = "Agent needs clarification on user requirements and confirmation of key decisions to proceed"
        info_extra = []
        if decision_areas:
            info_extra.append("Decisions to confirm: " + ", ".join(decision_areas))
        if constraints:
            info_extra.append("Constraints: " + ", ".join(constraints))
        information_needed = (
            "Clarified requirements, answers to specific questions, and priority levels"
            + (". " + "; ".join(info_extra) if info_extra else "")
        )
        current_understanding = (
            f"Current request: {current_request}. Gaps: {formatted_gaps}. Questions: {formatted_questions}"
            + (f". Candidate options: {'; '.join(options or [])}" if options else "")
        )

        dynamic_model = await generate_dynamic_elicitation_model(
            context_info, information_needed, current_understanding, ctx
        )

        # Use elicitation provider with capability checking
        elicit_result = await elicitation_provider.elicit_user_input(
            message=f"""REQUIREMENTS CLARIFICATION NEEDED

Current Understanding: {current_request}

Identified Requirement Gaps:
{formatted_gaps}

Specific Questions:
{formatted_questions}

Decisions To Confirm:
{chr(10).join(f"- {a}" for a in decision_areas) if decision_areas else "None provided"}

Candidate Options:
{chr(10).join(f"- {o}" for o in options) if options else "None provided"}

Constraints:
{chr(10).join(f"- {c}" for c in constraints) if constraints else "None provided"}

Please provide clarified requirements and indicate their priority level (high/medium/low).""",
            schema=dynamic_model,
            ctx=ctx,
        )

        if elicit_result.success:
            # Handle successful elicitation response
            user_response = elicit_result.data

            # Ensure user_response is a dictionary
            if not isinstance(user_response, dict):
                user_response = {"user_input": str(user_response)}  # type: ignore[unreachable]

            # Format the response data for display
            response_summary = []
            for field_name, field_value in user_response.items():
                if field_value:  # Only include non-empty values
                    formatted_key = field_name.replace("_", " ").title()
                    response_summary.append(f"**{formatted_key}:** {field_value}")

            response_text = (
                "\n".join(response_summary)
                if response_summary
                else "User provided clarifications"
            )

            # Update task metadata with clarified requirements
            clarified_requirements = (
                f"{current_request}\n\nClarifications: {response_text}"
            )
            task_metadata.update_requirements(
                clarified_requirements, source="clarification"
            )

            # HITL tools should always direct to set_coding_task to update requirements

            # Save successful interaction
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_id,  # Use task_id as primary key
                tool_name="raise_missing_requirements",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {
                        "clarification_needed": False,
                        "message": f"✅ REQUIREMENTS CLARIFIED: {response_text}",
                    }
                ),
            )
            return f"✅ REQUIREMENTS CLARIFIED: {response_text}"

        else:
            # Elicitation failed or not available - return the fallback message

            # Save failed interaction
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_id,  # Use task_id as primary key
                tool_name="raise_missing_requirements",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {
                        "clarification_needed": True,
                        "message": elicit_result.message,
                    }
                ),
            )
            return (
                f"❌ ERROR: Failed to elicit requirement clarifications. Error: {elicit_result.message}. "
                f"No messaging providers available"
            )

    except Exception as e:
        # Create error response

        # Save error interaction
        with contextlib.suppress(builtins.BaseException):
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=task_id,  # Use task_id as primary key
                tool_name="raise_missing_requirements",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    {
                        "clarification_needed": True,
                        "message": f"❌ ERROR: Failed to elicit requirement clarifications. Error: {e!s}. Cannot proceed without clear requirements.",
                    }
                ),
            )
        # Ensure we have non-None metadata for typing
        if "task_metadata" not in locals() or task_metadata is None:
            task_metadata = TaskMetadata(
                title="Requirements Clarification",
                description=f"Clarifying requirements: {current_request}",
                user_requirements=current_request,
                state=TaskState.CREATED,
                task_size=TaskSize.M,
                tags=["requirements"],
            )

        return (
            f"❌ ERROR: Failed to elicit requirement clarifications. Error: {e!s}. "
            f"No messaging providers available"
        )
