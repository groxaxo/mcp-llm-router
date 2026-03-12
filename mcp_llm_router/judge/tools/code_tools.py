"""Judge code review and completion tools."""

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
from mcp_llm_router.judge.roots import find_paths_outside_roots
from mcp_llm_router.judge.runtime import conversation_service, context_logger, logger
from mcp_llm_router.judge.tasks.manager import (
    create_new_coding_task,
    save_task_metadata_to_history,
    update_existing_coding_task,
)
from mcp_llm_router.judge.tools.common import (
    build_error_guidance,
    build_fallback_task_metadata,
    load_task_metadata_with_history_debug,
    save_tool_interaction,
)
from mcp_llm_router.judge.workflow import calculate_next_stage
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

async def judge_coding_task_completion(
    task_id: str,  # REQUIRED: Task ID for context and validation
    completion_summary: str,
    requirements_met: list[str],
    implementation_details: str,
    ctx: Context,
    # OPTIONAL
    remaining_work: list[str] | None = None,
    quality_notes: str = "",
    testing_status: str = "",
) -> TaskCompletionResult:
    """Final validation tool for coding task completion."""
    # Log tool execution start
    log_tool_execution("judge_coding_task_completion", task_id)
    remaining_work = list(remaining_work or [])

    # Store original input for saving later
    original_input = {
        "task_id": task_id,
        "completion_summary": completion_summary,
        "requirements_met": requirements_met,
        "implementation_details": implementation_details,
        "remaining_work": remaining_work,
        "quality_notes": quality_notes,
        "testing_status": testing_status,
    }

    try:
        task_metadata = await load_task_metadata_with_history_debug(
            task_id=task_id,
            ctx=ctx,
            log_prefix="judge_coding_task_completion",
        )

        if not task_metadata:
            task_metadata = build_fallback_task_metadata(
                title="Unknown Task",
                description="Task metadata could not be loaded from history",
                user_requirements="Task requirements not found",
                state=TaskState.COMPLETED,  # Appropriate state for completion check
                tags=["debug", "missing-metadata"],
            )

            # Return debug information
            return TaskCompletionResult(
                approved=False,
                feedback=f"Task {task_id} not found in conversation history. This usually means set_coding_task was not called first, or the server was restarted and lost the in-memory data.",
                current_task_metadata=task_metadata,
                workflow_guidance=WorkflowGuidance(
                    next_tool="get_current_coding_task",
                    reasoning="Task metadata not found; recover the active task context before proceeding.",
                    preparation_needed=[
                        "Call get_current_coding_task to fetch the active task_id",
                        "Retry completion or proceed per recovered state",
                    ],
                    guidance=(
                        "Use get_current_coding_task to retrieve the most recent task_id and metadata. Then continue the workflow based on the recovered state (typically judge_code_change → judge_testing_implementation → judge_coding_task_completion)."
                    ),
                ),
            )

        # STEP 1: Validate approvals from judge tools
        completion_readiness = task_metadata.validate_completion_readiness()
        approval_status = completion_readiness["approval_status"]
        missing_approvals = completion_readiness["missing_approvals"]

        # STEP 2: Check if all requirements are met
        has_remaining_work = remaining_work and len(remaining_work) > 0
        requirements_coverage = len(requirements_met) > 0

        # STEP 3: Determine if task is complete (now includes approval validation)
        task_complete = (
            completion_readiness["ready_for_completion"]  # All approvals validated
            and requirements_coverage
            and not has_remaining_work
            and completion_summary.strip() != ""
        )

        if task_complete:
            # Task is complete - update state to COMPLETED
            task_metadata.update_state(TaskState.COMPLETED)

            feedback = f"""✅ TASK COMPLETION APPROVED

**Completion Summary:** {completion_summary}

**Requirements Satisfied:**
{chr(10).join(f"• {req}" for req in requirements_met)}

**Implementation Details:** {implementation_details}

**✅ APPROVAL VALIDATION PASSED:**
• Plan Approved: {"✅" if approval_status["plan_approved"] else "❌"} {f"({approval_status['plan_approved_at']})" if approval_status["plan_approved_at"] else ""}
• Code Files Approved: {"✅" if approval_status["all_modified_files_approved"] else "❌"} ({approval_status["code_files_approved"]}/{len(task_metadata.modified_files)} files)
• Testing Approved: {"✅" if approval_status["testing_approved"] else "❌"} {f"({approval_status['testing_approved_at']})" if approval_status["testing_approved_at"] else ""}"""

            if quality_notes:
                feedback += f"\n\n**Quality Notes:** {quality_notes}"

            if testing_status:
                feedback += f"\n\n**Testing Status:** {testing_status}"

            feedback += (
                "\n\n🎉 **Task successfully completed with all required approvals!**"
            )

            # Update state to COMPLETED when task completion is approved
            task_metadata.update_state(TaskState.COMPLETED)

            workflow_guidance = await calculate_next_stage(
                task_metadata=task_metadata,
                current_operation="judge_coding_task_completion_approved",
                conversation_service=conversation_service,
                ctx=ctx,
            )

            result = TaskCompletionResult(
                approved=True,
                feedback=feedback,
                required_improvements=[],
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

        else:
            # Task is not complete - provide guidance for remaining work
            task_metadata.update_state(TaskState.IMPLEMENTING)

            feedback = f"""⚠️ TASK COMPLETION NOT APPROVED

**Current Progress:** {completion_summary}

**Requirements Satisfied:**
{chr(10).join(f"• {req}" for req in requirements_met) if requirements_met else "• None specified"}"""

            required_improvements = []

            # APPROVAL VALIDATION FAILURES
            if not completion_readiness["ready_for_completion"]:
                feedback += "\n\n**❌ APPROVAL VALIDATION FAILED:**"
                feedback += f"\n{completion_readiness['validation_message']}"
                feedback += "\n\n**Missing Approvals:**"
                for missing in missing_approvals:
                    feedback += f"\n• {missing}"
                required_improvements.extend(missing_approvals)

                # Detailed approval status
                feedback += "\n\n**Current Approval Status:**"
                feedback += f"\n• Plan Approved: {'✅' if approval_status['plan_approved'] else '❌'}"
                feedback += f"\n• Code Files Approved: {approval_status['code_files_approved']}/{len(task_metadata.modified_files)} files"
                feedback += f"\n• Testing Approved: {'✅' if approval_status['testing_approved'] else '❌'}"

            # OTHER COMPLETION ISSUES
            if has_remaining_work and remaining_work:
                feedback += f"\n\n**Remaining Work:**\n{chr(10).join(f'• {work}' for work in remaining_work)}"
                required_improvements.extend(remaining_work)

            if not requirements_coverage:
                feedback += "\n\n**Issue:** No requirements marked as satisfied"
                required_improvements.append("Specify which requirements have been met")

            if not completion_summary.strip():
                feedback += "\n\n**Issue:** No completion summary provided"
                required_improvements.append("Provide a detailed completion summary")

            feedback += "\n\n📋 **Complete all required approvals and remaining work before resubmitting for final approval.**"

            # Deterministic next step based on missing approvals
            next_tool = None
            if any("plan approval" in m for m in missing_approvals):
                next_tool = "judge_coding_plan"
            elif any("code approval" in m for m in missing_approvals) or (
                approval_status.get("code_files_approved", 0)
                < len(task_metadata.modified_files or [])
            ):
                next_tool = "judge_code_change"
            elif any("testing approval" in m for m in missing_approvals):
                next_tool = "judge_testing_implementation"
            else:
                # Default to code review gate for safety
                next_tool = "judge_code_change"

            # Construct guidance tailored to the required step
            if next_tool == "judge_coding_plan":
                reasoning = "Missing plan approval; revise and resubmit the plan."
                prep = [
                    "Address required improvements in the plan",
                    "Ensure design, file list, and research coverage are complete",
                ]
                guidance = "Update the plan addressing all feedback and call judge_coding_plan. After approval, proceed to implementation and code review."
            elif next_tool == "judge_code_change":
                reasoning = "Code has not been reviewed/approved; submit implementation for review."
                prep = [
                    "Implement or finalize code changes per requirements",
                    "Prepare file paths and a concise change summary",
                ]
                guidance = "Call judge_code_change with the modified files and a concise summary or diff. After approval, implement/verify tests and validate via judge_testing_implementation."
            else:  # judge_testing_implementation
                reasoning = "Testing approval missing; run and validate tests."
                prep = [
                    "Run the test suite and capture results",
                    "Provide coverage details if available",
                ]
                guidance = "Call judge_testing_implementation with test files, execution results, and coverage details. After approval, resubmit completion."

            workflow_guidance = WorkflowGuidance(
                next_tool=next_tool,
                reasoning=reasoning,
                preparation_needed=prep,
                guidance=guidance,
            )

            result = TaskCompletionResult(
                approved=False,
                feedback=feedback,
                required_improvements=required_improvements,
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

        # Save successful interaction
        await save_tool_interaction(
            session_id=task_id,
            tool_name="judge_coding_task_completion",
            tool_input=original_input,
            tool_output=result,
        )

        return result

    except Exception as e:
        # Create error response
        error_guidance = build_error_guidance(
            reasoning="Error occurred; recover active task context and continue with the correct step.",
            next_tool="get_current_coding_task",
            preparation_needed=[
                "Call get_current_coding_task to fetch active task_id",
                "Retry with the returned task_id or proceed based on recovered state",
            ],
            guidance=f"Error validating task completion: {e!s}. Use get_current_coding_task to recover the current task_id and continue the workflow (judge_code_change → judge_testing_implementation → judge_coding_task_completion).",
        )

        # Create minimal task metadata for error case
        if "task_metadata" in locals() and task_metadata is not None:
            error_metadata = task_metadata
        else:
            error_metadata = build_fallback_task_metadata(
                title="Error Task",
                description="Error occurred during completion validation",
                user_requirements="Error occurred before task metadata could be loaded",
                state=TaskState.IMPLEMENTING,
                tags=["error"],
            )

        error_result = TaskCompletionResult(
            approved=False,
            feedback=f"❌ ERROR: Failed to validate task completion. Error: {e!s}",
            required_improvements=["Fix the error and try again"],
            current_task_metadata=error_metadata,
            workflow_guidance=error_guidance,
        )

        # Save error interaction
        await save_tool_interaction(
            session_id=task_id,
            tool_name="judge_coding_task_completion",
            tool_input=original_input,
            tool_output=error_result,
            suppress_errors=True,
        )

        return error_result


async def judge_code_change(
    code_change: str,
    ctx: Context,
    file_path: str = "File path not specified",
    change_description: str = "Change description not provided",
    task_id: str = "",
    # OPTIONAL override
    user_requirements: str = "",
) -> JudgeResponse:
    """Code change evaluation tool - description loaded from tool_description_provider."""
    # Log tool execution start
    log_tool_execution("judge_code_change", task_id if task_id else "test_task")

    # Store original input for saving later
    original_input = {
        "task_id": task_id if task_id else "test_task",
        "code_change": code_change,
        "file_path": file_path,
        "change_description": change_description,
    }

    try:
        task_metadata = await load_task_metadata_with_history_debug(
            task_id=task_id or "test_task",
            ctx=ctx,
            log_prefix="judge_code_change",
        )

        if not task_metadata:
            task_metadata = build_fallback_task_metadata(
                title="Unknown Task",
                description="Task metadata could not be loaded from history",
                user_requirements="Task requirements not found",
                state=TaskState.IMPLEMENTING,
                tags=["debug", "missing-metadata"],
            )

            # Return debug information
            return JudgeResponse(
                approved=False,
                required_improvements=["Task not found in conversation history"],
                feedback=f"Task {task_id} not found in conversation history. This usually means set_coding_task was not called first, or the server was restarted and lost the in-memory data.",
                current_task_metadata=task_metadata,
                workflow_guidance=WorkflowGuidance(
                    next_tool="set_coding_task",
                    reasoning="Task metadata not found in history",
                    preparation_needed=[
                        "Call set_coding_task first to create the task"
                    ],
                    guidance="You must call set_coding_task before calling judge_code_change. The task_id must come from a successful set_coding_task call.",
                ),
            )

        # Transition to IMPLEMENTING state when implementation starts
        if task_metadata.state == TaskState.PLAN_APPROVED:
            task_metadata.update_state(TaskState.IMPLEMENTING)

        # Derive user requirements from task metadata (allow override)
        user_requirements = (
            user_requirements
            if user_requirements is not None
            else task_metadata.user_requirements
        )

        # QUICK VALIDATION: Require a unified Git diff to avoid generic approvals
        if not looks_like_unified_diff(code_change):
            # Do not proceed to LLM; return actionable guidance to provide a diff
            guidance = WorkflowGuidance(
                next_tool="judge_code_change",
                reasoning=(
                    "Code review requires a unified Git diff to evaluate specific changes."
                ),
                preparation_needed=[
                    "Generate a unified Git diff (e.g., `git diff`)",
                    "Include all relevant files in one patch",
                    "Pass it to judge_code_change as `code_change`",
                ],
                guidance=(
                    "Provide a unified Git diff patch of your changes. Avoid narrative summaries. "
                    "Example: run `git diff --unified` and pass the output."
                ),
            )
            return JudgeResponse(
                approved=False,
                required_improvements=[
                    "Provide a unified Git diff patch of the changes for review"
                ],
                feedback=(
                    "The input to judge_code_change must be a unified Git diff (with 'diff --git', '---', '+++', '@@'). "
                    "Received non-diff content; cannot perform a precise code review."
                ),
                current_task_metadata=task_metadata,
                workflow_guidance=guidance,
            )

        # STEP 1: Load conversation history and format as JSON array
        conversation_history = (
            await conversation_service.load_filtered_context_for_enrichment(
                task_id or "test_task", "", ctx
            )
        )
        history_json_array = (
            conversation_service.format_conversation_history_as_json_array(
                conversation_history
            )
        )

        # Extract changed files from unified diff for logging/validation
        changed_files = extract_changed_files(code_change)
        logger.info(
            f"judge_code_change: Files detected in diff ({len(changed_files)}): {', '.join(changed_files)}"
        )
        root_violations = await find_paths_outside_roots(
            changed_files + ([file_path] if file_path else []),
            ctx,
        )
        if root_violations:
            guidance = WorkflowGuidance(
                next_tool="judge_code_change",
                reasoning="One or more submitted paths fall outside the client roots exposed to the judge.",
                preparation_needed=[
                    "Regenerate the diff using only files inside the allowed workspace roots",
                    "Remove any path traversal segments or absolute paths outside the workspace",
                ],
                guidance=(
                    "The following paths are outside the allowed roots: "
                    + ", ".join(sorted(set(root_violations)))
                ),
            )
            return JudgeResponse(
                approved=False,
                required_improvements=[
                    "Limit the review submission to files inside the allowed MCP roots"
                ],
                feedback=guidance.guidance,
                current_task_metadata=task_metadata,
                workflow_guidance=guidance,
            )

        # STEP 2: Create system and user messages with separate context and conversation history
        system_vars = SystemVars(
            response_schema=json.dumps(JudgeResponse.model_json_schema()),
            max_tokens=MAX_TOKENS,
        )
        user_vars = JudgeCodeChangeUserVars(
            user_requirements=user_requirements,
            code_change=code_change,
            file_path=file_path,
            change_description=change_description,
            context="",  # Empty context for now - can be enhanced later
            conversation_history=history_json_array,  # JSON array with timestamps
        )
        messages = create_separate_messages(
            "system/judge_code_change.md",
            "user/judge_code_change.md",
            system_vars,
            user_vars,
        )

        # STEP 3: Use messaging layer for LLM evaluation
        response_text = await llm_provider.send_message(
            messages=messages,
            ctx=ctx,
            max_tokens=MAX_TOKENS,
            prefer_sampling=True,
        )

        # Parse the JSON response
        try:
            json_content = extract_json_from_response(response_text)
            judge_result = JudgeResponse.model_validate_json(json_content)

            # Enforce per-file coverage: every changed file must have a reviewed_files entry
            try:
                reviewed_paths = {
                    rf.path for rf in getattr(judge_result, "reviewed_files", [])
                }
            except Exception:
                reviewed_paths = set()
            missing_reviews = [p for p in changed_files if p not in reviewed_paths]
            if missing_reviews:
                logger.warning(
                    f"judge_code_change: Missing per-file reviews for: {', '.join(missing_reviews)}"
                )
                guidance = WorkflowGuidance(
                    next_tool="judge_code_change",
                    reasoning=(
                        "Per-file coverage incomplete: every changed file must be reviewed"
                    ),
                    preparation_needed=[
                        "Enumerate all changed files from the diff",
                        "Add a reviewed_files entry for each with per-file feedback",
                    ],
                    guidance=(
                        "Update the response to include reviewed_files entries for all missing files: "
                        + ", ".join(missing_reviews)
                    ),
                )
                return JudgeResponse(
                    approved=False,
                    required_improvements=[
                        f"Add reviewed_files entries for: {', '.join(missing_reviews)}"
                    ],
                    feedback=(
                        "Incomplete per-file coverage. Provide a reviewed_files entry for each changed file."
                    ),
                    current_task_metadata=task_metadata,
                    workflow_guidance=guidance,
                )

            # Track the file that was reviewed (if approved)
            if judge_result.approved:
                # Add all changed files to modified files and mark as approved
                for p in changed_files:
                    task_metadata.add_modified_file(p)
                    task_metadata.mark_code_approved(p)
                logger.info(f"Marked files as approved: {', '.join(changed_files)}")

                # Update state to TESTING when code is approved
                if task_metadata.state in [
                    TaskState.IMPLEMENTING,
                    TaskState.PLAN_APPROVED,
                ]:
                    task_metadata.update_state(TaskState.TESTING)

            # Calculate workflow guidance
            workflow_guidance = await calculate_next_stage(
                task_metadata=task_metadata,
                current_operation="judge_code_change_completed",
                conversation_service=conversation_service,
                ctx=ctx,
                validation_result=judge_result,
            )

            # Create enhanced response
            result = JudgeResponse(
                approved=judge_result.approved,
                required_improvements=judge_result.required_improvements,
                feedback=judge_result.feedback,
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

            # STEP 4: Save tool interaction to conversation history using the REAL task_id
            save_session_id = (
                task_metadata.task_id
                if getattr(task_metadata, "task_id", None)
                else (task_id or "test_task")
            )
            await save_tool_interaction(
                session_id=save_session_id,
                tool_name="judge_code_change",
                tool_input=original_input,
                tool_output=result,
            )

            return result

        except (ValidationError, ValueError) as e:
            raise ValueError(
                f"Failed to parse code change evaluation response: {e}. Raw response: {response_text}"
            ) from e

    except Exception as e:
        import traceback

        error_details = (
            f"Error during code review: {e!s}\nTraceback: {traceback.format_exc()}"
        )

        # Create error guidance
        error_guidance = build_error_guidance(
            reasoning="Error occurred during code change evaluation",
            guidance=f"Error during code review: {e!s}. Please review and try again.",
            preparation_needed=["Review error details", "Check task parameters"],
        )

        # Create minimal task metadata for error case
        error_metadata = (
            task_metadata
            if "task_metadata" in locals()
            else build_fallback_task_metadata(
                title="Error Task",
                description="Error occurred during code evaluation",
                user_requirements="",
                state=TaskState.IMPLEMENTING,
                tags=["error"],
            )
        )

        # For all errors, return enhanced error response
        error_result = JudgeResponse(
            approved=False,
            required_improvements=["Error occurred during review"],
            feedback=error_details,
            current_task_metadata=error_metadata,  # type: ignore[arg-type]
            workflow_guidance=error_guidance,
        )

        # Save error interaction
        await save_tool_interaction(
            session_id=task_id or "unknown",
            tool_name="judge_code_change",
            tool_input=original_input,
            tool_output=error_result,
            suppress_errors=True,
        )

        return error_result
