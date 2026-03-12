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
    load_recent_history_as_json,
    load_task_metadata_with_history_debug,
    save_tool_interaction,
)
from mcp_llm_router.judge.workflow import calculate_next_stage
from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

async def judge_testing_implementation(
    task_id: str,  # REQUIRED: Task ID for context and validation
    test_summary: str,
    test_files: list[str],
    test_execution_results: str,
    ctx: Context,
    test_coverage_report: str = "",
    test_types_implemented: list[str] | None = None,
    testing_framework: str = "",
    performance_test_results: str = "",
    manual_test_notes: str = "",
) -> JudgeResponse:
    """Testing implementation validation tool - description loaded from tool_description_provider."""
    # Log tool execution start
    log_tool_execution("judge_testing_implementation", task_id)
    test_types_implemented = list(test_types_implemented or [])

    # Store original input for saving later
    original_input = {
        "task_id": task_id,
        "test_summary": test_summary,
        "test_files": test_files,
        "test_execution_results": test_execution_results,
        "test_coverage_report": test_coverage_report,
        "test_types_implemented": test_types_implemented,
        "testing_framework": testing_framework,
        "performance_test_results": performance_test_results,
        "manual_test_notes": manual_test_notes,
    }

    try:
        task_metadata = await load_task_metadata_with_history_debug(
            task_id=task_id,
            ctx=ctx,
            log_prefix="judge_testing_implementation",
        )

        if not task_metadata:
            task_metadata = build_fallback_task_metadata(
                title="Unknown Task",
                description="Task metadata could not be loaded from history",
                user_requirements="Task requirements not found",
                state=TaskState.TESTING,
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
                    guidance="You must call set_coding_task before calling judge_testing_implementation. The task_id must come from a successful set_coding_task call.",
                ),
            )

        # Early validation: require credible test evidence
        missing_evidence: list[str] = []
        if not test_files:
            missing_evidence.append("List the test files created/modified")

        # Use LLM-based validation for test output
        test_output_valid = await validate_test_output(
            test_execution_results or "",
            ctx,
            context="Validating test execution output for judge_testing_implementation",
        )
        if not test_output_valid:
            missing_evidence.append(
                "Provide raw test runner output including pass/fail summary"
            )

        if missing_evidence:
            # Minimal metadata if not loaded yet
            minimal_metadata = task_metadata or build_fallback_task_metadata(
                title="Testing Validation",
                description="Insufficient test evidence provided",
                user_requirements="",
                state=TaskState.TESTING,
            )
            guidance = WorkflowGuidance(
                next_tool="judge_testing_implementation",
                reasoning="Testing validation requires raw runner output and listed test files",
                preparation_needed=[
                    "Run the test suite (e.g., pytest -q, npm test, go test)",
                    "Copy/paste the raw test output with the summary",
                    "List test file paths",
                    "Include coverage summary if available",
                ],
                guidance=(
                    "Please rerun tests and provide the raw output (not a narrative). "
                    "Include pass/fail counts and list the test files modified."
                ),
            )
            return JudgeResponse(
                approved=False,
                required_improvements=missing_evidence,
                feedback="Insufficient evidence to validate testing results.",
                current_task_metadata=minimal_metadata,
                workflow_guidance=guidance,
            )

        root_violations = await find_paths_outside_roots(test_files, ctx)
        if root_violations:
            guidance = WorkflowGuidance(
                next_tool="judge_testing_implementation",
                reasoning="Submitted test file paths fall outside the client roots exposed to the judge.",
                preparation_needed=[
                    "List only test files that live under the allowed workspace roots",
                    "Remove any absolute paths or parent-directory traversals outside the workspace",
                ],
                guidance=(
                    "The following test file paths are outside the allowed roots: "
                    + ", ".join(sorted(set(root_violations)))
                ),
            )
            return JudgeResponse(
                approved=False,
                required_improvements=[
                    "Limit test evidence to files inside the allowed MCP roots"
                ],
                feedback=guidance.guidance,
                current_task_metadata=task_metadata,
                workflow_guidance=guidance,
            )

        # Track test files in task metadata
        for test_file in test_files:
            task_metadata.add_test_file(test_file)

        # Update test types status
        if test_types_implemented:
            for test_type in test_types_implemented:
                # Determine status based on execution results
                if (
                    "failed" in test_execution_results.lower()
                    or "error" in test_execution_results.lower()
                ):
                    status = "failing"
                elif (
                    "passed" in test_execution_results.lower()
                    or "success" in test_execution_results.lower()
                ):
                    status = "passing"
                else:
                    status = "unknown"
                task_metadata.update_test_status(test_type, status)

        test_coverage = task_metadata.get_test_coverage_summary()
        logger.debug(f"Test coverage summary: {test_coverage}")

        # COMPREHENSIVE TESTING EVALUATION using LLM
        user_requirements = task_metadata.user_requirements

        # Load conversation history for context
        history_json_array = await load_recent_history_as_json(task_id=task_id, ctx=ctx)

        # Prepare comprehensive test evaluation using LLM
        from mcp_llm_router.judge.models import (
            SystemVars,
            TestingEvaluationUserVars,
        )
        from mcp_llm_router.judge.prompting.loader import create_separate_messages

        # Create system and user variables for testing evaluation
        system_vars = SystemVars(
            response_schema=json.dumps(JudgeResponse.model_json_schema()),
            max_tokens=MAX_TOKENS,
        )
        user_vars = TestingEvaluationUserVars(
            user_requirements=user_requirements,
            task_description=task_metadata.description,
            modified_files=task_metadata.modified_files,
            test_summary=test_summary,
            test_files=test_files,
            test_execution_results=test_execution_results,
            test_coverage_report=test_coverage_report
            if test_coverage_report
            else "No coverage report provided",
            test_types_implemented=test_types_implemented
            if test_types_implemented
            else [],
            testing_framework=testing_framework
            if testing_framework
            else "Not specified",
            performance_test_results=performance_test_results
            if performance_test_results
            else "No performance tests",
            manual_test_notes=manual_test_notes
            if manual_test_notes
            else "No manual testing notes",
            conversation_history=history_json_array,
        )

        # Create messages for comprehensive testing evaluation
        messages = create_separate_messages(
            "system/judge_testing_implementation.md",
            "user/judge_testing_implementation.md",
            system_vars,
            user_vars,
        )

        # Use LLM for comprehensive testing evaluation
        response_text = await llm_provider.send_message(
            messages=messages,
            ctx=ctx,
            max_tokens=MAX_TOKENS,
            prefer_sampling=True,
        )

        # Parse the comprehensive evaluation response
        try:
            json_content = extract_json_from_response(response_text)
            testing_evaluation = JudgeResponse.model_validate_json(json_content)

            testing_approved = testing_evaluation.approved
            required_improvements = testing_evaluation.required_improvements
            evaluation_feedback = testing_evaluation.feedback

        except (ValidationError, ValueError) as e:
            # Fallback to basic evaluation if LLM fails
            logger.warning(
                f"LLM testing evaluation failed, using basic validation: {e}"
            )

            # Basic validation as fallback
            has_adequate_tests = len(test_files) > 0
            tests_passing = (
                "passed" in test_execution_results.lower()
                and "failed" not in test_execution_results.lower()
            )
            no_warnings = "warning" not in test_execution_results.lower()
            no_failures = (
                "failed" not in test_execution_results.lower()
                and "error" not in test_execution_results.lower()
            )
            has_coverage = test_coverage_report and test_coverage_report.strip() != ""

            testing_approved = (
                has_adequate_tests and tests_passing and no_warnings and no_failures
            )

            required_improvements = []
            if not has_adequate_tests:
                required_improvements.append("No test files provided")
            if not tests_passing:
                required_improvements.append("Tests are not passing")
            if not no_warnings:
                required_improvements.append(
                    "Test execution contains warnings that need to be addressed"
                )
            if not no_failures:
                required_improvements.append(
                    "Test execution contains failures or errors"
                )
            if not has_coverage and len(test_files) > 0:
                required_improvements.append(
                    "Test coverage report not provided - coverage analysis recommended"
                )

            evaluation_feedback = (
                "Basic validation performed due to LLM evaluation failure"
            )

        if testing_approved:
            # Mark testing as approved for completion validation
            task_metadata.mark_testing_approved()

            # Keep task state as TESTING - final completion will transition to COMPLETED
            # The workflow will guide to judge_coding_task_completion next

            # Use LLM evaluation feedback if available, otherwise create basic feedback
            if "evaluation_feedback" in locals():
                feedback = f"""✅ **TESTING IMPLEMENTATION APPROVED**

{evaluation_feedback}

**Test Summary:** {test_summary}

**Test Files ({len(test_files)}):**
{chr(10).join(f"- {file}" for file in test_files)}

**Test Execution:** {test_execution_results}

**Test Types:** {", ".join(test_types_implemented) if test_types_implemented else "Not specified"}

**Testing Framework:** {testing_framework if testing_framework else "Not specified"}

**Coverage:** {test_coverage_report if test_coverage_report else "Not provided"}

✅ **Ready for final task completion review.**"""
            else:
                feedback = f"""✅ **TESTING IMPLEMENTATION APPROVED**

**Test Summary:** {test_summary}

**Test Files ({len(test_files)}):**
{chr(10).join(f"- {file}" for file in test_files)}

**Test Execution:** {test_execution_results}

**Assessment:** The testing implementation meets the requirements. All tests are passing and provide adequate coverage for the implemented functionality.

✅ **Ready for final task completion review.**"""

        else:
            # Use LLM evaluation feedback if available, otherwise create basic feedback
            if "evaluation_feedback" in locals():
                feedback = f"""❌ **TESTING IMPLEMENTATION NEEDS IMPROVEMENT**

{evaluation_feedback}

**Test Summary:** {test_summary}

**Test Execution Results:** {test_execution_results}

📋 **Please address these testing issues before proceeding to task completion.**"""
            else:
                feedback = f"""❌ **TESTING IMPLEMENTATION NEEDS IMPROVEMENT**

**Test Summary:** {test_summary}

**Issues Found:**
{chr(10).join(f"- {issue}" for issue in required_improvements)}

**Test Execution Results:** {test_execution_results}

**Required Actions:**
- Write comprehensive tests for all implemented functionality
- Ensure all tests pass successfully
- Provide test coverage analysis
- Follow testing best practices for the framework

📋 **Please address these testing issues before proceeding to task completion.**"""

        # Calculate workflow guidance
        workflow_guidance = await calculate_next_stage(
            task_metadata=task_metadata,
            current_operation="judge_testing_implementation_completed",
            conversation_service=conversation_service,
            ctx=ctx,
        )

        # Create enhanced response
        result = JudgeResponse(
            approved=testing_approved,
            required_improvements=required_improvements,
            feedback=feedback,
            current_task_metadata=task_metadata,
            workflow_guidance=workflow_guidance,
        )

        # Save tool interaction to conversation history
        await save_tool_interaction(
            session_id=task_id,
            tool_name="judge_testing_implementation",
            tool_input=original_input,
            tool_output=result,
        )

        return result

    except Exception as e:
        import traceback

        error_details = f"Error during testing validation: {e!s}\nTraceback: {traceback.format_exc()}"

        # Create error guidance
        error_guidance = build_error_guidance(
            reasoning="Error occurred during testing validation",
            guidance=f"Error during testing validation: {e!s}. Please review and try again.",
            preparation_needed=["Review error details", "Check task parameters"],
        )

        # Create minimal task metadata for error case
        if "task_metadata" in locals() and task_metadata is not None:
            error_metadata = task_metadata
        else:
            error_metadata = build_fallback_task_metadata(
                title="Error Task",
                description="Error occurred during testing validation",
                user_requirements="Error occurred before task metadata could be loaded",
                state=TaskState.TESTING,
                tags=["error"],
            )

        # For all errors, return enhanced error response
        error_result = JudgeResponse(
            approved=False,
            required_improvements=["Error occurred during testing validation"],
            feedback=error_details,
            current_task_metadata=error_metadata,
            workflow_guidance=error_guidance,
        )

        # Save error interaction
        await save_tool_interaction(
            session_id=task_id if "task_id" in locals() else "unknown",
            tool_name="judge_testing_implementation",
            tool_input=original_input,
            tool_output=error_result,
            suppress_errors=True,
        )

        return error_result
