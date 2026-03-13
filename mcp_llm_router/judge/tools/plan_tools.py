"""Judge planning and plan-approval tools."""

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

async def request_plan_approval(
    plan: str,
    design: str,
    research: str,
    task_id: str,
    ctx: Context,
    research_urls: list[str] | None = None,
    problem_domain: str = "",
    problem_non_goals: list[str] | None = None,
    library_plan: list[dict] | None = None,
    internal_reuse_components: list[dict] | None = None,
) -> PlanApprovalResult:
    """Present the plan to the user for approval before proceeding to judge_coding_plan."""
    # Log tool execution start
    log_tool_execution("request_plan_approval", task_id)
    research_urls = list(research_urls or [])
    problem_non_goals = list(problem_non_goals or [])
    library_plan = list(library_plan or [])
    internal_reuse_components = list(internal_reuse_components or [])

    try:
        # Load task metadata
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_metadata = await load_task_metadata_from_history(
            task_id, conversation_service
        )

        if not task_metadata:
            # Create a minimal task metadata for error response
            from mcp_llm_router.judge.models.task_metadata import TaskSize
            from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

            error_task_metadata = TaskMetadata(
                title="Error Task", description="Task not found", task_size=TaskSize.M
            )
            error_guidance = WorkflowGuidance(
                next_tool="set_coding_task",
                reasoning="Task not found, need to create a new task",
                preparation_needed=["Create a new task"],
                guidance="Call set_coding_task to create a new task",
            )
            return PlanApprovalResult(
                approved=False,
                user_feedback="Task not found. Please call set_coding_task first.",
                next_action="Call set_coding_task to create a new task",
                current_task_metadata=error_task_metadata,
                workflow_guidance=error_guidance,
            )

        # Update task state to PLAN_PENDING_APPROVAL
        task_metadata.update_state(TaskState.PLAN_PENDING_APPROVAL)

        # Format plan for user presentation
        plan_presentation = f"""
# Implementation Plan for: {task_metadata.title}

## Overview
{task_metadata.description}

## Implementation Plan
{plan}

## Technical Design
{design}

## Research Summary
{research}
"""

        if research_urls:
            plan_presentation += "\n## Research Sources\n"
            for url in research_urls:
                plan_presentation += f"- {url}\n"

        if problem_domain:
            plan_presentation += f"\n## Problem Domain\n{problem_domain}\n"

        if problem_non_goals:
            plan_presentation += "\n## Non-Goals\n"
            for goal in problem_non_goals:
                plan_presentation += f"- {goal}\n"

        if library_plan:
            plan_presentation += "\n## Library Plan\n"
            for lib in library_plan:
                plan_presentation += f"- **{lib.get('purpose', 'Unknown')}**: {lib.get('selection', 'Unknown')} ({lib.get('source', 'Unknown')})\n"

        if internal_reuse_components:
            plan_presentation += "\n## Internal Components to Reuse\n"
            for comp in internal_reuse_components:
                plan_presentation += f"- **{comp.get('path', 'Unknown')}**: {comp.get('purpose', 'Unknown')}\n"

        plan_presentation += """

## Your Options
Please review the plan above and choose one of the following:

1. **Approve** - Proceed with this plan as-is
2. **Modify** - Request changes to the plan (please provide specific feedback)
3. **Reject** - Start over with a different approach
"""

        # Use elicitation to get user approval
        elicitation_result = await elicitation_provider.elicit_user_input(
            message=plan_presentation, schema=PlanApprovalResponse, ctx=ctx
        )

        if not elicitation_result.success:
            error_guidance = WorkflowGuidance(
                next_tool="request_plan_approval",
                reasoning="Failed to get user input for plan approval",
                preparation_needed=["Check elicitation system", "Retry plan approval"],
                guidance="Retry plan approval or proceed without user input",
            )
            return PlanApprovalResult(
                approved=False,
                user_feedback="Failed to get user input: " + elicitation_result.message,
                next_action="Retry plan approval or proceed without user input",
                current_task_metadata=task_metadata,
                workflow_guidance=error_guidance,
            )

        # Process user response
        user_response = elicitation_result.data
        action = user_response.get("action", "").lower()
        feedback = user_response.get("feedback", "")

        if action == "approve":
            # User approved - keep state as PLAN_PENDING_APPROVAL until AI judge validates
            # Do NOT set to PLAN_APPROVED yet - that happens only after judge_coding_plan approval

            # Save the user-approved plan data to task metadata
            history_input = json.dumps(
                {
                    "plan": plan,
                    "design": design,
                    "research": research,
                    "research_urls": research_urls,
                    "problem_domain": problem_domain,
                    "problem_non_goals": problem_non_goals,
                    "library_plan": library_plan,
                    "internal_reuse_components": internal_reuse_components,
                    "user_action": action,
                    "user_feedback": feedback,
                }
            )

            await save_task_metadata_to_history(
                task_metadata=task_metadata,
                user_request=history_input,
                action="plan_user_approved",  # Changed to indicate user approval, not final approval
                conversation_service=conversation_service,
            )

            # Generate workflow guidance for next step
            from mcp_llm_router.judge.workflow.workflow_guidance import WorkflowGuidance

            workflow_guidance = WorkflowGuidance(
                next_tool="judge_coding_plan",
                reasoning="Plan approved by user; proceed to AI validation before implementation",
                preparation_needed=[
                    "Ensure all plan components are complete (plan, design, research)",
                    "Include library_plan and internal_reuse_components if applicable",
                    "Add identified_risks and risk_mitigation_strategies if required",
                ],
                guidance="Call judge_coding_plan with the complete plan details for AI validation. After approval, proceed to implementation.",
            )

            return PlanApprovalResult(
                approved=True,
                user_feedback=feedback or "Plan approved by user",
                next_action="Proceed to judge_coding_plan for validation",
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

        elif action == "modify":
            # User wants modifications - return to PLANNING state
            task_metadata.update_state(TaskState.PLANNING)

            # Update requirements with user feedback
            if feedback:
                task_metadata.update_requirements(
                    f"{task_metadata.user_requirements}\n\nUser feedback on plan: {feedback}",
                    source="plan_approval_feedback",
                )

            history_input = json.dumps(
                {
                    "plan": plan,
                    "design": design,
                    "research": research,
                    "research_urls": research_urls,
                    "problem_domain": problem_domain,
                    "problem_non_goals": problem_non_goals,
                    "library_plan": library_plan,
                    "internal_reuse_components": internal_reuse_components,
                    "user_action": action,
                    "user_feedback": feedback,
                }
            )

            await save_task_metadata_to_history(
                task_metadata=task_metadata,
                user_request=history_input,
                action="plan_modification_requested",
                conversation_service=conversation_service,
            )

            # Generate workflow guidance for plan revision
            workflow_guidance = WorkflowGuidance(
                next_tool=None,  # No specific tool, let AI create revised plan
                reasoning="User requested plan modifications; revise plan based on feedback",
                preparation_needed=[
                    "Review user feedback carefully",
                    "Revise plan to address specific concerns",
                    "Ensure all plan components remain complete",
                ],
                guidance=f"User feedback: {feedback}. Revise the implementation plan to address these concerns, then call request_plan_approval again with the updated plan.",
            )

            return PlanApprovalResult(
                approved=False,
                user_feedback=feedback or "User requested plan modifications",
                next_action="Revise plan based on user feedback and resubmit for approval",
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

        else:  # reject or any other action
            # User rejected - return to PLANNING state
            task_metadata.update_state(TaskState.PLANNING)
            history_input = json.dumps(
                {
                    "plan": plan,
                    "design": design,
                    "research": research,
                    "research_urls": research_urls,
                    "problem_domain": problem_domain,
                    "problem_non_goals": problem_non_goals,
                    "library_plan": library_plan,
                    "internal_reuse_components": internal_reuse_components,
                    "user_action": action,
                    "user_feedback": feedback,
                }
            )

            await save_task_metadata_to_history(
                task_metadata=task_metadata,
                user_request=history_input,
                action="plan_rejected",
                conversation_service=conversation_service,
            )

            # Generate workflow guidance for new plan creation
            workflow_guidance = WorkflowGuidance(
                next_tool=None,  # No specific tool, let AI create new plan
                reasoning="User rejected the plan; create a completely new approach",
                preparation_needed=[
                    "Review user feedback for rejection reasons",
                    "Consider alternative approaches and architectures",
                    "Create a fundamentally different plan",
                ],
                guidance=f"User rejected the plan. Feedback: {feedback}. Create a completely new implementation plan with a different approach, then call request_plan_approval with the new plan.",
            )

            return PlanApprovalResult(
                approved=False,
                user_feedback=feedback or "Plan rejected by user",
                next_action="Create a new plan with a different approach",
                current_task_metadata=task_metadata,
                workflow_guidance=workflow_guidance,
            )

    except Exception as e:
        logger.error(f"Error in request_plan_approval: {e!s}")

        # Create error workflow guidance
        error_guidance = WorkflowGuidance(
            next_tool=None,
            reasoning="Error occurred during plan approval process",
            preparation_needed=[
                "Review error details",
                "Check task metadata",
                "Retry or proceed manually",
            ],
            guidance=f"Error in plan approval: {e!s}. Review the error and retry the plan approval process or proceed without user input if necessary.",
        )

        # Try to get task metadata for error response
        try:
            from mcp_llm_router.judge.models.task_metadata import TaskSize
            from mcp_llm_router.judge.tasks.manager import (
                load_task_metadata_from_history,
            )

            error_task_metadata_maybe = await load_task_metadata_from_history(
                task_id, conversation_service
            )
            if not error_task_metadata_maybe:
                error_task_metadata = TaskMetadata(
                    title="Error Task",
                    description="Error occurred during plan approval",
                    task_size=TaskSize.M,
                )
            else:
                error_task_metadata = error_task_metadata_maybe
        except Exception:
            from mcp_llm_router.judge.models.task_metadata import TaskSize

            error_task_metadata = TaskMetadata(
                title="Error Task",
                description="Error occurred during plan approval",
                task_size=TaskSize.M,
            )

        return PlanApprovalResult(
            approved=False,
            user_feedback=f"Error occurred: {e!s}",
            next_action="Retry plan approval or proceed without user input",
            current_task_metadata=error_task_metadata,
            workflow_guidance=error_guidance,
        )


async def judge_coding_plan(
    plan: str,
    design: str,
    research: str,
    research_urls: list[str],
    ctx: Context,
    task_id: str = "",
    context: str = "",
    # OPTIONAL override
    user_requirements: str = "",
    # OPTIONAL explicit inputs to avoid rejection on missing deliverables
    problem_domain: str = "",
    problem_non_goals: list[str] | None = None,
    library_plan: list[dict] | None = None,
    internal_reuse_components: list[dict] | None = None,
    design_patterns: list[dict] | None = None,
    identified_risks: list[str] | None = None,
    risk_mitigation_strategies: list[str] | None = None,
) -> JudgeResponse:
    """Coding plan evaluation tool - description loaded from tool_description_provider."""
    # Log tool execution start
    log_tool_execution("judge_coding_plan", task_id if task_id else "test_task")
    problem_non_goals = list(problem_non_goals or [])
    library_plan = list(library_plan or [])
    internal_reuse_components = list(internal_reuse_components or [])
    design_patterns = list(design_patterns or [])
    identified_risks = list(identified_risks or [])
    risk_mitigation_strategies = list(risk_mitigation_strategies or [])

    # Store original input for saving later
    original_input = {
        "task_id": task_id if task_id else "test_task",
        "plan": plan,
        "design": design,
        "research": research,
        "context": context,
        "research_urls": research_urls,
        "problem_domain": problem_domain,
        "problem_non_goals": problem_non_goals,
        "library_plan": library_plan,
        "internal_reuse_components": internal_reuse_components,
        "design_patterns": design_patterns,
    }

    try:
        # If neither MCP sampling nor LLM API are available, short-circuit with a clear error
        sampling_available = llm_provider.is_sampling_available(ctx)
        llm_available = llm_provider.is_llm_api_available()
        if not (sampling_available or llm_available):
            minimal_metadata = TaskMetadata(
                title="Unknown Task",
                description="Task metadata could not be loaded from history",
                user_requirements=user_requirements if user_requirements else "",
                state=TaskState.CREATED,
                task_size=TaskSize.M,
                tags=["debug", "missing-metadata"],
            )
            return EnhancedResponseFactory.create_judge_response(
                approved=False,
                feedback=(
                    "Error during coding plan evaluation: No messaging providers available"
                ),
                required_improvements=["Error occurred during review"],
                current_task_metadata=minimal_metadata,
                workflow_guidance=WorkflowGuidance(
                    next_tool=None,
                    reasoning="No messaging providers available",
                    preparation_needed=["Configure MCP sampling or LLM API"],
                    guidance="Set up a provider and retry the evaluation.",
                ),
            )
        # Load task metadata to get current context and user requirements
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        logger.info(
            f"judge_coding_plan: Loading task metadata for task_id: {task_id if task_id else 'test_task'}"
        )

        task_metadata = await load_task_metadata_from_history(
            task_id=task_id if task_id else "test_task",
            conversation_service=conversation_service,
        )

        logger.info(
            f"judge_coding_plan: Task metadata loaded: {task_metadata is not None}"
        )
        if task_metadata:
            logger.info(
                f"judge_coding_plan: Task state: {task_metadata.state}, title: {task_metadata.title}"
            )
        else:
            conversation_history = (
                await conversation_service.load_filtered_context_for_enrichment(
                    task_id if task_id else "test_task", "", ctx
                )
            )
            logger.info(
                f"judge_coding_plan: Conversation history entries: {len(conversation_history)}"
            )
            for entry in conversation_history[-5:]:
                logger.info(
                    f"judge_coding_plan: History entry: {entry.source} at {entry.timestamp}"
                )

        if not task_metadata:
            # Create a minimal task metadata fallback but continue evaluation
            task_metadata = TaskMetadata(
                title="Unknown Task",
                description="Task metadata could not be loaded from history",
                user_requirements="Task requirements not found",
                state=TaskState.CREATED,
                task_size=TaskSize.M,
                tags=["debug", "missing-metadata"],
            )

        # Transition to PLANNING state when planning starts
        if task_metadata.state == TaskState.CREATED:
            task_metadata.update_state(TaskState.PLANNING)

        # Derive user requirements from task metadata (allow override)
        user_requirements = (
            user_requirements
            if user_requirements is not None
            else task_metadata.user_requirements
        )

        effective_identified_risks = list(
            identified_risks or task_metadata.identified_risks or []
        )
        effective_risk_mitigations = list(
            risk_mitigation_strategies or task_metadata.risk_mitigation_strategies or []
        )

        # Clean up risk assessment data if required
        if task_metadata.risk_assessment_required:
            cleaned_risks = [
                risk.strip()
                for risk in effective_identified_risks
                if isinstance(risk, str) and risk.strip()
            ]
            cleaned_mitigations = [
                mitigation.strip()
                for mitigation in effective_risk_mitigations
                if isinstance(mitigation, str) and mitigation.strip()
            ]

            # Ensure 1:1 mapping between risks and mitigations
            if len(cleaned_mitigations) < len(cleaned_risks):
                for _ in range(len(cleaned_mitigations), len(cleaned_risks)):
                    cleaned_mitigations.append(
                        "Document concrete mitigation strategy for this risk"
                    )
            elif len(cleaned_mitigations) > len(cleaned_risks):
                cleaned_mitigations = cleaned_mitigations[: len(cleaned_risks)]

            effective_identified_risks = cleaned_risks
            effective_risk_mitigations = cleaned_mitigations

        if (
            task_metadata.risk_assessment_required
            and effective_identified_risks
            and not task_metadata.identified_risks
        ):
            task_metadata.identified_risks = list(effective_identified_risks)
        if (
            task_metadata.risk_assessment_required
            and effective_risk_mitigations
            and not task_metadata.risk_mitigation_strategies
        ):
            task_metadata.risk_mitigation_strategies = list(effective_risk_mitigations)

        original_input["identified_risks"] = effective_identified_risks
        original_input["risk_mitigation_strategies"] = effective_risk_mitigations

        # NOTE: Conditional research, internal analysis, and risk assessment requirements
        # are now determined dynamically by the LLM through the workflow guidance system
        # rather than using hardcoded rule-based analysis

        research_required = bool(task_metadata.research_required)
        auto_approved_due_to_limit = False

        # DYNAMIC RESEARCH VALIDATION - Only validate if research is actually required
        if research_required and not task_metadata.has_exceeded_plan_rejection_limit():
            # Import dynamic research analysis functions
            from mcp_llm_router.judge.tasks.research import (
                analyze_research_requirements,
                update_task_metadata_with_analysis,
                validate_url_adequacy,
            )

            # Step 1: Perform research requirements analysis if not already done
            if task_metadata.expected_url_count is None:
                logger.info(
                    f"Performing dynamic research requirements analysis for task {task_id or 'test_task'}"
                )
                try:
                    requirements_analysis = await analyze_research_requirements(
                        task_metadata=task_metadata,
                        user_requirements=user_requirements,
                        ctx=ctx,
                    )
                    # Update task metadata with analysis results
                    update_task_metadata_with_analysis(
                        task_metadata, requirements_analysis
                    )
                    logger.info(
                        f"Research analysis complete: Expected={task_metadata.expected_url_count}, Minimum={task_metadata.minimum_url_count}"
                    )

                    # Save the analysis results to task history
                    await save_task_metadata_to_history(
                        task_metadata=task_metadata,
                        user_request=user_requirements,
                        action="research_requirements_analyzed",
                        conversation_service=conversation_service,
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Research analysis failed: {e}. Using fallback validation."
                    )
                    # Fall back to basic empty check if analysis fails
                    if not research_urls or len(research_urls) == 0:
                        validation_issue = f"Research is required (scope: {task_metadata.research_scope}). No research URLs provided. Rationale: {task_metadata.research_rationale}"
                        context_info = f"User requirements: {user_requirements}. Plan: {plan[:200]}..."

                        descriptive_feedback = await generate_validation_error_message(
                            validation_issue, context_info, ctx
                        )

                        # Increment rejection count for insufficient research
                        task_metadata.increment_plan_rejection()
                        logger.info(
                            f"Plan rejected due to insufficient research. Rejection count: {task_metadata.plan_rejection_count}/1"
                        )

                        workflow_guidance = await calculate_next_stage(
                            task_metadata=task_metadata,
                            current_operation="judge_coding_plan_insufficient_research",
                            conversation_service=conversation_service,
                            ctx=ctx,
                        )

                        return JudgeResponse(
                            approved=False,
                            required_improvements=[
                                "Research required but no URLs provided",
                            ],
                            feedback=descriptive_feedback,
                            current_task_metadata=task_metadata,
                            workflow_guidance=workflow_guidance,
                        )

            # Step 2: Validate provided URLs against dynamic requirements
            if task_metadata.expected_url_count is not None:
                url_validation = await validate_url_adequacy(
                    provided_urls=research_urls,
                    expected_count=task_metadata.expected_url_count,
                    minimum_count=task_metadata.minimum_url_count or 1,
                    reasoning=task_metadata.url_requirement_reasoning,
                    ctx=ctx,
                )

                if not url_validation.adequate:
                    logger.warning(
                        f"⚠️ URL validation failed for task {task_id or 'test_task'}: {url_validation.feedback}"
                    )

                    descriptive_feedback = await generate_validation_error_message(
                        url_validation.feedback,
                        f"User requirements: {user_requirements}. Research scope: {task_metadata.research_scope}",
                        ctx,
                    )

                    # Increment rejection count for URL validation failure
                    task_metadata.increment_plan_rejection()
                    logger.info(
                        f"Plan rejected due to insufficient URL count. Rejection count: {task_metadata.plan_rejection_count}/1"
                    )

                    workflow_guidance = await calculate_next_stage(
                        task_metadata=task_metadata,
                        current_operation="judge_coding_plan_insufficient_research",
                        conversation_service=conversation_service,
                        ctx=ctx,
                    )

                    return JudgeResponse(
                        approved=False,
                        required_improvements=[
                            f"Provide at least {url_validation.minimum_count} research URLs",
                        ],
                        feedback=descriptive_feedback,
                        current_task_metadata=task_metadata,
                        workflow_guidance=workflow_guidance,
                    )
                else:
                    logger.info(
                        f"✅ URL validation passed for task {task_id}: {url_validation.provided_count} URLs meet requirements"
                    )

            # Research URLs provided - mark completion and let LLM prompts handle quality validation
            task_metadata.research_completed = int(time.time())
            task_metadata.updated_at = int(time.time())

            # Save updated task metadata
            await save_task_metadata_to_history(
                task_metadata=task_metadata,
                user_request=user_requirements,
                action="research_completed",
                conversation_service=conversation_service,
            )

        elif research_required:
            logger.info(
                "Skipping research validation because plan rejection limit was reached; "
                "auto-approval safeguard will handle the workflow progression."
            )
        else:
            # Research is optional - log but don't block
            logger.info(
                f"Research optional for task {task_id} (research_required={task_metadata.research_required})"
            )
            if research_urls:
                logger.info(f"Optional research provided: {len(research_urls)} URLs")

        # HITL is guided by workflow prompts and elicitation tools, not rule-based gating here

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

        # STEP 4: Use helper function for main evaluation with JSON array conversation history
        # Provide contextual note to avoid blocking on non-existent internal components
        eval_context = ""
        try:
            if (
                task_metadata.internal_research_required is True
                and not task_metadata.related_code_snippets
            ):
                eval_context = (
                    "No repository-local related components are currently identified in task metadata. "
                    "If none can be found in this repository, do not block on internal codebase analysis; "
                    "set internal_research_required=false in current_task_metadata and proceed with clear rationale."
                )
        except Exception:
            # Be resilient; context is optional
            eval_context = ""

        # Check rejection limit - auto-approve if already rejected once
        if task_metadata.has_exceeded_plan_rejection_limit():
            auto_approved_due_to_limit = True
            logger.info(
                f"Plan has already been rejected {task_metadata.plan_rejection_count} time(s). "
                f"Auto-approving to prevent endless iteration cycles."
            )

            # Create auto-approval result
            evaluation_result = EnhancedResponseFactory.create_judge_response(
                approved=True,
                feedback="Plan auto-approved after reaching rejection limit (max 1 rejection allowed). "
                "Moving forward to prevent endless iteration cycles.",
                required_improvements=[],
                current_task_metadata=task_metadata,
                workflow_guidance=WorkflowGuidance(
                    next_tool=None,
                    reasoning="Auto-approved due to rejection limit",
                    preparation_needed=[],
                    guidance="Proceed with implementation",
                ),
            )
        else:
            # Perform normal evaluation
            evaluation_result = await evaluate_coding_plan(
                plan,
                design,
                research,
                research_urls,
                user_requirements,
                eval_context,
                history_json_array,
                task_metadata,  # Pass task metadata for conditional features
                ctx,
                problem_domain=problem_domain,
                problem_non_goals=problem_non_goals,
                library_plan=library_plan,
                internal_reuse_components=internal_reuse_components,
                design_patterns=design_patterns,
                identified_risks_override=effective_identified_risks,
                risk_mitigation_override=effective_risk_mitigations,
            )

        # Additional research validation if approved
        if evaluation_result.approved and not auto_approved_due_to_limit:
            research_validation_result = await validate_research_quality(
                research, research_urls, plan, design, user_requirements, ctx
            )
            if research_validation_result:
                # Increment rejection count for research validation failure
                task_metadata.increment_plan_rejection()
                logger.info(
                    f"Plan rejected due to research validation failure. Rejection count: {task_metadata.plan_rejection_count}/1"
                )

                workflow_guidance = await calculate_next_stage(
                    task_metadata=task_metadata,
                    current_operation="judge_coding_plan_research_failed",
                    conversation_service=conversation_service,
                    ctx=ctx,
                )

                return JudgeResponse(
                    approved=False,
                    required_improvements=research_validation_result.get(
                        "required_improvements", []
                    ),
                    feedback=research_validation_result.get(
                        "feedback", "Research validation failed"
                    ),
                    current_task_metadata=task_metadata,
                    workflow_guidance=workflow_guidance,
                )

        # Use the updated task metadata from the evaluation result (includes conditional requirements)
        updated_task_metadata = evaluation_result.current_task_metadata

        # Enforce mandatory planning deliverables: problem_domain and library_plan
        # If missing but the plan was approved, convert to required improvements
        missing_deliverables: list[str] = []
        try:
            # Fill from explicit inputs if LLM omitted them in metadata
            if (
                problem_domain
                and not getattr(updated_task_metadata, "problem_domain", "").strip()
            ):
                updated_task_metadata.problem_domain = problem_domain
            if problem_non_goals and not getattr(
                updated_task_metadata, "problem_non_goals", None
            ):
                updated_task_metadata.problem_non_goals = problem_non_goals
            if library_plan and (
                not getattr(updated_task_metadata, "library_plan", None)
                or len(getattr(updated_task_metadata, "library_plan", [])) == 0
            ):
                # Convert dict list to LibraryPlanItem list
                library_plan_items = [
                    TaskMetadata.LibraryPlanItem(**item) for item in library_plan
                ]
                updated_task_metadata.library_plan = library_plan_items
            if internal_reuse_components and (
                not getattr(updated_task_metadata, "internal_reuse_components", None)
                or len(getattr(updated_task_metadata, "internal_reuse_components", []))
                == 0
            ):
                # Convert dict list to ReuseComponent list
                reuse_components = [
                    TaskMetadata.ReuseComponent(**item)
                    for item in internal_reuse_components
                ]
                updated_task_metadata.internal_reuse_components = reuse_components
            if effective_identified_risks and not getattr(
                updated_task_metadata, "identified_risks", []
            ):
                updated_task_metadata.identified_risks = effective_identified_risks
            if effective_risk_mitigations and not getattr(
                updated_task_metadata, "risk_mitigation_strategies", []
            ):
                updated_task_metadata.risk_mitigation_strategies = (
                    effective_risk_mitigations
                )

            # Now check for missing deliverables
            if not getattr(updated_task_metadata, "problem_domain", "").strip():
                missing_deliverables.append(
                    "Add a clear Problem Domain Statement with explicit non-goals"
                )
            if (
                not getattr(updated_task_metadata, "library_plan", [])
                or len(getattr(updated_task_metadata, "library_plan", [])) == 0
            ):
                missing_deliverables.append(
                    "Provide a Library Selection Map (purpose → internal/external library with justification)"
                )
        except Exception:  # nosec B110
            pass

        if auto_approved_due_to_limit:
            # Preserve auto-approval even if optional deliverables are missing
            effective_approved = True
        else:
            effective_approved = evaluation_result.approved and not missing_deliverables
        effective_required_improvements = list(evaluation_result.required_improvements)
        if missing_deliverables:
            # Merge missing deliverables to required improvements
            effective_required_improvements.extend(missing_deliverables)

        # Preserve canonical task_id so we never drift across sessions due to LLM outputs
        canonical_task_id = None
        if task_metadata and getattr(task_metadata, "task_id", None):
            canonical_task_id = task_metadata.task_id
        elif task_id:
            canonical_task_id = task_id

        if (
            canonical_task_id
            and getattr(updated_task_metadata, "task_id", None) != canonical_task_id
        ):
            with contextlib.suppress(Exception):
                # Overwrite to ensure consistency across conversation history and routing
                updated_task_metadata.task_id = canonical_task_id

        # Update task metadata state BEFORE calculating workflow guidance to ensure consistency
        if effective_approved:
            # Mark plan as approved for completion validation and update state
            updated_task_metadata.mark_plan_approved()
            updated_task_metadata.update_state(TaskState.PLAN_APPROVED)

            # Delete previous failed plan attempts, keeping only the most recent approved one
            await conversation_service.db.delete_previous_plan(
                updated_task_metadata.task_id
            )
        else:
            # Increment rejection count for tracking
            updated_task_metadata.increment_plan_rejection()
            logger.info(
                f"Plan rejected. Rejection count: {updated_task_metadata.plan_rejection_count}/1"
            )

            # Keep/return to planning state and request plan improvements
            updated_task_metadata.update_state(TaskState.PLANNING)

        # Calculate workflow guidance with correct task state
        # Build a synthetic validation_result with the effective approval and improvements
        synthetic_eval = EnhancedResponseFactory.create_judge_response(
            approved=effective_approved,
            feedback=evaluation_result.feedback,
            required_improvements=effective_required_improvements,
            current_task_metadata=updated_task_metadata,
            workflow_guidance=WorkflowGuidance(
                next_tool=None,
                reasoning="",
                preparation_needed=[],
                guidance="",
            ),
        )
        workflow_guidance = await calculate_next_stage(
            task_metadata=updated_task_metadata,
            current_operation="judge_coding_plan_completed",
            conversation_service=conversation_service,
            ctx=ctx,
            validation_result=synthetic_eval,
        )

        # Apply deterministic overrides for plan outcome to ensure correct routing
        if effective_approved:
            # Force next step to code review implementation gate
            workflow_guidance.next_tool = "judge_code_change"
            if not workflow_guidance.reasoning:
                workflow_guidance.reasoning = (
                    "Plan approved; proceed with implementation and code review."
                )
            if not workflow_guidance.preparation_needed:
                workflow_guidance.preparation_needed = [
                    "Implement according to the approved plan",
                    "Prepare file paths and change summary for review",
                ]
            if not workflow_guidance.guidance:
                workflow_guidance.guidance = "Start implementation. When a cohesive set of changes is ready, call judge_code_change with file paths and a concise summary or diff."
        else:
            # Force next step to plan revision
            workflow_guidance.next_tool = "judge_coding_plan"
            if not workflow_guidance.reasoning:
                workflow_guidance.reasoning = (
                    "Plan not approved; address feedback and resubmit."
                )
            if not workflow_guidance.preparation_needed:
                workflow_guidance.preparation_needed = [
                    "Revise plan per required improvements",
                    "Ensure design, file list, and research coverage meet requirements",
                ]
            if not workflow_guidance.guidance:
                workflow_guidance.guidance = "Update the plan addressing all required improvements and resubmit to judge_coding_plan."

        result = JudgeResponse(
            approved=effective_approved,
            required_improvements=effective_required_improvements,
            feedback=evaluation_result.feedback,
            current_task_metadata=updated_task_metadata,
            workflow_guidance=workflow_guidance,
        )

        # STEP 3: Save tool interaction to conversation history using the REAL task_id
        save_session_id = (
            (task_metadata.task_id if task_metadata else None)
            or task_id
            or getattr(updated_task_metadata, "task_id", None)
            or "test_task"
        )
        await conversation_service.save_tool_interaction_and_cleanup(
            session_id=save_session_id,  # Always prefer real task_id
            tool_name="judge_coding_plan",
            tool_input=json.dumps(original_input),
            tool_output=json.dumps(
                result.model_dump(
                    mode="json",
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                )
            ),
        )

        return result

    except Exception as e:
        import traceback

        error_details = (
            f"Error during plan review: {e!s}\nTraceback: {traceback.format_exc()}"
        )
        logger.error(error_details)

        # Create error guidance
        error_guidance = WorkflowGuidance(
            next_tool="get_current_coding_task",
            reasoning="Error occurred during coding plan evaluation; recover active task context and retry.",
            preparation_needed=[
                "Call get_current_coding_task to fetch active task_id",
                "Retry evaluation with the returned task_id",
            ],
            guidance=f"Error during plan review: {e!s}. Use get_current_coding_task to recover the current task_id and retry.",
        )

        # Create minimal task metadata for error case
        if "task_metadata" in locals() and task_metadata is not None:
            error_metadata = task_metadata
            # Increment rejection count for error cases too
            error_metadata.increment_plan_rejection()
            logger.info(
                f"Plan rejected due to error. Rejection count: {error_metadata.plan_rejection_count}/1"
            )
        else:
            error_metadata = TaskMetadata(
                title="Error Task",
                description="Error occurred during plan evaluation",
                user_requirements="Error occurred before task metadata could be loaded",
                state=TaskState.PLANNING,
                task_size=TaskSize.M,
                tags=["error"],
            )

        # For all errors, return enhanced error response
        error_result = JudgeResponse(
            approved=False,
            required_improvements=["Error occurred during review"],
            feedback=f"Error during coding plan evaluation: {e!s}",
            current_task_metadata=error_metadata,
            workflow_guidance=error_guidance,
        )

        # Save error interaction
        with contextlib.suppress(builtins.BaseException):
            await conversation_service.save_tool_interaction_and_cleanup(
                session_id=(task_id or "unknown")
                if "task_id" in locals()
                else "unknown",
                tool_name="judge_coding_plan",
                tool_input=json.dumps(original_input),
                tool_output=json.dumps(
                    error_result.model_dump(
                        mode="json",
                        exclude_unset=True,
                        exclude_none=True,
                        exclude_defaults=True,
                    )
                ),
            )

        return error_result
