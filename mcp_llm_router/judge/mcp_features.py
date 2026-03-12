"""Additional MCP-native resources and prompts for the embedded judge."""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from mcp_llm_router.judge.models.task_metadata import TaskState
from mcp_llm_router.judge.runtime import conversation_service
from mcp_llm_router.judge.tools.common import serialize_tool_output


def _task_capability_status() -> dict[str, object]:
    enabled = os.getenv("MCP_JUDGE_ENABLE_TASKS", "").lower() in {"1", "true", "yes"}
    return {
        "requested": enabled,
        "active": False,
        "reason": (
            "Current FastMCP release in this repository does not expose server-side MCP Tasks registration APIs; stdio tool workflows remain authoritative."
        ),
    }


def register_judge_resources(mcp_server: FastMCP) -> None:
    @mcp_server.resource(
        "judge://current-task",
        name="current_judge_task",
        description="The most recently active judge task and its current metadata.",
        mime_type="application/json",
    )
    async def current_task_resource() -> str:
        recent = await conversation_service.db.get_recent_sessions(limit=1)
        if not recent:
            return json.dumps({"found": False, "message": "No coding task sessions found."})

        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_id, last_activity = recent[0]
        task_metadata = await load_task_metadata_from_history(
            task_id=task_id,
            conversation_service=conversation_service,
        )
        return json.dumps(
            {
                "found": True,
                "task_id": task_id,
                "last_activity": last_activity,
                "current_task_metadata": task_metadata.model_dump(mode="json")
                if task_metadata
                else None,
            }
        )

    @mcp_server.resource(
        "judge://task/{task_id}",
        name="judge_task",
        description="Metadata for a specific judge task.",
        mime_type="application/json",
    )
    async def task_resource(task_id: str) -> str:
        from mcp_llm_router.judge.tasks.manager import load_task_metadata_from_history

        task_metadata = await load_task_metadata_from_history(
            task_id=task_id,
            conversation_service=conversation_service,
        )
        return json.dumps(
            {
                "task_id": task_id,
                "found": task_metadata is not None,
                "current_task_metadata": task_metadata.model_dump(mode="json")
                if task_metadata
                else None,
            }
        )

    @mcp_server.resource(
        "judge://task/{task_id}/history",
        name="judge_task_history",
        description="Recent conversation and tool history for a judge task.",
        mime_type="application/json",
    )
    async def task_history_resource(task_id: str) -> str:
        records = await conversation_service.get_conversation_history(task_id)
        return json.dumps(
            conversation_service.format_conversation_history_as_json_array(records)
        )

    @mcp_server.resource(
        "judge://policy/rubric",
        name="judge_policy_rubric",
        description="The review rubric used by the embedded judge workflow.",
        mime_type="application/json",
    )
    async def policy_rubric_resource() -> str:
        return json.dumps(
            {
                "plan_review": [
                    "Validate the proposed design, sequencing, dependencies, and risk handling.",
                    "Require research evidence only when the workflow marks it as necessary.",
                ],
                "code_review": [
                    "Require unified diffs for implementation review.",
                    "Ensure every changed file receives explicit review coverage.",
                ],
                "testing_review": [
                    "Require raw runner output plus the list of test files.",
                    "Promote successful reviews to completion readiness, not directly to completion.",
                ],
                "completion_review": [
                    "Require prior plan, code, and testing approvals before marking a task complete.",
                ],
            }
        )

    @mcp_server.resource(
        "judge://workflow/states",
        name="judge_workflow_states",
        description="The judge workflow state machine and task capability status.",
        mime_type="application/json",
    )
    async def workflow_states_resource() -> str:
        return json.dumps(
            {
                "states": [
                    {"name": state.value, "label": state.name.lower()} for state in TaskState
                ],
                "tasks_capability": _task_capability_status(),
            }
        )


def register_judge_prompts(mcp_server: FastMCP) -> None:
    @mcp_server.prompt(
        name="start_judged_coding_task",
        description="Prepare an assistant to call set_coding_task with enough structure for the judge workflow.",
    )
    def start_judged_coding_task_prompt(
        task_title: str,
        task_description: str,
        user_requirements: str = "",
    ) -> str:
        return (
            "Start a judged coding task by calling set_coding_task with:\n"
            f"- task_title: {task_title}\n"
            f"- task_description: {task_description}\n"
            f"- user_requirements: {user_requirements or 'Summarize the user constraints explicitly.'}\n"
            "- Include task_size when the scope is obvious.\n"
            "- Preserve the returned task_id for every later judge tool call."
        )

    @mcp_server.prompt(
        name="submit_implementation_for_review",
        description="Prepare a unified diff submission for judge_code_change.",
    )
    def submit_implementation_for_review_prompt(task_id: str) -> str:
        return (
            f"Prepare a judge_code_change call for task_id {task_id}.\n"
            "- Generate a unified git diff containing every changed file.\n"
            "- Provide a concise change_description that explains the intent of the patch.\n"
            "- Do not submit narrative summaries in place of the diff."
        )

    @mcp_server.prompt(
        name="prepare_testing_evidence",
        description="Prepare the evidence bundle required by judge_testing_implementation.",
    )
    def prepare_testing_evidence_prompt(task_id: str) -> str:
        return (
            f"Prepare a judge_testing_implementation call for task_id {task_id}.\n"
            "- Include raw test runner output with pass/fail summary.\n"
            "- List every test file that was created or modified.\n"
            "- Include coverage or manual verification notes when available."
        )


def get_task_capability_status() -> str:
    return serialize_tool_output(_task_capability_status())
