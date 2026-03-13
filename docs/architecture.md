# Architecture

## Request flow

1. `mcp_llm_router.server` boots the FastMCP router and registers the embedded judge.
2. `mcp_llm_router.judge.server` is the thin compatibility/bootstrap layer.
3. Runtime state lives in `mcp_llm_router.judge.runtime`.
4. Tool implementations live under `mcp_llm_router.judge.tools/`.
5. Persistence flows through `ConversationHistoryService`, which stores task history and tool outputs.

## Tool registration

- Router tools are declared directly in `mcp_llm_router.server`.
- Judge tools are registered by `register_judge_tools(...)`.
- MCP-native judge resources and prompts are added in `mcp_llm_router.judge.mcp_features`.

## Task state machine

The embedded judge follows this high-level path:

`created -> planning -> plan_pending_approval -> plan_approved -> implementing -> review_ready -> testing -> completed`

Cross-cutting escape hatches:

- `blocked`
- `cancelled`

Approval bookkeeping lives in `mcp_llm_router.judge.models.task_metadata.TaskMetadata`.

## Provider fallback model

- Router model selection flows through `BrainClient`.
- Judge model interactions flow through `llm_provider`.
- If no direct LLM API is configured, the judge prefers MCP sampling / elicitation when available.
- Local embeddings and reranking stay independent from judge/provider routing.

## Persistence model

- The router keeps lightweight session data in-memory.
- The embedded judge persists task metadata and conversation history to the configured judge database.
- MCP resources expose read-only snapshots of this persisted state so hosts/models do not need to resend the same context repeatedly.

## Roots-aware validation

- When a client exposes MCP roots, judge review/testing tools validate submitted paths against those roots.
- When roots are unavailable, the server keeps the existing stdio-first behavior and does not reject relative workspace paths.
