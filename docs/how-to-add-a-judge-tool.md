# How to add a new judge tool

1. Pick the closest module under `mcp_llm_router/judge/tools/` or create a new tool-family module if the behavior does not fit an existing one.
2. Reuse helpers from `mcp_llm_router.judge.tools.common` for:
   - task lookup/fallback metadata
   - workflow/error guidance
   - tool interaction persistence
3. Keep tool signatures MCP-friendly:
   - prefer explicit scalar/list parameters
   - use `None` for optional collections, then normalize inside the tool
4. Update `mcp_llm_router.judge.server._tool_definitions()` to register the tool and hook up the generated description.
5. If the tool exposes durable read-only state, consider adding a resource in `mcp_llm_router.judge.mcp_features`.
6. Add focused tests:
   - signature / schema drift if relevant
   - success path
   - error/fallback path
   - workflow transition or persistence behavior
7. Run:

```bash
python -m pytest -q
python scripts/inspector_smoke.py
```
