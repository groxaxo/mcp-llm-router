import asyncio
import os
import sys

# Ensure we can import the module from the repo root.
sys.path.append(os.getcwd())

EXPECTED_JUDGE_TOOLS = {
    "set_coding_task",
    "get_current_coding_task",
    "request_plan_approval",
    "raise_obstacle",
    "raise_missing_requirements",
    "judge_coding_task_completion",
    "judge_coding_plan",
    "judge_code_change",
    "judge_testing_implementation",
}


def _print_import_status() -> None:
    try:
        import mcp_llm_router.server  # noqa: F401

        print("Importing mcp_llm_router.server...")
        print("Import successful.")
    except ImportError as exc:
        print(f"FAILURE: ImportError: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"FAILURE: Exception during import: {exc}")
        sys.exit(1)


async def _check_tools() -> None:
    from mcp_llm_router.server import mcp
    from mcp_llm_router import judge_bridge

    print("Checking registered tools...")
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    print(f"Found {len(tool_names)} tools: {sorted(tool_names)}")

    missing = EXPECTED_JUDGE_TOOLS - tool_names
    if missing:
        print(f"FAILURE: Missing judge tools: {sorted(missing)}")
        if not judge_bridge.judge_available():
            print(
                "FAILURE: Judge module failed to load: "
                f"{judge_bridge.judge_import_error()}"
            )
        sys.exit(1)

    print("SUCCESS: Judge tools are registered.")


def main() -> None:
    _print_import_status()
    asyncio.run(_check_tools())


if __name__ == "__main__":
    main()
