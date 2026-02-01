#!/usr/bin/env python3
"""Auto-install project dependencies from pyproject.toml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_dependencies_fallback(text: str) -> list[str]:
    deps: list[str] = []
    in_block = False
    buffer = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("dependencies = ["):
                in_block = True
                buffer = stripped[len("dependencies = [") :].strip()
                if buffer.endswith("]"):
                    buffer = buffer[: -1].strip()
                    in_block = False
                    if buffer:
                        deps.extend(_parse_inline_list(buffer))
            continue

        if stripped.startswith("]"):
            in_block = False
            continue

        if stripped.startswith("#") or not stripped:
            continue

        buffer = stripped
        deps.extend(_parse_inline_list(buffer))

    return [d for d in deps if d]


def _parse_inline_list(text: str) -> list[str]:
    # Remove inline comments and trailing commas
    raw = text.split("#", 1)[0].strip().rstrip(",")
    if not raw:
        return []

    # Handle comma-separated items on the same line
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    deps: list[str] = []
    for part in parts:
        if (part.startswith('"') and part.endswith('"')) or (
            part.startswith("'") and part.endswith("'")
        ):
            deps.append(part[1:-1])
        else:
            deps.append(part)
    return deps


def _load_pyproject_dependencies(pyproject_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    try:
        import tomllib  # Python 3.11+
    except Exception:
        tomllib = None

    if tomllib is not None:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        deps = project.get("dependencies", []) or []
        optional = project.get("optional-dependencies", {}) or {}
        return list(deps), {k: list(v) for k, v in optional.items()}

    # Fallback parser for older Python versions
    deps = _parse_dependencies_fallback(pyproject_path.read_text(encoding="utf-8"))
    return deps, {}


def _build_pip_command(
    deps: list[str],
    use_user: bool,
    upgrade: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_user:
        cmd.append("--user")
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(extra_args)
    cmd.extend(deps)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-install dependencies.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to the project root (default: current directory).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Optional dependency group(s) to include.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Install into the system site-packages (default: --user).",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade packages to the latest compatible versions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without executing it.",
    )
    parser.add_argument(
        "pip_args",
        nargs="*",
        help="Additional arguments to pass to pip.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pyproject_path = root / "pyproject.toml"
    deps, optional = _load_pyproject_dependencies(pyproject_path)

    extras = []
    for extra in args.extra:
        if extra not in optional:
            print(f"ERROR: Optional dependency group '{extra}' not found.")
            return 1
        extras.extend(optional[extra])

    combined = deps + extras
    if not combined:
        print("ERROR: No dependencies found to install.")
        return 1

    cmd = _build_pip_command(
        combined, use_user=not args.system, upgrade=args.upgrade, extra_args=args.pip_args
    )
    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
