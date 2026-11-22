"""Deterministic PreToolUse and Stop guardrails for the CodeCad repository."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_PYTHON_LINES = 500
BLOCKED_BASENAME = "todo.md"
SKIP_DIRS = {".venv", ".git", "__pycache__", ".preview", ".ruff_cache"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))


def _is_blocked_path(raw_path: str) -> bool:
    normalized = raw_path.replace("\\", "/").lower()
    return normalized == BLOCKED_BASENAME or normalized.endswith(f"/{BLOCKED_BASENAME}")


def _command_touches_blocked(cmd: str) -> bool:
    pattern = rf"(^|[\s\"'/\\=:,;]){re.escape(BLOCKED_BASENAME)}($|[\s\"'.,;:)])"
    return re.search(pattern, cmd.lower()) is not None


def _command_uses_bare_python_or_pip(cmd: str) -> bool:
    segments = re.split(r"(?:&&|\|\||;|\|)", cmd)
    for seg in segments:
        stripped = seg.strip()
        if not stripped:
            continue
        if re.match(r"^(?:python|python3|py|pip|pip3)(?:\.exe)?(?:\s|$)", stripped, re.IGNORECASE):
            return True
    return False


def _estimate_new_line_count(tool_name: str, args: dict[str, Any], path: Path) -> int | None:
    if tool_name == "write_file":
        content = str(args.get("Content", ""))
        new_lines = len(content.splitlines())
        if args.get("Append") and path.exists():
            existing = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            return existing + new_lines
        return new_lines
    if tool_name == "edit_file" and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        target = str(args.get("Target", ""))
        replacement = str(args.get("Replacement", ""))
        updated = text.replace(target, replacement) if target else text
        return len(updated.splitlines())
    return None


def _handle_pre_tool_use(payload: dict[str, Any]) -> dict[str, Any]:
    tool_call = payload.get("toolCall", {})
    tool_name = str(tool_call.get("name", ""))
    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        return {"decision": "allow"}

    # 1. Check path arguments only (never scan file Content)
    for key in ("Path", "TargetFile", "ScriptPath"):
        val = args.get(key)
        if isinstance(val, str) and _is_blocked_path(val):
            return {"decision": "deny", "reason": f"Access to {BLOCKED_BASENAME} is blocked."}

    # 2. Check shell command arguments
    cmd = args.get("CommandLine")
    if isinstance(cmd, str):
        if _command_touches_blocked(cmd):
            return {"decision": "deny", "reason": f"Access to {BLOCKED_BASENAME} is blocked."}
        if _command_uses_bare_python_or_pip(cmd):
            return {
                "decision": "deny",
                "reason": "Use 'uv run ...' or 'uv add ...' instead of bare python/pip commands.",
            }

    # 3. Block .stl writes
    path_str = args.get("Path")
    if tool_name in {"write_file", "edit_file"} and isinstance(path_str, str):
        path = Path(path_str)
        if path.suffix.lower() == ".stl":
            return {
                "decision": "deny",
                "reason": "Export only .step files. Writing .stl files is forbidden.",
            }
        # 4. Enforce 500-line AI cognitive limit on repo .py files
        if path.suffix.lower() == ".py":
            try:
                resolved = path.resolve()
                is_in_repo = _repo_root() in resolved.parents
            except OSError:
                is_in_repo = False
            if is_in_repo:
                est_lines = _estimate_new_line_count(tool_name, args, resolved)
                if est_lines is not None and est_lines > MAX_PYTHON_LINES:
                    return {
                        "decision": "deny",
                        "reason": (
                            f"File {resolved.name} would have {est_lines} lines "
                            f"(limit: {MAX_PYTHON_LINES}). Split into modular files."
                        ),
                    }

    return {"decision": "allow"}


def _iter_repo_py_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        result.append(p)
    return sorted(result)


def _iter_cad_project_scripts(root: Path) -> list[Path]:
    scripts: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name in SKIP_DIRS:
            continue
        candidate = child / f"{child.name}.py"
        if candidate.is_file():
            scripts.append(candidate)
    return scripts


def _run_check(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output


def _is_repo_path(raw_path: str, root: Path) -> bool:
    try:
        resolved = Path(raw_path).resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _check_compliance_review(payload: dict[str, Any], root: Path) -> str | None:
    raw_transcript = payload.get("transcriptPath")
    if not isinstance(raw_transcript, str) or not raw_transcript:
        return None
    transcript_path = Path(raw_transcript)
    if not transcript_path.is_file():
        return None

    events: list[dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    last_user_idx = -1
    for i, ev in enumerate(events):
        if ev.get("type") == "USER_INPUT":
            last_user_idx = i

    turn_events = events[last_user_idx + 1 :] if last_user_idx >= 0 else events
    last_mutating_idx = -1
    last_review_call_idx = -1
    review_passed = False

    for i, ev in enumerate(turn_events):
        if ev.get("type") == "PLANNER_RESPONSE":
            for tc in ev.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                name = str(tc.get("name", ""))
                args = tc.get("args", {})
                if not isinstance(args, dict):
                    continue
                if name in {"write_file", "edit_file"}:
                    target = args.get("Path")
                    if isinstance(target, str) and _is_repo_path(target, root):
                        last_mutating_idx = i
                        review_passed = False
                elif name == "invoke_subagent":
                    role = str(args.get("Role", "")).lower()
                    if "reviewer" in role or "compliance" in role:
                        last_review_call_idx = i
                        review_passed = False
        elif (
            last_review_call_idx > last_mutating_idx
            and ev.get("type") in {"SYSTEM_MESSAGE", "GENERIC"}
        ):
            content = str(ev.get("content", ""))
            if "VERDICT: PASS" in content.upper():
                review_passed = True

    if last_mutating_idx == -1:
        return None
    if last_review_call_idx <= last_mutating_idx:
        return (
            "Compliance Review Gate: Workspace files were modified without a subsequent "
            "fresh-context review. Follow .agents/skills/repo-reviewer/SKILL.md and invoke "
            "a read-only 'research' subagent (Role: 'Repo Compliance Reviewer') before finishing."
        )
    if not review_passed:
        return (
            "Compliance Review Gate: Repo Compliance Reviewer did not return 'VERDICT: PASS' "
            "after the latest edits. Fix any FAIL items and re-verify."
        )
    return None


def _handle_stop(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("terminationReason") != "NO_TOOL_CALL":
        return {"decision": "allow"}
    if not payload.get("fullyIdle", True):
        return {"decision": "allow"}
    if payload.get("continuationActive"):
        return {"decision": "allow"}

    root = _repo_root()
    issues: list[str] = []

    # 1. File size limit check
    for py_file in _iter_repo_py_files(root):
        line_count = len(py_file.read_text(encoding="utf-8", errors="replace").splitlines())
        if line_count > MAX_PYTHON_LINES:
            rel = py_file.relative_to(root)
            issues.append(f"{rel} has {line_count} lines (max {MAX_PYTHON_LINES}).")

    # 2. Static analysis (ruff + basedpyright)
    ok_ruff, out_ruff = _run_check(["uv", "run", "ruff", "check", "."], root)
    if not ok_ruff:
        issues.append(f"ruff check failed:\n{out_ruff[:600]}")

    ok_pyright, out_pyright = _run_check(["uv", "run", "basedpyright"], root)
    if not ok_pyright:
        issues.append(f"basedpyright failed:\n{out_pyright[:600]}")

    # 3. CAD project freshness, README.md, .step, and strict PASS status
    for script in _iter_cad_project_scripts(root):
        proj_dir = script.parent
        rel_proj = proj_dir.relative_to(root)
        script_mtime = script.stat().st_mtime

        readme = proj_dir / "README.md"
        if not readme.is_file():
            issues.append(f"{rel_proj}: missing README.md engineering specification.")

        step_file = proj_dir / f"{proj_dir.name}.step"
        if not step_file.is_file():
            issues.append(f"{rel_proj}: missing exported {step_file.name}.")
        elif step_file.stat().st_mtime + 1.0 < script_mtime:
            issues.append(f"{rel_proj}: {step_file.name} is older than {script.name}.")

        report_file = proj_dir / ".preview" / "report.json"
        if not report_file.is_file():
            issues.append(f"{rel_proj}: missing .preview/report.json (run uv run preview.py).")
        else:
            if report_file.stat().st_mtime + 1.0 < script_mtime:
                issues.append(f"{rel_proj}: .preview/report.json is older than {script.name}.")
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                status = data.get("overall_status")
                if status != "PASS":
                    issues.append(
                        f"{rel_proj}: report.json overall_status is {status!r} "
                        "(strict policy requires PASS; warnings are treated as errors)."
                    )
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{rel_proj}: invalid report.json ({exc}).")

    if issues:
        bullet_list = "\n- ".join(issues)
        return {
            "decision": "continue",
            "reason": f"Deterministic Stop Gate failed. Fix all issues before finishing:\n- {bullet_list}",
        }

    review_issue = _check_compliance_review(payload, root)
    if review_issue:
        return {"decision": "continue", "reason": review_issue}
    return {"decision": "allow"}


def main() -> None:
    payload = json.load(sys.stdin)
    event = str(payload.get("hookEventName", ""))
    if event == "PreToolUse":
        _emit(_handle_pre_tool_use(payload))
    elif event == "Stop":
        _emit(_handle_stop(payload))
    else:
        _emit({"decision": "allow"})


if __name__ == "__main__":
    main()
