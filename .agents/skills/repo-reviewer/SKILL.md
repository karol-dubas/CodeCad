---
name: repo-reviewer
description: >-
  Use this skill to run an isolated, fresh-context compliance review after
  modifying repository files. Defines the read-only Repo Compliance Reviewer
  protocol, SSoT flow-direction checks, and structured PASS/FAIL verdict format.
---

# Repo Compliance Reviewer Protocol

Invoke a fresh-context, read-only `research` subagent (`Role="Repo Compliance Reviewer"`)
after finishing edits to repository files. Do **not** repeat rules from `AGENTS.md`
or other documents in the subagent prompt — instruct it to read those
canonical sources directly.

## Subagent Prompt Template

Pass this concise prompt to `invoke_subagent(TypeName="research", Role="Repo Compliance Reviewer", ...)`:

1. Read the `AGENTS.md` file and other relevant files for the current review on AI agents.
2. Inspect `git status` and `git diff` (plus any newly created files) for the latest changes.
3. Verify the changes against these **semantic** criteria (not covered by `ruff`/`basedpyright`):
   - **SSoT & Flow Direction**:
     - External measurements/requirements flow **only** `User -> <project>/README.md -> Code`.
     - Reject any copy-back from code, config, or report metrics into documentation, or rule duplication across docs.
   - **Intent & Anti-Reward-Hacking**: Changes match the user request with no unrequested refactors, weakened assertions, disabled lint rules, or unnecessary `# type: ignore`.
   - **CAD & Visual Audit** (when CAD `.py` changed): Verify `SKILL.md` modeling compliance and inspect `<project>/.preview/` files against `<project>/README.md`.
4. Return a strict structured verdict starting on the first line:
   - `VERDICT: PASS` (if all checks hold), or
   - `VERDICT: FAIL` followed by exact `file:line — rule — required fix` bullets.
