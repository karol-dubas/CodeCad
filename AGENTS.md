# CodeCad — Repository Rules

- **Purpose**: Parametric 3D models (`build123d`) for FDM 3D printing on a **Bambu Lab P1S** sliced in **OrcaSlicer**. Interactive viewer: VS Code + OCP CAD Viewer.
- **Toolchain**: Always execute commands with `uv run ...` (never bare `python` or `pip`). Export `.step` files only (never `.stl`).
- **Single Source of Truth (SSoT)**:
  - Code and configuration (`pyproject.toml`, `.agents/guardrails.py`, `preview_pipeline/models.py`) are the sole source of truth for numeric thresholds, file-size limits, colors, linter/type rules, and printer constants.
  - `<project_dir>/README.md` records **only** external ground-truth context that cannot be derived from code: user requirements, physical mating measurements, load/environmental constraints, and key design decisions.
  - **Flow Direction**: External specs are written into `<project_dir>/README.md` **first**, and code is generated from them (`README.md -> Code`). Never copy back code parameters, config flags, linter rule codes, numeric constants, or report metrics into documentation.
- **Progressive Disclosure**:
  - Keep this file concise with global invariants only.
  - Before writing or editing any `build123d` CAD geometry, load and follow `.agents/skills/build123d-fdm/SKILL.md` (which contains `build123d` Algebra/Sketch patterns, anti-TNP rules, FDM tolerances, and a reference to `references/harness_engineering.md`).
- **Strict Quality & Modularity (Warnings = Errors)**:
  - Obey the per-file line limit enforced by `.agents/guardrails.py` (`MAX_PYTHON_LINES`) and split logic into focused modules.
  - Full type annotations, explicit imports (no `import *`), zero linter/type-checker warnings (`pyproject.toml`), and zero non-`PASS` DFM statuses in `report.json`.
  - Organize multiple design versions into separate folders.
- **CAD & Compliance Verification Pipeline**:
  - After every geometry change, run `uv run preview.py <script_path>` (add `--slice` for OrcaSlicer G-code telemetry).
  - Read `<project_dir>/.preview/report.json` **first** for exact numbers and `PASS`/`WARN`/`FAIL` checks (Zero-OCR policy), then visually inspect `preview.png` and `preview_engineering.png`.
  - After completing workspace edits, run a fresh-context `Repo Compliance Reviewer` subagent per `.agents/skills/repo-reviewer/SKILL.md`.
