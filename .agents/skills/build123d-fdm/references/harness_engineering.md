# Harness Engineering Reference for `CodeCad`

This document archives the architectural rationale and research principles behind
`CodeCad`'s agentic harness. In accordance with **Single Source of Truth (SSoT)**,
this file describes *why* each layer exists and points to its canonical owner — it
never duplicates specific thresholds, linter rule codes, or config flags.

## 1. Core Doctrine: `Agent = Model + Harness`

Synthesized from industry practitioners (**Mitchell Hashimoto**, **Armin Ronacher**,
**Thorsten Ball**, **Simon Willison**, and **Anthropic Engineering**, 2025–2026):

- **Probabilistic vs. Deterministic Control**:
  - Prose instructions (`AGENTS.md`) operate inside the LLM attention window and degrade over long trajectories.
  - Deterministic guardrails (lifecycle hooks, AST linters, strict type-checkers, physical B-Rep assertions, and headless verification gates) operate outside the LLM loop and provide hard enforcement.
- **The Hashimoto Rule**: Whenever an agent makes a recurring mistake, do not merely add another bullet point to `AGENTS.md` — engineer a deterministic barrier in the harness so the mistake becomes impossible.
- **Irrefutable Proof (Thorsten Ball / Simon Willison)**: Never trust conversational claims that a part "looks good" or "compiles". Require deterministic artifacts (`report.json`, `.step`, zero-exit static checks).

## 2. Single Source of Truth (SSoT) & Progressive Disclosure

- **Single Source of Truth (SSoT)**: Every fact, threshold, and rule has exactly one canonical owner:
  - Printer constants, DFM thresholds, and color semantics -> `preview_pipeline/models.py`
  - Linter rules, banned APIs, and type-checking strictness -> `pyproject.toml`
  - File-size limits, command/file policies, and verification gates -> `.agents/guardrails.py`
  - External real-world measurements and design intent -> `<project>/README.md`
  - *Rule for agents*: Never copy config flags, rule codes, or numeric constants into markdown docs; reference the canonical owner instead.
- **Progressive Disclosure**: Keep `AGENTS.md` concise with global invariants only; load domain runbooks (`.agents/skills/`) on demand when working in that domain.

## 3. Modularity & Cognitive File-Size Ceiling

Empirical benchmarks on AI coding agents show sharp degradation on large monolithic files:
- Lost-in-the-middle attention decay and line-number drift during edits.
- Multi-chunk reading overhead and silent omissions during full-file rewrites.
- Tendency of LLMs to append to existing files instead of creating clean modules.

`CodeCad` enforces a strict per-file line ceiling in `.agents/guardrails.py` (both pre-write and on stop) to keep every module small, focused, and readable in a single pass.

## 4. Neuro-Symbolic VLM + CAD Verification Loop

Grounded in *CADCodeVerify* (ICLR 2025), *CADFusion* (ICML 2025), and *Seek-CAD* (2025):

1. **Zero-OCR Policy**: VLMs hallucinate when reading tiny dimensions from 3D renders. Exact numeric measurements, feature catalogs, and check statuses live in `report.json`.
2. **Set-of-Mark (SoM) Anchoring**: Connected defect clusters receive stable IDs (`D1`, `D2`, ...) in `report.json` and matching callout badges in both 3D and 2D views.
3. **Strict Parallel Projection + 360° Octant Coverage**: Eliminates perspective distortion and self-occlusion blind spots.
4. **Cutaway + 2D Vector HLR & Hatched Section**: Guarantees internal walls, clearances, and Z-levels are verifiable on a millimeter grid.
5. **Anti-TNP Code Structure**: Most LLM CAD failures stem from OpenCASCADE `fillet`/`chamfer` errors and unstable topological indices. Hybrid 2D-Sketch + 3D-Algebra Mode with geometric selectors and `fillet-last` prevents these failures.

## 5. Three-Tier "Warnings = Errors" Policy

To prevent silent quality drift, warnings are promoted to hard failures across all three layers:
1. **Static**: Configured in `pyproject.toml` (linter + strict type-checker).
2. **Runtime**: Configured in `preview.py` (repository runtime warnings treated as errors).
3. **Geometric DFM**: Enforced by `preview.py` and `.agents/guardrails.py` (any non-passing DFM status blocks completion).
