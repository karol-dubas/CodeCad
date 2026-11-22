---
name: build123d-fdm
description: >-
  Use this skill whenever designing, editing, or debugging build123d 3D CAD models
  for FDM 3D printing on the Bambu Lab P1S. Covers Algebra Mode + 2D Sketch patterns,
  OpenCASCADE topological naming resilience, FDM tolerances, and physical invariants.
---

# build123d & FDM Design Runbook (Bambu Lab P1S)

This skill defines the modeling patterns, FDM design rules, and verification
workflow for `CodeCad`. For the underlying harness architecture and research
rationale, see [harness_engineering.md](./references/harness_engineering.md).

## 1. Python & build123d Code Architecture

1. **Explicit Imports & Strict Typing**:
   - Never use `from build123d import *` or `from ocp_vscode import *`. Import only what you use so static analysis (`pyproject.toml`) catches typos and bad signatures before running OCCT.
   - Wrap geometry creation in a pure typed function `def make_<part>() -> Part:`.
2. **2D Sketches + 3D Algebra Mode (Hybrid Pattern)**:
   - Build 2D profiles in `BuildSketch` (or 2D algebra) and apply **2D corner fillets** (`fillet(sk.vertices(), radius=...)`) *before* `extrude()`.
   - Combine 3D solids exclusively in **stateless Algebra Mode** (`body = base + boss - pocket`, `Pos(x, y, z) * Cylinder(...)`, `Rot(...)`). Avoid deep nested `with BuildPart():` stacks that cause implicit-state confusion.
3. **Resilience Against OpenCASCADE TNP (Topological Naming Problem)**:
   - Never index raw topology lists (`part.faces()[3]`, `part.edges()[12]`).
   - Always use parametric geometric queries:
     - `part.faces().sort_by(Axis.Z)[0]` (bottom at `Z=0`) / `[-1]` (top).
     - `part.faces().filter_by_position(Axis.Z, z_target - 0.1, z_target + 0.1)`.
   - **Boolean Overcut**: When subtracting pockets or through-holes, extend the cutting tool past the target boundary (e.g., `OVERCUT = 1.0` mm) to avoid coplanar zero-thickness OCCT artifacts.
   - **Fillet / Chamfer Last**: Apply 3D `fillet()` and `chamfer()` at the very end of `make_<part>()`, ensuring every radius is strictly smaller than adjacent wall thicknesses and edge lengths.

## 2. FDM Design Rules (Bambu Lab P1S, 0.4 mm Nozzle)

- **Build Plate (`Z = 0`)**:
  - The part must rest flat on `Z = 0` with ample bed contact area and stable Center-of-Mass.
  - Use **45° chamfers** on `Z = 0` bottom edges — never curved fillets touching the build plate.
- **Support-Free Geometry**:
  - Keep all unsupported overhangs within safe print angles (`<= 45°` from vertical).
  - For vertical windows/cutouts, use steep triangular/teardrop roofs (`>= 55–60°` from horizontal) and exclude sharp apex edges from 3D `fillet()` so no flat ceiling overhang forms.
- **Wall Thickness & Material Efficiency**:
  - Keep structural walls above `WALL_WARN_MM` and strictly never below `WALL_CRIT_MM` (defined in `preview_pipeline/models.py`).
  - Hollow, rib, or add weight-saving cutouts where structurally sound instead of bulky solid volumes.
  - Align layer lines with primary tensile/bending loads; avoid sharp internal stress-concentrating corners.
- **Dimensional Clearances (PLA / PETG on Bambu Lab P1S)**:
  - Tight press / friction fit: `+0.05 to +0.10 mm` per side.
  - Sliding / removable mating fit: `+0.10 to +0.20 mm` per side.
  - Loose / clearance hole (e.g., M3 = `3.4 mm`, M4 = `4.5 mm`): `+0.40 mm` diameter.

## 3. Mandatory Physical Invariants (In-Script Assertions)

Every CAD script must end with deterministic B-Rep assertions before exporting `.step`
(checking build volume against `BUILD_VOL_MM` from `preview_pipeline/models.py` or
ensuring `part.is_valid`, `len(part.solids()) == 1`, and `abs(bbox.min.Z) < 1e-3`):

```python
bbox = part.bounding_box()
assert part.is_valid, "Invalid OpenCASCADE B-Rep topology."
assert len(part.solids()) == 1, f"Expected 1 solid, got {len(part.solids())}."
assert abs(bbox.min.Z) < 1e-3, f"Part must rest flat on Z=0, got Z_min={bbox.min.Z:.3f}."
```

## 4. Verification Workflow

1. Keep `<project_dir>/README.md` updated with real-world measurements and requirements.
2. Run `uv run <project>/<project>.py` to export `<project>.step` and verify assertions.
3. Run `uv run preview.py <project>/<project>.py` (strict mode: warnings are errors).
4. Read `<project>/.preview/report.json` first, then visually inspect `preview.png` and `preview_engineering.png`.
5. Run `uv run ruff check .` and `uv run basedpyright` before finishing.
