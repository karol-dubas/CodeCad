# %%
from __future__ import annotations

from pathlib import Path

from build123d import (
    Align,
    Axis,
    Circle,
    Cylinder,
    Part,
    Pos,
    RectangleRounded,
    Sketch,
    SlotOverall,
    chamfer,
    export_step,
    extrude,
    fillet,
)
from ocp_vscode import show

# %%
# --- Parameters (mm) ---
SQUARE_SIZE = 5.0  # 5.0 mm stem square
SQUARE_CORNER_R = 0.4  # Rounded corners of the square socket
STEM_DEPTH = 10.0  # 10.0 mm stem length (square + ball) + margin

THREAD_POCKET_D = 13.0  # Clearance diameter for the protruding brass thread collar
THREAD_POCKET_H = 5.0  # Depth of skirt covering the protruding thread collar

COLLAR_D = 18.5  # Outer diameter matching the 18.5 mm chrome valve sleeve
BASE_HUB_D = 20.5  # Slightly wider lower hub for strength and clean step
FRONT_WALL = 3.0  # Solid floor thickness at the front face (Z=0)
TOTAL_H = FRONT_WALL + STEM_DEPTH + THREAD_POCKET_H  # 18.0 mm total height

WING_RADIUS = 19.5  # < 23.0 mm clearance to the lower knob (39 mm max span)
WING_H = 10.0  # Handle grip height from the front face (Z=0)
OVERCUT = 1.0  # Boolean overcut margin to avoid coplanar OCCT artifacts


def make_butterfly_profile() -> Sketch:
    """Build the 2D butterfly (2-wing) handle profile with smooth hub transitions."""
    profile = Circle(BASE_HUB_D / 2.0) + SlotOverall(width=2.0 * WING_RADIUS, height=8.5)
    blend_verts = profile.vertices().filter_by_position(
        Axis.X, -BASE_HUB_D / 2.0, BASE_HUB_D / 2.0
    )
    result = fillet(blend_verts, radius=5.0)
    assert isinstance(result, Sketch)
    return result


def make_valve_knob(handle_2d: Sketch) -> Part:
    """Build the complete butterfly valve knob with square stem socket and thread skirt."""
    grip_part = extrude(handle_2d, amount=WING_H)
    grip_part = chamfer(grip_part.faces().sort_by(Axis.Z)[0].edges(), length=0.8)
    grip_part = fillet(grip_part.faces().sort_by(Axis.Z)[-1].edges(), radius=1.2)

    collar_part = Cylinder(
        radius=COLLAR_D / 2.0,
        height=TOTAL_H,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    body = grip_part + collar_part

    # Square socket for the 5 mm square stem + ball tip
    sq_profile = RectangleRounded(SQUARE_SIZE, SQUARE_SIZE, radius=SQUARE_CORNER_R)
    sq_hole = Pos(0.0, 0.0, FRONT_WALL) * extrude(sq_profile, amount=TOTAL_H)

    # Cylindrical counterbore for the protruding brass threaded collar
    thread_pocket = Pos(0.0, 0.0, FRONT_WALL + STEM_DEPTH) * Cylinder(
        radius=THREAD_POCKET_D / 2.0,
        height=THREAD_POCKET_H + OVERCUT,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    part = body - sq_hole - thread_pocket

    # Chamfer entrance edges at the top (valve side) and square socket lead-in
    top_edges = part.faces().sort_by(Axis.Z)[-1].edges()
    socket_entry_z = FRONT_WALL + STEM_DEPTH
    socket_entry_face = part.faces().filter_by_position(
        Axis.Z, socket_entry_z - 0.1, socket_entry_z + 0.1
    )[0]
    finished = chamfer(top_edges + socket_entry_face.inner_wires().edges(), length=0.5)
    assert isinstance(finished, Part)
    return finished


knob = make_valve_knob(make_butterfly_profile())

# --- Deterministic Physical Invariants ---
bbox = knob.bounding_box()
assert knob.is_valid, "Invalid OpenCASCADE B-Rep topology."
assert len(knob.solids()) == 1, f"Expected 1 solid, got {len(knob.solids())}."
assert abs(bbox.min.Z) < 1e-3, f"Part must rest flat on Z=0, got Z_min={bbox.min.Z:.3f}."

show(knob)

out_dir = Path(__file__).resolve().parent
export_step(knob, str(out_dir / "valve_knob.step"))

# %%
