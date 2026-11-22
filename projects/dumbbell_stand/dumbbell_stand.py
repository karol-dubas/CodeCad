# %%
from __future__ import annotations

from pathlib import Path

from build123d import (
    Axis,
    BuildLine,
    BuildSketch,
    Line,
    Part,
    Plane,
    Polygon,
    RectangleRounded,
    Sketch,
    ThreePointArc,
    export_step,
    extrude,
    fillet,
    make_face,
)
from ocp_vscode import show

# %%
# --- Parameters (mm) ---
PLATE_D = 240.0  # Largest weight plate diameter
HANDLE_D = 28.0  # Handle grip diameter
HANDLE_LEN = 127.0  # Exact length between inner collars

PLATE_CLEARANCE = 10.0  # Clearance beneath the 240 mm plate
SADDLE_D = 29.5  # Saddle diameter (28 mm + 1.5 mm clearance)
SADDLE_R = SADDLE_D / 2.0

# Z heights (Z=0 is the flat print bed / floor)
CENTER_Z = PLATE_D / 2.0 + PLATE_CLEARANCE  # 130.0 mm (handle axis height)
TOP_Z = CENTER_Z + 12.0  # 142.0 mm (lip height to cradle the handle)

# X dimensions (along the handle axis)
TOP_WIDTH = 122.0  # Fits inside 127 mm handle with 2.5 mm margin per side
HAND_GAP = 90.0  # Clear space at the top/middle for the hand
INNER_BOTTOM_GAP = 54.0  # Inner flare (+18 mm per side at base)
OUTER_BASE_WIDTH = 158.0  # Outer flare (+18 mm per side at base)
GUSSET_TOP_Z = 72.0  # Height where inner/outer buttresses start flaring
BRIDGE_H = 14.0  # Solid flat floor bridge thickness

# Y dimensions (front-to-back depth for stability)
BASE_DEPTH = 140.0  # Footprint depth on the floor
TOP_DEPTH = 56.0  # Depth at the top of the towers


def _require_sketch(builder: BuildSketch) -> Sketch:
    assert isinstance(builder.sketch, Sketch)
    return builder.sketch


def make_dumbbell_stand() -> Part:
    """Build the single-piece support-free adjustable dumbbell floor stand."""
    # 1. Main side profile (YZ plane, extruded across full outer base width)
    with BuildSketch(Plane.YZ) as side_sk:
        with BuildLine():
            Line((-BASE_DEPTH / 2.0, 0.0), (BASE_DEPTH / 2.0, 0.0))
            Line((BASE_DEPTH / 2.0, 0.0), (TOP_DEPTH / 2.0, TOP_Z))
            Line((TOP_DEPTH / 2.0, TOP_Z), (-TOP_DEPTH / 2.0, TOP_Z))
            Line((-TOP_DEPTH / 2.0, TOP_Z), (-BASE_DEPTH / 2.0, 0.0))
        make_face()
        top_verts = side_sk.vertices().filter_by_position(Axis.Y, TOP_Z - 1.0, TOP_Z + 1.0)
        fillet(top_verts, radius=6.0)

    body = extrude(_require_sketch(side_sk), amount=OUTER_BASE_WIDTH / 2.0, both=True)

    # 2. Outer contour shaping in XZ plane
    with BuildSketch(Plane.XZ) as outer_xz_sk:
        Polygon(
            (-OUTER_BASE_WIDTH / 2.0, 0.0),
            (OUTER_BASE_WIDTH / 2.0, 0.0),
            (OUTER_BASE_WIDTH / 2.0, BRIDGE_H),
            (TOP_WIDTH / 2.0, GUSSET_TOP_Z),
            (TOP_WIDTH / 2.0, TOP_Z + 10.0),
            (-TOP_WIDTH / 2.0, TOP_Z + 10.0),
            (-TOP_WIDTH / 2.0, GUSSET_TOP_Z),
            (-OUTER_BASE_WIDTH / 2.0, BRIDGE_H),
            align=None,
        )
        outer_bend_verts = outer_xz_sk.vertices().filter_by_position(
            Axis.Y, BRIDGE_H - 1.0, GUSSET_TOP_Z + 1.0
        )
        fillet(outer_bend_verts, radius=20.0)

    outer_envelope = extrude(_require_sketch(outer_xz_sk), amount=BASE_DEPTH, both=True)
    body = body & outer_envelope

    # 3. Saddle cutout (YZ plane, extruded through all X)
    flare_top_y = SADDLE_R + 6.5
    with BuildSketch(Plane.YZ) as saddle_sk:
        with BuildLine():
            p_tl = (-flare_top_y, TOP_Z + 5.0)
            p_ml = (-SADDLE_R, CENTER_Z)
            p_mr = (SADDLE_R, CENTER_Z)
            p_tr = (flare_top_y, TOP_Z + 5.0)
            Line(p_tl, p_ml)
            ThreePointArc(p_ml, (0.0, CENTER_Z - SADDLE_R), p_mr)
            Line(p_mr, p_tr)
            Line(p_tr, p_tl)
        make_face()

    saddle_cut = extrude(_require_sketch(saddle_sk), amount=OUTER_BASE_WIDTH, both=True)
    body = body - saddle_cut

    saddle_top_edges = (
        body.edges().filter_by(Axis.X).filter_by_position(Axis.Z, TOP_Z - 1.0, TOP_Z + 1.0)
    )
    body = fillet(saddle_top_edges, radius=4.0)

    # 4. Inner hand clearance + smooth inner buttress profile (XZ plane)
    with BuildSketch(Plane.XZ) as gap_sk:
        Polygon(
            (-HAND_GAP / 2.0, TOP_Z + 10.0),
            (HAND_GAP / 2.0, TOP_Z + 10.0),
            (HAND_GAP / 2.0, GUSSET_TOP_Z),
            (INNER_BOTTOM_GAP / 2.0, BRIDGE_H),
            (-INNER_BOTTOM_GAP / 2.0, BRIDGE_H),
            (-HAND_GAP / 2.0, GUSSET_TOP_Z),
            align=None,
        )
        inner_verts = gap_sk.vertices().filter_by_position(
            Axis.Y, BRIDGE_H - 1.0, GUSSET_TOP_Z + 1.0
        )
        fillet(inner_verts, radius=20.0)

    gap_cut = extrude(_require_sketch(gap_sk), amount=BASE_DEPTH, both=True)
    body = body - gap_cut

    # 5. Clean side A-frame triangular window through towers & buttresses (YZ plane)
    # Steep ~61 deg roof with sharp apex prints completely support-free
    with BuildSketch(Plane.YZ) as window_sk:
        Polygon((-38.0, 22.0), (38.0, 22.0), (0.0, 92.0), align=None)
        win_bot_verts = window_sk.vertices().filter_by_position(Axis.Y, 21.0, 23.0)
        fillet(win_bot_verts, radius=10.0)

    window_cut = extrude(_require_sketch(window_sk), amount=OUTER_BASE_WIDTH, both=True)
    body = body - window_cut

    # 6. Center weight-saving window in the flat base
    with BuildSketch(Plane.XY) as base_cut_sk:
        RectangleRounded(INNER_BOTTOM_GAP - 6.0, BASE_DEPTH - 2.0 * 28.0, radius=12.0)

    base_window = extrude(_require_sketch(base_cut_sk), amount=BRIDGE_H + 5.0)
    body = body - base_window

    # 7. Fillet all sharp edges above Z=0 (excluding the triangular window apex to keep it support-free)
    bottom_edges = set(body.faces().sort_by(Axis.Z)[0].edges())
    window_apex_edges = set(
        body.edges()
        .filter_by(Axis.X)
        .filter_by_position(Axis.Y, -1.0, 1.0)
        .filter_by_position(Axis.Z, 91.0, 93.0)
    )
    edges_to_fillet = [
        e for e in body.edges() if e not in bottom_edges and e not in window_apex_edges
    ]
    finished = fillet(edges_to_fillet, radius=2.0)
    assert isinstance(finished, Part)
    return finished


stand = make_dumbbell_stand()

# --- Deterministic Physical Invariants ---
bbox = stand.bounding_box()
assert stand.is_valid, "Invalid OpenCASCADE B-Rep topology."
assert len(stand.solids()) == 1, f"Expected 1 solid, got {len(stand.solids())}."
assert abs(bbox.min.Z) < 1e-3, f"Part must rest flat on Z=0, got Z_min={bbox.min.Z:.3f}."

show(stand)

out_dir = Path(__file__).resolve().parent
export_step(stand, str(out_dir / "dumbbell_stand.step"))

# %%
