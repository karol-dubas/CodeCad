# %%
from pathlib import Path

from build123d import *
from ocp_vscode import *

# %%
# --- Parameters (mm) ---
plate_d = 240.0  # Largest weight plate diameter
handle_d = 28.0  # Handle grip diameter
handle_len = 127.0  # Exact length between inner collars

plate_clearance = 10.0  # Clearance beneath the 240 mm plate
saddle_d = 29.5  # Saddle diameter (28 mm + 1.5 mm clearance)
saddle_r = saddle_d / 2.0

# Z heights (Z=0 is the flat print bed / floor)
center_z = plate_d / 2.0 + plate_clearance  # 130.0 mm (handle axis height)
saddle_bottom_z = center_z - saddle_r  # 115.25 mm
top_z = center_z + 12.0  # 142.0 mm (lip height to cradle the handle)

# X dimensions (along the handle axis)
top_width = 122.0  # Width at the top saddle (leaves 2.5 mm margin per side inside 127 mm handle)
hand_gap = 90.0  # 90 mm clear space at the top/middle for the hand
tower_w = (top_width - hand_gap) / 2.0  # 16.0 mm width at the top of each tower

# Inner & Outer lateral buttresses (flare out at the bottom against lateral wobble)
# At Z < 72 mm, the 240 mm weight plate curves away from the handle axis
# (at Z=70 it is 23 mm away; at Z=0 it is 120 mm away), so outer buttresses have huge clearance!
inner_bottom_gap = 50.0  # Inner gap narrows from 90 mm to 50 mm at the base (+20 mm inner flare per side)
outer_base_width = (
    162.0  # Base widens outward from 122 mm to 162 mm (+20 mm outer flare per side)
)
gusset_top_z = 72.0  # Height where both inner and outer buttresses begin flaring out
bridge_h = 16.0  # Solid flat floor bridge thickness

# Y dimensions (front-to-back depth for stability)
base_depth = 140.0  # Footprint depth on the floor
top_depth = 56.0  # Depth at the top of the towers
rib_wall = 16.0  # Front and back sloped flange thickness on the buttresses


def make_dumbbell_stand() -> Part:
    # 1. Main side profile (YZ plane, extruded across full outer_base_width)
    # 100% flat bottom at Z=0 -> prints upright on the bed with zero supports
    with BuildSketch(Plane.YZ) as side_sk:
        with BuildLine():
            Line((-base_depth / 2, 0), (base_depth / 2, 0))
            Line((base_depth / 2, 0), (top_depth / 2, top_z))
            Line((top_depth / 2, top_z), (-top_depth / 2, top_z))
            Line((-top_depth / 2, top_z), (-base_depth / 2, 0))
        make_face()
        top_verts = side_sk.vertices().filter_by_position(Axis.Y, top_z - 1, top_z + 1)
        fillet(top_verts, radius=6.0)

    body = extrude(side_sk.sketch, amount=outer_base_width / 2, both=True)

    # 2. Outer contour shaping in XZ plane (keeps top at 122 mm so it fits inside 127 mm handle,
    #    flares out below Z=72 mm to 162 mm at the floor for external stabilization)
    with BuildSketch(Plane.XZ) as outer_xz_sk:
        Polygon(
            (-outer_base_width / 2, 0.0),
            (outer_base_width / 2, 0.0),
            (outer_base_width / 2, bridge_h),
            (top_width / 2, gusset_top_z),
            (top_width / 2, top_z + 10.0),
            (-top_width / 2, top_z + 10.0),
            (-top_width / 2, gusset_top_z),
            (-outer_base_width / 2, bridge_h),
            align=None,
        )
        outer_bend_verts = outer_xz_sk.vertices().filter_by_position(
            Axis.Y, bridge_h - 1.0, gusset_top_z + 1.0
        )
        fillet(outer_bend_verts, radius=18.0)

    outer_envelope = extrude(outer_xz_sk.sketch, amount=base_depth, both=True)
    body = body & outer_envelope

    # 3. Saddle cutout (YZ plane, extruded through all X)
    flare_top_y = saddle_r + 6.5
    with BuildSketch(Plane.YZ) as saddle_sk:
        with BuildLine():
            p_tl = (-flare_top_y, top_z + 5.0)
            p_ml = (-saddle_r, center_z)
            p_mr = (saddle_r, center_z)
            p_tr = (flare_top_y, top_z + 5.0)

            Line(p_tl, p_ml)
            ThreePointArc(p_ml, (0, center_z - saddle_r), p_mr)
            Line(p_mr, p_tr)
            Line(p_tr, p_tl)
        make_face()

    saddle_cut = extrude(saddle_sk.sketch, amount=outer_base_width, both=True)
    body = body - saddle_cut

    saddle_top_edges = (
        body.edges()
        .filter_by(Axis.X)
        .filter_by_position(Axis.Z, top_z - 1.0, top_z + 1.0)
    )
    body = fillet(saddle_top_edges, radius=4.0)

    # 4. Inner hand clearance + inner buttress profile (XZ plane, extruded through all Y)
    # Full 90 mm hand gap at the top, flaring inward below Z=72 mm to 50 mm at Z=16 mm.
    # Combined with the outer buttresses, each tower base is 56 mm wide at the floor!
    with BuildSketch(Plane.XZ) as gap_sk:
        Polygon(
            (-hand_gap / 2, top_z + 10.0),
            (hand_gap / 2, top_z + 10.0),
            (hand_gap / 2, gusset_top_z),
            (inner_bottom_gap / 2, bridge_h),
            (-inner_bottom_gap / 2, bridge_h),
            (-hand_gap / 2, gusset_top_z),
            align=None,
        )
        inner_verts = gap_sk.vertices().filter_by_position(
            Axis.Y, bridge_h - 1.0, gusset_top_z + 1.0
        )
        fillet(inner_verts, radius=18.0)

    gap_cut = extrude(gap_sk.sketch, amount=base_depth, both=True)
    body = body - gap_cut

    # 5. Pocket out the middle web of both the inner and outer buttresses (leaving the 16 mm
    # main tower web + front and rear 16 mm sloped gusset ribs on both inside and outside)
    with BuildSketch(Plane.YZ) as pocket_yz:
        with BuildLine():
            p1 = (-base_depth / 2 + rib_wall, bridge_h)
            p2 = (base_depth / 2 - rib_wall, bridge_h)
            p3 = (top_depth / 2 - rib_wall, top_z)
            p4 = (-top_depth / 2 + rib_wall, top_z)
            Line(p1, p2)
            Line(p2, p3)
            Line(p3, p4)
            Line(p4, p1)
        make_face()
        bot_p_verts = pocket_yz.vertices().filter_by_position(
            Axis.Y, bridge_h - 1.0, bridge_h + 1.0
        )
        fillet(bot_p_verts, radius=10.0)

    # Inner buttress pockets (between X = -43 and +43, leaving 2 mm step before the 16 mm main tower web)
    inner_pocket = extrude(pocket_yz.sketch, amount=hand_gap / 2 - 2.0, both=True)
    body = body - inner_pocket

    # Outer buttress pockets (outside X = 63 and X = -63, leaving 2 mm step from the 16 mm main tower web)
    outer_pocket_pos = Pos(top_width / 2 + 2.0, 0, 0) * extrude(
        pocket_yz.sketch, amount=outer_base_width
    )
    outer_pocket_neg = mirror(outer_pocket_pos, about=Plane.YZ)
    body = body - outer_pocket_pos - outer_pocket_neg

    # 6. Larger side weight-saving triangular window in the towers (YZ plane)
    # Steep ~61° roof prints cleanly without supports
    with BuildSketch(Plane.YZ) as window_sk:
        Polygon(
            (-32.0, 26.0),
            (32.0, 26.0),
            (0.0, 92.0),
            align=None,
        )
        fillet(window_sk.vertices(), radius=8.0)

    window_cut = extrude(window_sk.sketch, amount=outer_base_width, both=True)
    body = body - window_cut

    # 7. Center weight-saving window in the flat base (leaves front & back 32 mm wide base beams)
    with BuildSketch(Plane.XY) as base_cut_sk:
        RectangleRounded(inner_bottom_gap - 6.0, base_depth - 2 * 32.0, radius=12.0)

    base_window = extrude(base_cut_sk.sketch, amount=bridge_h + 5.0)
    body = body - base_window

    # 8. Fillet all sharp edges above Z=0
    bottom_edges = set(body.faces().sort_by(Axis.Z)[0].edges())
    edges_to_fillet = [e for e in body.edges() if e not in bottom_edges]
    body = fillet(edges_to_fillet, radius=1.5)

    return body


stand = make_dumbbell_stand()
show(stand)

out_dir = Path(__file__).resolve().parent
export_step(stand, str(out_dir / "dumbbell_stand.step"))

# %%
