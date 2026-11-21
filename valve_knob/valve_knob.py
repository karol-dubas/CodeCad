# %%
from pathlib import Path

from build123d import *
from ocp_vscode import *

# %%
# --- Parameters (mm) ---
square_size = 5.0  # 5.0 mm stem square
square_corner_r = 0.4  # Rounded corners of the square socket
stem_depth = 10  # 10.0 mm stem length (square + ball) + 0.5 mm margin

thread_pocket_d = 13.0  # Clearance diameter for the protruding brass thread collar
thread_pocket_h = 5.0  # Depth of skirt covering the protruding thread collar

collar_d = 18.5  # Outer diameter matching the 18.5 mm chrome valve sleeve
base_hub_d = 20.5  # Slightly wider lower hub for strength and clean step
front_wall = 3.0  # Solid floor thickness at the front face (Z=0)
total_h = front_wall + stem_depth + thread_pocket_h  # 18.5 mm total height

wing_radius = 19.5  # < 23.0 mm clearance to the lower knob (39 mm max span)
wing_h = 10.0  # Handle grip height from the front face (Z=0)


def make_knob(handle_2d: Sketch) -> Part:
    """Build a complete valve knob from a 2D handle profile with shared inner dimensions."""
    grip_part = extrude(handle_2d, amount=wing_h)
    grip_part = chamfer(grip_part.faces().sort_by(Axis.Z)[0].edges(), length=0.8)
    grip_part = fillet(grip_part.faces().sort_by(Axis.Z)[-1].edges(), radius=1.2)

    collar_part = Cylinder(
        radius=collar_d / 2,
        height=total_h,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    body = grip_part + collar_part

    # Square socket for the 5 mm square stem + ball tip
    sq_profile = RectangleRounded(square_size, square_size, radius=square_corner_r)
    sq_hole = Pos(0, 0, front_wall) * extrude(sq_profile, amount=total_h)

    # Cylindrical counterbore for the protruding brass threaded collar
    thread_pocket = Pos(0, 0, front_wall + stem_depth) * Cylinder(
        radius=thread_pocket_d / 2,
        height=thread_pocket_h + 1.0,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    part = body - sq_hole - thread_pocket

    # Chamfer the entrance edges at the top (valve side) and square socket lead-in
    top_edges = part.faces().sort_by(Axis.Z)[-1].edges()
    socket_entry_z = front_wall + stem_depth
    socket_entry_face = part.faces().filter_by_position(
        Axis.Z, socket_entry_z - 0.1, socket_entry_z + 0.1
    )[0]
    return chamfer(top_edges + socket_entry_face.inner_wires().edges(), length=0.5)


# --- Style 1: Butterfly (2-wing handle) ---
sk_butterfly = Circle(base_hub_d / 2) + SlotOverall(width=2 * wing_radius, height=8.5)
sk_butterfly = fillet(
    sk_butterfly.vertices().filter_by_position(Axis.X, -base_hub_d / 2, base_hub_d / 2),
    radius=5.0,
)
knob_butterfly = make_knob(sk_butterfly)

# --- Style 2: Trilobe (3-arm ergonomic handle matching the lower knob style) ---
sk_trilobe = Circle(base_hub_d / 2)
for angle in (0, 120, 240):
    arm = Rot(0, 0, angle) * Pos(wing_radius / 2, 0) * SlotOverall(wing_radius, 8.5)
    sk_trilobe = sk_trilobe + arm
sk_trilobe = fillet(sk_trilobe.vertices(), radius=4.5)
knob_trilobe = make_knob(sk_trilobe)

# --- Style 3: Fluted Star (6-lobe compact round grip, diameter 36 mm) ---
star_outer_r = 18.0
flute_centers = PolarLocations(radius=star_outer_r + 2.2, count=6)
sk_star = Circle(star_outer_r) - flute_centers * Circle(4.8)
sk_star = fillet(sk_star.vertices(), radius=2.0)
knob_star = make_knob(sk_star)

# --- Style 4: Single Lever / Teardrop (compact 1-arm lever) ---
with BuildSketch() as lever_sk:
    Circle(base_hub_d / 2)
    with Locations((wing_radius - 4.5, 0)):
        Circle(4.5)
    make_hull()
knob_lever = make_knob(lever_sk.sketch)

# --- Choose active style or view all ---
STYLE = "butterfly"  # Options: "butterfly", "trilobe", "star", "lever"
styles = {
    "butterfly": knob_butterfly,
    "trilobe": knob_trilobe,
    "star": knob_star,
    "lever": knob_lever,
}

knob = styles[STYLE]
show(knob)

# Export all 4 styles Ready for Orca Slicer (.step only)
out_dir = Path(__file__).resolve().parent
for name, part in styles.items():
    export_step(part, str(out_dir / f"valve_knob_{name}.step"))
export_step(knob, str(out_dir / "valve_knob.step"))

# %%
