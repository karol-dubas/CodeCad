"""3D PyVista parallel-projection renderer for Sheet 1 (`preview.png`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import build123d as b3d
import numpy as np
import pyvista as pv
import pyvista_cad
from build123d import GeomType, Part
from matplotlib.colors import ListedColormap

from preview_pipeline.loader import crop_whitespace_png
from preview_pipeline.models import (
    COLOR_BADGE_AMBER,
    COLOR_BADGE_RED,
    COLOR_BED_BLUE,
    COLOR_BG_LIGHT,
    COLOR_CAP_CYAN,
    COLOR_CRIT_RED,
    COLOR_EDGE_DARK,
    COLOR_GHOST_GRAY,
    COLOR_OK_GRAY,
    COLOR_SAFE_CAD,
    COLOR_WARN_AMBER,
    CadReport,
    SectionCutSpec,
)


def build_edge_polydata(shape: Part) -> pv.PolyData | None:
    """Sample all exact build123d B-Rep edges into a single PyVista PolyData."""
    all_points: list[tuple[float, float, float]] = []
    all_lines: list[int] = []

    for edge in shape.edges():
        if edge.length < 1e-4:
            continue
        n = 2 if edge.geom_type == GeomType.LINE else max(20, min(80, int(edge.length / 0.22) + 2))
        pts = edge.positions([i / (n - 1) for i in range(n)])
        start_idx = len(all_points)
        all_points.extend((p.X, p.Y, p.Z) for p in pts)
        all_lines.append(n)
        all_lines.extend(range(start_idx, start_idx + n))

    if not all_points:
        return None
    return pv.PolyData(np.array(all_points, dtype=float), lines=np.array(all_lines, dtype=int))


def _add_edges(
    plotter: pv.Plotter,
    edge_mesh: pv.PolyData | None,
    color: str = COLOR_EDGE_DARK,
    line_width: float = 1.6,
    opacity: float = 1.0,
) -> None:
    if edge_mesh is None or edge_mesh.n_points == 0:
        return
    actor = plotter.add_mesh(
        edge_mesh,
        color=color,
        line_width=line_width,
        opacity=opacity,
        lighting=False,
        pickable=False,
    )
    actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
    actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -1.0)


def _set_parallel_iso_camera(
    plotter: pv.Plotter,
    report: CadReport,
    direction: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zoom: float = 0.90,
) -> None:
    """Configure an orthographic parallel-projection camera."""
    xmin, ymin, zmin = report.bbox_min
    xmax, ymax, zmax = report.bbox_max
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    dist = max(diag * 2.2, 10.0)
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    pos = (cx + dist * dx / norm, cy + dist * dy / norm, cz + dist * dz / norm)
    plotter_dyn: Any = plotter
    plotter_dyn.enable_parallel_projection()
    plotter.camera_position = [pos, (cx, cy, cz), up]
    plotter_dyn.reset_camera(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
    plotter.camera.zoom(zoom)


def _add_conditional_build_box(plotter: pv.Plotter, report: CadReport) -> None:
    """Render printer build-volume wireframe only if near or exceeding limits."""
    if not report.near_or_exceeds_build_volume:
        return
    xmin, ymin, zmin = report.bbox_min
    xmax, ymax, _ = report.bbox_max
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw, bd, bh = report.build_volume_mm
    box = pv.Box(
        bounds=(
            cx - bw / 2.0,
            cx + bw / 2.0,
            cy - bd / 2.0,
            cy + bd / 2.0,
            zmin,
            zmin + bh,
        )
    )
    exceeds = any(m < 0.0 for m in report.build_volume_margin_mm)
    box_color = COLOR_CRIT_RED if exceeds else COLOR_WARN_AMBER
    plotter.add_mesh(
        box,
        color=box_color,
        style="wireframe",
        line_width=2.2,
        opacity=0.85,
        lighting=False,
    )


def _add_3d_som_markers(plotter: pv.Plotter, report: CadReport) -> None:
    """Overlay Set-of-Mark 3D callout badges [D1], [D2] at defect centroids."""
    if not report.defect_markers:
        return
    r_pin = max(min(report.bbox_size) * 0.025, 0.7)
    for m in report.defect_markers:
        is_crit = "critical" in m.category or m.severity == "FAIL"
        badge_bg = COLOR_BADGE_RED if is_crit else COLOR_BADGE_AMBER
        pin_color = COLOR_CRIT_RED if is_crit else COLOR_WARN_AMBER
        pin_sphere = pv.Sphere(radius=r_pin, center=m.centroid_mm)
        plotter.add_mesh(pin_sphere, color=pin_color, lighting=False)
        plotter.add_point_labels(
            points=np.array([m.centroid_mm], dtype=float),
            labels=[f" {m.id}: {m.short_label} "],
            font_size=11,
            text_color="#FFFFFF",
            point_size=1,
            shape="rounded_rect",
            fill_shape=True,
            shape_color=badge_bg,
            shape_opacity=0.92,
            always_visible=True,
            show_points=False,
        )


def render_sheet1_3d_overview(
    shape: Part,
    analyzed_poly: pv.PolyData,
    edge_mesh: pv.PolyData | None,
    report: CadReport,
    section_spec: SectionCutSpec,
    out_path: Path,
) -> None:
    """Render Sheet 1 (`preview.png`): 2x2 Parallel-Projection 3D Spatial, DFM & Cutaway."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        shape=(2, 2),
        off_screen=True,
        window_size=[1800, 1500],
        lighting="light kit",
    )
    plotter_dyn: Any = plotter
    plotter_dyn.set_background(COLOR_BG_LIGHT)

    xmin, ymin, zmin = report.bbox_min
    xmax, ymax, _ = report.bbox_max
    sx, sy, _ = report.bbox_size
    cx, cy, cz = report.center_of_mass_mm

    pad_x = max(sx * 0.14, 5.0)
    pad_y = max(sy * 0.14, 5.0)
    span_x = sx + 2.0 * pad_x
    span_y = sy + 2.0 * pad_y
    grid_step = 10.0 if max(span_x, span_y) >= 40.0 else 5.0
    bed_plane = pv.Plane(
        center=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, zmin),
        direction=(0, 0, 1),
        i_size=span_x,
        j_size=span_y,
        i_resolution=max(4, min(32, round(span_x / grid_step))),
        j_resolution=max(4, min(32, round(span_y / grid_step))),
    )

    dfm_cmap = ListedColormap([COLOR_BED_BLUE, COLOR_SAFE_CAD, COLOR_WARN_AMBER, COLOR_CRIT_RED])

    top_iso_views = [
        ((0, 0), "1. Iso Front-Right-Top (+X -Y +Z) — CAD & DFM", (1.05, -1.15, 0.80)),
        ((0, 1), "2. Iso Back-Left-Top (-X +Y +Z) — Opposite Side", (-1.05, 1.15, 0.80)),
    ]
    for (row, col), title, cam_dir in top_iso_views:
        plotter.subplot(row, col)
        plotter.add_mesh(
            analyzed_poly,
            scalars="OverhangClass",
            cmap=dfm_cmap,
            clim=[-0.5, 3.5],
            preference="cell",
            show_scalar_bar=False,
            smooth_shading=True,
            ambient=0.28,
            specular=0.22,
            specular_power=22,
        )
        _add_edges(plotter, edge_mesh, color=COLOR_EDGE_DARK, line_width=1.6)
        plotter.add_mesh(
            bed_plane,
            color="#CBD5E1",
            style="wireframe",
            line_width=1.0,
            opacity=0.65,
            lighting=False,
        )
        _add_conditional_build_box(plotter, report)
        _add_3d_som_markers(plotter, report)
        plotter.add_text(title, position="upper_left", font_size=12, color=COLOR_EDGE_DARK)
        plotter_dyn.add_axes()
        _set_parallel_iso_camera(plotter, report, direction=cam_dir)

    plotter.subplot(1, 0)
    plotter.add_mesh(
        analyzed_poly,
        scalars="OverhangClass",
        cmap=dfm_cmap,
        clim=[-0.5, 3.5],
        preference="cell",
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.32,
        specular=0.15,
    )
    _add_edges(plotter, edge_mesh, color=COLOR_EDGE_DARK, line_width=1.5)
    _add_3d_som_markers(plotter, report)
    plotter.add_text(
        "3. Iso Underside (-Z) — Bed Contact (Blue) & Overhangs (Amber/Red)",
        position="upper_left",
        font_size=12,
        color=COLOR_EDGE_DARK,
    )
    plotter_dyn.add_axes()
    _set_parallel_iso_camera(plotter, report, direction=(-1.05, -1.15, -0.75))

    plotter.subplot(1, 1)
    wall_cmap = ListedColormap([COLOR_OK_GRAY, COLOR_WARN_AMBER, COLOR_CRIT_RED])
    clip_normal = (1.0, 0.0, 0.0) if section_spec.axis_name == "X" else (0.0, -1.0, 0.0)
    clipped_solid = analyzed_poly.clip(normal=clip_normal, origin=section_spec.origin, invert=True)
    if clipped_solid.n_cells > 0:
        plotter.add_mesh(
            clipped_solid,
            scalars="WallClass",
            cmap=wall_cmap,
            clim=[-0.5, 2.5],
            preference="cell",
            show_scalar_bar=False,
            smooth_shading=True,
            ambient=0.30,
            specular=0.18,
        )

    plotter.add_mesh(
        analyzed_poly,
        color="#94A3B8",
        opacity=0.07,
        smooth_shading=True,
        lighting=False,
    )
    _add_edges(plotter, edge_mesh, color=COLOR_GHOST_GRAY, line_width=1.0, opacity=0.22)

    if edge_mesh is not None and edge_mesh.n_points > 0:
        kept_edges = edge_mesh.clip(normal=clip_normal, origin=section_spec.origin, invert=True)
        if kept_edges.n_points > 0:
            _add_edges(plotter, kept_edges, color=COLOR_EDGE_DARK, line_width=1.5, opacity=0.90)

    try:
        sec_shape = b3d.section(shape, section_by=section_spec.plane)
        sec_faces = sec_shape.faces() if hasattr(sec_shape, "faces") else []
        if sec_faces:
            pyvista_cad_dyn: Any = pyvista_cad
            pyvista_cad_dyn.add_cad(
                plotter,
                sec_shape,
                color=COLOR_CAP_CYAN,
                edges=True,
                edge_color=COLOR_EDGE_DARK,
                line_width=2.6,
                linear_deflection=0.025,
                angular_deflection=0.10,
            )
    except Exception:
        slc = analyzed_poly.slice(normal=section_spec.normal, origin=section_spec.origin)
        if slc.n_cells > 0:
            plotter.add_mesh(slc, color=COLOR_CAP_CYAN, line_width=3.0, lighting=False)

    r_marker = max(min(report.bbox_size) * 0.032, 0.85)
    plotter.add_mesh(pv.Sphere(radius=r_marker, center=(cx, cy, cz)), color=COLOR_BED_BLUE, lighting=False)
    plotter.add_mesh(
        pv.Sphere(radius=r_marker * 0.72, center=(cx, cy, zmin)), color=COLOR_BED_BLUE, lighting=False
    )
    plotter.add_mesh(
        pv.Line((cx, cy, cz), (cx, cy, zmin)), color=COLOR_BED_BLUE, line_width=2.6, lighting=False
    )

    fxmin, fxmax, fymin, fymax = report.support_footprint_bounds_xy
    if fxmax > fxmin and fymax > fymin:
        footprint_rect = pv.Rectangle(
            [(fxmin, fymin, zmin), (fxmax, fymin, zmin), (fxmax, fymax, zmin)]
        )
        fp_color = COLOR_BED_BLUE if report.stability_tip_margin_mm >= 3.0 else COLOR_WARN_AMBER
        plotter.add_mesh(
            footprint_rect,
            color=fp_color,
            style="wireframe",
            line_width=2.0,
            opacity=0.80,
            lighting=False,
        )

    _add_3d_som_markers(plotter, report)
    cut_title = (
        f"4. Cutaway A-A ({section_spec.axis_name}={section_spec.coord_mm:.1f}mm) "
        f"| Wall (<0.8 Red, 0.8-1.6 Amber) & CoM"
    )
    plotter.add_text(cut_title, position="upper_left", font_size=11, color=COLOR_EDGE_DARK)
    plotter_dyn.add_axes()
    _set_parallel_iso_camera(plotter, report, direction=(1.05, -1.15, 0.80))

    plotter.screenshot(str(out_path))
    plotter.close()
    crop_whitespace_png(out_path)
