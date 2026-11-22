"""2D vector Matplotlib renderer for Sheet 2 (`preview_engineering.png`)."""

from __future__ import annotations

from pathlib import Path

import build123d as b3d
import matplotlib.pyplot as plt
import numpy as np
from build123d import Part
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch

from preview_pipeline.drawing_2d import (
    draw_2d_som_markers,
    draw_ordinate_z_levels,
    draw_outer_dim_h,
    draw_outer_dim_v,
    face_to_compound_mpl_path,
    sample_2d_projected_edge,
)
from preview_pipeline.loader import crop_whitespace_png
from preview_pipeline.models import (
    COLOR_BED_BLUE,
    COLOR_BG_LIGHT,
    COLOR_EDGE_DARK,
    COLOR_GHOST_GRAY,
    COLOR_OK_GRAY,
    CadReport,
    SectionCutSpec,
)


def _render_clean_hlr_view(
    ax: Axes,
    shape: Part,
    report: CadReport,
    title: str,
    viewport_origin: tuple[float, float, float],
    viewport_up: tuple[float, float, float],
    look_at: tuple[float, float, float],
    u_label: str,
    v_label: str,
    u_offset: float = 0.0,
    v_offset: float = 0.0,
    section_spec: SectionCutSpec | None = None,
    show_z_levels: bool = False,
) -> None:
    """Render clean orthogonal 2D HLR projection with grid scale, outer dims, and SoM pins."""
    vis, hid = shape.project_to_viewport(
        viewport_origin=viewport_origin,
        viewport_up=viewport_up,
        look_at=look_at,
    )

    all_u: list[float] = []
    all_v: list[float] = []

    for e in hid:
        if e.length < 0.2:
            continue
        xs, ys = sample_2d_projected_edge(e, u_offset, v_offset)
        ax.plot(xs, ys, color=COLOR_OK_GRAY, lw=0.85, ls=(0, (4.0, 2.5)), alpha=0.62, zorder=2)

    for e in vis:
        if e.length < 0.05:
            continue
        xs, ys = sample_2d_projected_edge(e, u_offset, v_offset)
        all_u.extend(xs.tolist())
        all_v.extend(ys.tolist())
        ax.plot(xs, ys, color=COLOR_EDGE_DARK, lw=1.5, solid_capstyle="round", zorder=4)

    if not all_u or not all_v:
        return

    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    ui = axis_idx[u_label]
    vi = axis_idx[v_label]
    u_min = float(report.bbox_min[ui])
    u_max = float(report.bbox_max[ui])
    v_min = float(report.bbox_min[vi])
    v_max = float(report.bbox_max[vi])
    u_span = max(u_max - u_min, 1.0)
    v_span = max(v_max - v_min, 1.0)
    max_span = max(u_span, v_span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", color="#CBD5E1", lw=0.75, ls="-", zorder=0)
    ax.grid(True, which="minor", color="#E2E8F0", lw=0.45, ls=":", zorder=0)
    ax.minorticks_on()

    ax.set_title(
        title,
        fontsize=13.5,
        fontfamily="monospace",
        fontweight="bold",
        color=COLOR_EDGE_DARK,
        loc="left",
        pad=10,
    )

    ext = max_span * 0.05
    if u_min <= 0.0 <= u_max:
        ax.plot(
            [0, 0],
            [v_min - ext, v_max + ext],
            color=COLOR_GHOST_GRAY,
            lw=0.75,
            ls="dashdot",
            alpha=0.7,
            zorder=3,
        )
    if v_min <= 0.0 <= v_max:
        ax.plot(
            [u_min - ext, u_max + ext],
            [0, 0],
            color=COLOR_GHOST_GRAY,
            lw=0.75,
            ls="dashdot",
            alpha=0.7,
            zorder=3,
        )

    if section_spec is not None and u_label == "X" and v_label == "Y":
        cut_ext = max_span * 0.08
        if section_spec.axis_name == "X":
            xc = section_spec.coord_mm
            ax.plot(
                [xc, xc],
                [v_min - cut_ext, v_max + cut_ext],
                color=COLOR_BED_BLUE,
                lw=1.5,
                ls=(0, (6.0, 3.0)),
                zorder=7,
            )
            for y_end, va_align in [(v_max + cut_ext, "bottom"), (v_min - cut_ext, "top")]:
                ax.text(
                    xc,
                    y_end,
                    "A",
                    fontsize=10.0,
                    fontfamily="monospace",
                    fontweight="bold",
                    color="#FFFFFF",
                    ha="center",
                    va=va_align,
                    bbox={"boxstyle": "square,pad=0.18", "fc": COLOR_BED_BLUE, "ec": "none"},
                    zorder=8,
                )
        else:
            yc = section_spec.coord_mm
            ax.plot(
                [u_min - cut_ext, u_max + cut_ext],
                [yc, yc],
                color=COLOR_BED_BLUE,
                lw=1.5,
                ls=(0, (6.0, 3.0)),
                zorder=7,
            )
            for x_end, ha_align in [(u_min - cut_ext, "right"), (u_max + cut_ext, "left")]:
                ax.text(
                    x_end,
                    yc,
                    "A",
                    fontsize=10.0,
                    fontfamily="monospace",
                    fontweight="bold",
                    color="#FFFFFF",
                    ha=ha_align,
                    va="center",
                    bbox={"boxstyle": "square,pad=0.18", "fc": COLOR_BED_BLUE, "ec": "none"},
                    zorder=8,
                )

    dim_offset = max_span * 0.13
    draw_outer_dim_h(ax, u_min, u_max, v_min, v_min - dim_offset, f"{u_span:.2f} mm ({u_label})")
    draw_outer_dim_v(ax, v_min, v_max, u_min, u_min - dim_offset, f"{v_span:.2f} mm ({v_label})")

    z_levels = report.features.get("horizontal_planar_z_levels_mm", [])
    if show_z_levels and v_label == "Z":
        draw_ordinate_z_levels(ax, u_max, v_min, v_max, max_span, z_levels)

    draw_2d_som_markers(ax, report, u_label=u_label, v_label=v_label)

    pad_u = max_span * 0.24
    pad_v = max_span * 0.24
    right_mult = 0.85 if (show_z_levels and v_label == "Z") else 0.65
    ax.set_xlim(u_min - pad_u, u_max + pad_u * right_mult)
    ax.set_ylim(v_min - pad_v, v_max + pad_v * 0.65)
    ax.set_xlabel(f"{u_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155")
    ax.set_ylabel(f"{v_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155")
    ax.tick_params(labelsize=10, colors="#334155")


def _render_2d_hatched_section_view(
    ax: Axes,
    shape: Part,
    report: CadReport,
    section_spec: SectionCutSpec,
) -> None:
    """Render Panel 2 of Sheet 2: True 2D Hatched B-Rep Cross-Section A-A on a [mm] grid."""
    cx = (report.bbox_min[0] + report.bbox_max[0]) / 2.0
    cy = (report.bbox_min[1] + report.bbox_max[1]) / 2.0
    cz = (report.bbox_min[2] + report.bbox_max[2]) / 2.0
    center = (cx, cy, cz)
    dist = max(report.bbox_size) * 5.0 + 100.0

    u_label = section_spec.u_label
    v_label = section_spec.v_label

    if section_spec.axis_name == "X":
        vp_origin = (cx + dist, cy, cz)
        u_offset, v_offset = cy, cz
        u_min_bb, u_max_bb = report.bbox_min[1], report.bbox_max[1]
    else:
        vp_origin = (cx, cy - dist, cz)
        u_offset, v_offset = cx, cz
        u_min_bb, u_max_bb = report.bbox_min[0], report.bbox_max[0]
    v_min_bb, v_max_bb = report.bbox_min[2], report.bbox_max[2]

    try:
        vis_bg, _ = shape.project_to_viewport(
            viewport_origin=vp_origin, viewport_up=(0, 0, 1), look_at=center
        )
        for e in vis_bg:
            if e.length < 0.1:
                continue
            xs, ys = sample_2d_projected_edge(e, u_offset, v_offset)
            ax.plot(
                xs,
                ys,
                color=COLOR_GHOST_GRAY,
                lw=0.75,
                ls=(0, (3.0, 2.5)),
                alpha=0.45,
                zorder=2,
            )
    except Exception:
        pass

    all_u: list[float] = [u_min_bb, u_max_bb]
    all_v: list[float] = [v_min_bb, v_max_bb]
    try:
        sec_shape = b3d.section(shape, section_by=section_spec.plane)
        sec_faces = sec_shape.faces() if hasattr(sec_shape, "faces") else []
        for f in sec_faces:
            cpath = face_to_compound_mpl_path(f, section_spec.axis_name)
            if cpath is not None:
                patch = PathPatch(
                    cpath,
                    facecolor="#BAE6FD",
                    edgecolor=COLOR_EDGE_DARK,
                    hatch="///",
                    lw=1.65,
                    alpha=0.88,
                    zorder=5,
                )
                ax.add_patch(patch)
                verts = np.asarray(cpath.vertices, dtype=float)
                all_u.extend(verts[:, 0].tolist())
                all_v.extend(verts[:, 1].tolist())
    except Exception:
        pass

    u_min, u_max = min(all_u), max(all_u)
    v_min, v_max = min(all_v), max(all_v)
    u_span = max(u_max - u_min, 1.0)
    v_span = max(v_max - v_min, 1.0)
    max_span = max(u_span, v_span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", color="#CBD5E1", lw=0.75, ls="-", zorder=0)
    ax.grid(True, which="minor", color="#E2E8F0", lw=0.45, ls=":", zorder=0)
    ax.minorticks_on()

    title = (
        f"2. SECTION A-A ({section_spec.axis_name} = {section_spec.coord_mm:.1f} mm) "
        f"— {section_spec.reason}"
    )
    ax.set_title(
        title,
        fontsize=13.0,
        fontfamily="monospace",
        fontweight="bold",
        color=COLOR_EDGE_DARK,
        loc="left",
        pad=10,
    )

    ext = max_span * 0.05
    if u_min <= 0.0 <= u_max:
        ax.plot(
            [0, 0],
            [v_min - ext, v_max + ext],
            color=COLOR_GHOST_GRAY,
            lw=0.75,
            ls="dashdot",
            alpha=0.7,
            zorder=3,
        )

    dim_offset = max_span * 0.13
    draw_outer_dim_h(ax, u_min, u_max, v_min, v_min - dim_offset, f"{u_span:.2f} mm ({u_label})")
    draw_outer_dim_v(ax, v_min, v_max, u_min, u_min - dim_offset, f"{v_span:.2f} mm ({v_label})")

    z_levels = report.features.get("horizontal_planar_z_levels_mm", [])
    draw_ordinate_z_levels(ax, u_max, v_min, v_max, max_span, z_levels)
    draw_2d_som_markers(ax, report, u_label=u_label, v_label=v_label, filter_section=section_spec)

    ax.text(
        0.50,
        0.96,
        "HATCHED (///): Cut Solid Material  |  DASHED: Background Silhouette",
        transform=ax.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        fontweight="bold",
        color=COLOR_EDGE_DARK,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.28", "fc": "#F8FAFC", "ec": "#CBD5E1", "lw": 0.9, "alpha": 0.92},
        zorder=8,
    )

    pad_u = max_span * 0.24
    pad_v = max_span * 0.24
    ax.set_xlim(u_min - pad_u, u_max + pad_u * 0.85)
    ax.set_ylim(v_min - pad_v, v_max + pad_v * 0.65)
    ax.set_xlabel(f"{u_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155")
    ax.set_ylabel(f"{v_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155")
    ax.tick_params(labelsize=10, colors="#334155")


def render_sheet2_2d_engineering(
    shape: Part,
    report: CadReport,
    section_spec: SectionCutSpec,
    out_path: Path,
) -> None:
    """Render Sheet 2 (`preview_engineering.png`): 2x2 2D Vector HLR & Hatched Section A-A."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cx = (report.bbox_min[0] + report.bbox_max[0]) / 2.0
    cy = (report.bbox_min[1] + report.bbox_max[1]) / 2.0
    cz = (report.bbox_min[2] + report.bbox_max[2]) / 2.0
    center = (cx, cy, cz)
    dist = max(report.bbox_size) * 5.0 + 100.0

    fig, axes = plt.subplots(2, 2, figsize=(18, 14.5), dpi=130, facecolor=COLOR_BG_LIGHT)
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.06, top=0.94, wspace=0.18, hspace=0.22)

    ax_top: Axes = axes[0, 0]
    ax_section: Axes = axes[0, 1]
    ax_front: Axes = axes[1, 0]
    ax_right: Axes = axes[1, 1]

    for ax in (ax_top, ax_section, ax_front, ax_right):
        ax.set_facecolor("#FFFFFF")

    _render_clean_hlr_view(
        ax_top,
        shape,
        report,
        "1. TOP HLR VIEW (XY Plane) — Cut Line A-A",
        viewport_origin=(cx, cy, cz + dist),
        viewport_up=(0, 1, 0),
        look_at=center,
        u_label="X",
        v_label="Y",
        u_offset=cx,
        v_offset=cy,
        section_spec=section_spec,
        show_z_levels=False,
    )
    _render_2d_hatched_section_view(ax_section, shape, report, section_spec=section_spec)
    _render_clean_hlr_view(
        ax_front,
        shape,
        report,
        "3. FRONT HLR VIEW (XZ Plane)",
        viewport_origin=(cx, cy - dist, cz),
        viewport_up=(0, 0, 1),
        look_at=center,
        u_label="X",
        v_label="Z",
        u_offset=cx,
        v_offset=cz,
        show_z_levels=True,
    )
    _render_clean_hlr_view(
        ax_right,
        shape,
        report,
        "4. RIGHT HLR VIEW (YZ Plane)",
        viewport_origin=(cx + dist, cy, cz),
        viewport_up=(0, 0, 1),
        look_at=center,
        u_label="Y",
        v_label="Z",
        u_offset=cy,
        v_offset=cz,
        show_z_levels=True,
    )

    fig.savefig(str(out_path), format="png", dpi=130)
    plt.close(fig)
    crop_whitespace_png(out_path, target_aspect=6.0 / 5.0)
