"""2D vector drafting primitives: edge sampling, dimension callouts, SoM pins, and B-Rep paths."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from build123d import Edge, Face, GeomType, Wire
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath

from preview_pipeline.models import (
    COLOR_BADGE_AMBER,
    COLOR_BADGE_RED,
    COLOR_BED_BLUE,
    CadReport,
    SectionCutSpec,
)


def sample_2d_projected_edge(
    edge: Edge, u_offset: float = 0.0, v_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    n = 2 if edge.geom_type == GeomType.LINE else max(24, min(72, int(edge.length / 0.35) + 2))
    pts = edge.positions([i / (n - 1) for i in range(n)])
    xs = np.array([p.X + u_offset for p in pts], dtype=float)
    ys = np.array([p.Y + v_offset for p in pts], dtype=float)
    return xs, ys


def draw_outer_dim_h(
    ax: Axes,
    x1: float,
    x2: float,
    y_ref: float,
    y_dim: float,
    label: str,
    color: str = COLOR_BED_BLUE,
    fontsize: float = 11.0,
) -> None:
    if abs(x2 - x1) < 1e-2:
        return
    sign = 1.0 if y_dim >= y_ref else -1.0
    overshoot = abs(y_dim - y_ref) * 0.16
    gap = abs(y_dim - y_ref) * 0.08
    ax.plot([x1, x1], [y_ref + sign * gap, y_dim + sign * overshoot], color=color, lw=0.9, alpha=0.85)
    ax.plot([x2, x2], [y_ref + sign * gap, y_dim + sign * overshoot], color=color, lw=0.9, alpha=0.85)
    ax.annotate(
        "",
        xy=(x1, y_dim),
        xytext=(x2, y_dim),
        arrowprops={"arrowstyle": "<|-|>", "color": color, "lw": 1.2, "mutation_scale": 11},
    )
    ax.text(
        (x1 + x2) / 2.0,
        y_dim + sign * overshoot * 0.35,
        label,
        color=color,
        fontsize=fontsize,
        fontfamily="monospace",
        fontweight="bold",
        ha="center",
        va="bottom" if sign > 0 else "top",
        bbox={"boxstyle": "square,pad=0.18", "fc": "#FFFFFF", "ec": "none", "alpha": 0.92},
    )


def draw_outer_dim_v(
    ax: Axes,
    y1: float,
    y2: float,
    x_ref: float,
    x_dim: float,
    label: str,
    color: str = COLOR_BED_BLUE,
    fontsize: float = 11.0,
) -> None:
    if abs(y2 - y1) < 1e-2:
        return
    sign = 1.0 if x_dim >= x_ref else -1.0
    overshoot = abs(x_dim - x_ref) * 0.16
    gap = abs(x_dim - x_ref) * 0.08
    ax.plot([x_ref + sign * gap, x_dim + sign * overshoot], [y1, y1], color=color, lw=0.9, alpha=0.85)
    ax.plot([x_ref + sign * gap, x_dim + sign * overshoot], [y2, y2], color=color, lw=0.9, alpha=0.85)
    ax.annotate(
        "",
        xy=(x_dim, y1),
        xytext=(x_dim, y2),
        arrowprops={"arrowstyle": "<|-|>", "color": color, "lw": 1.2, "mutation_scale": 11},
    )
    ax.text(
        x_dim + sign * overshoot * 0.35,
        (y1 + y2) / 2.0,
        label,
        color=color,
        fontsize=fontsize,
        fontfamily="monospace",
        fontweight="bold",
        ha="left" if sign > 0 else "right",
        va="center",
        rotation=90,
        bbox={"boxstyle": "square,pad=0.18", "fc": "#FFFFFF", "ec": "none", "alpha": 0.92},
    )


def draw_ordinate_z_levels(
    ax: Axes,
    u_max: float,
    v_min: float,
    v_max: float,
    max_span: float,
    z_levels: list[float],
) -> None:
    """Draw non-overlapping ordinate Z-step level callouts on the right margin of XZ/YZ views."""
    if not z_levels:
        return
    deduped: list[float] = []
    for z in sorted(z_levels):
        if not deduped or abs(z - deduped[-1]) >= 0.4:
            deduped.append(z)
        elif abs(z - v_max) < 1e-2:
            deduped[-1] = z

    min_gap = max(max_span * 0.065, (v_max - v_min) * 0.075, 1.0)
    label_ys = list(deduped)
    for i in range(1, len(label_ys)):
        if label_ys[i] - label_ys[i - 1] < min_gap:
            label_ys[i] = label_ys[i - 1] + min_gap

    x_tick_start = u_max + max_span * 0.02
    x_tick_mid = u_max + max_span * 0.045
    x_tick_end = u_max + max_span * 0.07
    for z, y_lbl in zip(deduped, label_ys, strict=True):
        ax.plot(
            [x_tick_start, x_tick_mid, x_tick_end],
            [z, z, y_lbl],
            color="#475569",
            lw=0.85,
            ls="-",
            alpha=0.85,
            zorder=5,
        )
        ax.text(
            x_tick_end + max_span * 0.01,
            y_lbl,
            f"Z={z:.1f}",
            fontsize=9.0,
            fontfamily="monospace",
            fontweight="bold",
            color="#334155",
            ha="left",
            va="center",
            bbox={"boxstyle": "square,pad=0.12", "fc": "#FFFFFF", "ec": "none", "alpha": 0.88},
            zorder=6,
        )


def draw_2d_som_markers(
    ax: Axes,
    report: CadReport,
    u_label: str,
    v_label: str,
    filter_section: SectionCutSpec | None = None,
) -> None:
    """Project Set-of-Mark [D1], [D2] defect pins onto a 2D (u, v) orthographic axis."""
    if not report.defect_markers:
        return
    axis_idx = {"X": 0, "Y": 1, "Z": 2}
    ui = axis_idx[u_label]
    vi = axis_idx[v_label]

    grouped: list[dict[str, Any]] = []
    for m in report.defect_markers:
        if filter_section is not None:
            sec_idx = axis_idx[filter_section.axis_name]
            if abs(m.centroid_mm[sec_idx] - filter_section.coord_mm) > 15.0:
                continue
        u_pt = m.centroid_mm[ui]
        v_pt = m.centroid_mm[vi]
        is_crit = "critical" in m.category or m.severity == "FAIL"
        merged = False
        for g in grouped:
            if math.hypot(u_pt - g["u"], v_pt - g["v"]) < 2.0:
                g["ids"].append(m.id)
                g["is_crit"] = g["is_crit"] or is_crit
                merged = True
                break
        if not merged:
            grouped.append({"u": u_pt, "v": v_pt, "ids": [m.id], "is_crit": is_crit})

    for idx_g, g in enumerate(grouped):
        badge_color = COLOR_BADGE_RED if g["is_crit"] else COLOR_BADGE_AMBER
        label_str = ",".join(g["ids"])
        ax.plot(
            g["u"],
            g["v"],
            marker="o",
            markersize=7.5,
            markerfacecolor="none",
            markeredgecolor=badge_color,
            markeredgewidth=2.0,
            zorder=9,
        )
        offset_y = 9 if (idx_g % 2 == 0) else -15
        ax.annotate(
            label_str,
            xy=(g["u"], g["v"]),
            xytext=(8, offset_y),
            textcoords="offset points",
            fontsize=9.5,
            fontfamily="monospace",
            fontweight="bold",
            color="#FFFFFF",
            bbox={"boxstyle": "round,pad=0.22", "fc": badge_color, "ec": "#FFFFFF", "lw": 0.9},
            zorder=10,
        )


def _wire_to_ordered_2d_polygon(wire: Wire, axis_name: str) -> np.ndarray | None:
    edge_chains: list[np.ndarray] = []
    for e in wire.edges():
        if e.length < 1e-4:
            continue
        n = 2 if e.geom_type == GeomType.LINE else max(18, min(64, int(e.length / 0.3) + 2))
        pts = e.positions([i / (n - 1) for i in range(n)])
        if axis_name == "X":
            arr = np.array([[p.Y, p.Z] for p in pts], dtype=float)
        else:
            arr = np.array([[p.X, p.Z] for p in pts], dtype=float)
        edge_chains.append(arr)

    if not edge_chains:
        return None

    ordered = [edge_chains.pop(0)]
    while edge_chains:
        curr_end = ordered[-1][-1]
        best_idx = 0
        best_dist = float("inf")
        flip = False
        for i, cand in enumerate(edge_chains):
            d_start = float(np.linalg.norm(cand[0] - curr_end))
            d_end = float(np.linalg.norm(cand[-1] - curr_end))
            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                flip = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                flip = True
        chosen = edge_chains.pop(best_idx)
        if flip:
            chosen = chosen[::-1]
        ordered.append(chosen[1:] if len(chosen) > 1 else chosen)

    poly = np.vstack(ordered)
    if len(poly) < 3:
        return None
    return poly


def _signed_area_2d(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def face_to_compound_mpl_path(face: Face, axis_name: str) -> MplPath | None:
    """Convert a planar B-Rep Face (with optional inner holes) into a compound Matplotlib Path."""
    outer_poly = _wire_to_ordered_2d_polygon(face.outer_wire(), axis_name)
    if outer_poly is None:
        return None
    if _signed_area_2d(outer_poly) < 0:
        outer_poly = outer_poly[::-1]

    rings = [outer_poly]
    for iw in face.inner_wires():
        inner_poly = _wire_to_ordered_2d_polygon(iw, axis_name)
        if inner_poly is not None:
            if _signed_area_2d(inner_poly) > 0:
                inner_poly = inner_poly[::-1]
            rings.append(inner_poly)

    verts_list: list[np.ndarray] = []
    codes_list: list[int] = []
    for r in rings:
        closed = np.vstack([r, r[0]])
        verts_list.append(closed)
        codes_list.append(int(MplPath.MOVETO))
        codes_list.extend([int(MplPath.LINETO)] * (len(r) - 1))
        codes_list.append(int(MplPath.CLOSEPOLY))

    return MplPath(np.vstack(verts_list), codes_list)
