"""SOTA Visual & Symbolic CAD + FDM DFM Inspection Pipeline for build123d.

TL;DR:
  Run `uv run preview.py <script.py> [--slice]` to always generate 3 files in `.preview/`:
  1. `report.json`            — Exact dimensions, PASS/WARN/FAIL checks & `[D1]` defect coords (never OCR renders).
  2. `preview.png`            — 2x2 parallel-projection 3D views (360° CAD+DFM, underside bed/overhangs, 4-in-1 cutaway).
  3. `preview_engineering.png`— 2x2 2D vector blueprint on a `[mm]` grid (Top/Front/Right HLR + hatched section `A-A`).
  Colors: Steel-Blue/Gray = Safe | Deep Blue = Bed/Cut A-A | Amber = Warn (45-60° / 0.8-1.6mm) | Red = Fail (>60° / <0.8mm).

===============================================================================
RESEARCH & ARCHITECTURAL MANIFESTO (HARD RULES FOR HUMAN + VLM COLLABORATION)
===============================================================================
This pipeline implements a neuro-symbolic visual feedback loop grounded in
recent Vision-Language Model (VLM) for CAD research (CADCodeVerify [ICLR 2025],
CADFusion [ICML 2025], Seek-CAD [2025], Chat3D, and Set-of-Mark [CVPR]) and
industrial FDM Design-for-Manufacturing (DFM) engineering practice:

1. NEURO-SYMBOLIC SEPARATION (ZERO-OCR POLICY):
   VLMs excel at topological, spatial, and qualitative reasoning but frequently
   hallucinate when performing OCR on dense, tiny 3D dimension callouts. All
   exact dimensions, tolerances, feature catalogs, G-code telemetry, and
   deterministic PASS/WARN/FAIL checks are written to `report.json`. Rendered
   images contain only high-contrast panel headers, outer bounding dimensions,
   and alphanumeric defect anchors.

2. FLAT, ZERO-TIER 2-SHEET ARCHITECTURE (NO CONDITIONAL MISSING VIEWS):
   Every execution deterministically produces the exact same 3 artifacts:
     - `report.json`: Symbolic ground truth, DFM checks, and `[D1..Dn]` defect coordinates.
     - `preview.png` (2x2 3D Spatial, DFM & Smart Cutaway Sheet).
     - `preview_engineering.png` (2x2 2D Vector Orthographic HLR & Hatched Section Sheet).
   Eliminating conditional Tier-1/Tier-2 splits prevents VLM context-window bloat
   while guaranteeing that internal cross-sections and wall-thickness maps are
   always available—even when simple numeric checks return PASS.

3. STRICT PARALLEL PROJECTION (`enable_parallel_projection()`):
   Perspective foreshortening distorts parallel edges, hides draft-angle errors,
   and makes symmetric towers appear unequal in height. Every 3D viewport uses
   true orthographic/axonometric parallel projection so 3D parallel lines remain
   strictly parallel in 2D pixel space.

4. COMPLETE 360-DEGREE OCTANT COVERAGE (ZERO SELF-OCCLUSION BLIND SPOTS):
   Single isometric renders hide up to 50% of a part's exterior. Sheet 1 pairs
   two diametrically opposite upper isometric octants (`+X -Y +Z` and `-X +Y +Z`)
   with a dedicated underside octant (`-Z`, dynamically oriented toward the
   dominant overhang cluster) to expose 100% of external and downward surfaces.

5. UNIFIED CAD SHADING + DISCRETE TRAFFIC-LIGHT DFM PALETTE + B-REP EDGES:
   Rather than duplicating views for "plain CAD" vs "DFM overhang analysis", all
   exterior 3D views combine smooth directional surface shading, explicit dark
   B-Rep topological edge curves (`#0F172A`, polygon-offset), and a discrete
   4-state semantic color palette:
     - Safe Surface (<= 45 deg): Steel-Slate CAD Blue (`#5B7C99`)
     - Build Plate Contact (Z = Z_min): Deep Cyan-Blue (`#0284C7`)
     - Moderate Overhang (45 - 60 deg): Warning Amber (`#F59E0B`)
     - Critical Overhang / Bridge (> 60 deg): Critical Crimson Red (`#EF4444`)

6. 4-IN-1 SMART 3D CUTAWAY (WALL THICKNESS + SECTION CAP + GHOST + CoM):
   Slicing a 3D mesh in half without context leaves a floating fragment in a
   void. Panel 4 of `preview.png` combines:
     (a) The kept solid half colored by fixed absolute FDM wall-thickness thresholds
         (< 0.8 mm Red [<2 perimeters], 0.8-1.6 mm Amber [2-4 perimeters], > 1.6 mm Gray),
     (b) A high-contrast B-Rep section cap (`#38BDF8` fill + thick `#0F172A` contour),
     (c) The cut-away half rendered as a translucent ghost wireframe (`opacity=0.15`)
         to preserve full whole-part spatial grounding, and
     (d) The Center-of-Mass (CoM) sphere, vertical gravity plumb line, and bed
         tipping-stability footprint.

7. PURE 2D VECTOR BLUEPRINT WITH 45-DEG HATCHED METRIC SECTION (`A-A`):
   Never paste 3D raster screenshots into a 2D engineering drawing. Sheet 2
   (`preview_engineering.png`) is a 100% vector Matplotlib technical blueprint
   with ISO-aligned Top (XY), Front (XZ), and Right (YZ) Hidden-Line-Removal
   (HLR) projections plus a true planar B-Rep Cross-Section (`A-A`) rendered
   with 45-degree technical hatching (`///`) on a millimeter coordinate grid.
   Solid material is hatched; internal pockets and holes remain white over the
   grid so internal clearances and wall thicknesses can be read directly in mm.

8. SET-OF-MARK (SoM) VISUAL-TO-JSON DEFECT ANCHORING (`[D1]`, `[D2]`):
   To bridge symbolic coordinates and visual pixels, connected defect clusters
   are assigned stable IDs (`D1`, `D2`, ...) in `report.json` and pinned with
   matching high-contrast `[D1]` callout badges in both 3D (`preview.png`) and
   2D (`preview_engineering.png`).

9. CLUSTER-FILTERED OPPOSING-WALL RAY-CASTING (`dot < -0.75`, area >= 4 mm^2):
   Naive inward ray-casting flags sharp corners and 45-degree build-plate
   chamfers as fake `< 0.2 mm` thin walls. Filtering hits by strict antiparallel
   surface normals (`dot(n_entry, n_hit) < -0.75`) and connected-component
   surface area eliminates tessellation noise and ensures section planes cut
   through genuine structural defects.

10. LOSSLESS FRAMING & WHITESPACE CROPPING:
    Each 2x2 sheet renders at high native resolution (`1800x1500` px) and is
    losslessly trimmed of outer margins to maximize VLM visual token density.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import build123d as b3d
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvista_cad
import vtk
from build123d import (
    Axis,
    Builder,
    Edge,
    Face,
    GeomType,
    Plane,
    Shape,
    Wire,
    export_stl,
)
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageChops

# =============================================================================
# GLOBAL CONSTANTS, SEMANTIC PALETTE & FDM THRESHOLDS
# =============================================================================
PRINTER_NAME = "Bambu Lab P1S"
BUILD_VOL_MM = (256.0, 256.0, 256.0)

# Global semantic color palette
COLOR_SAFE_CAD = "#5B7C99"  # Safe <= 45 deg overhang / normal CAD surface
COLOR_OK_GRAY = "#64748B"  # Safe > 1.6 mm wall thickness
COLOR_BED_BLUE = "#0284C7"  # Z = Z_min build-plate contact & primary dimension accents
COLOR_CAP_CYAN = "#38BDF8"  # Cross-section cut cap fill
COLOR_WARN_AMBER = "#F59E0B"  # Warning: 45-60 deg overhang or 0.8-1.6 mm wall
COLOR_CRIT_RED = "#EF4444"  # Critical: > 60 deg overhang/bridge or < 0.8 mm wall
COLOR_BADGE_RED = "#DC2626"  # SoM critical badge background
COLOR_BADGE_AMBER = "#D97706"  # SoM warning badge background
COLOR_EDGE_DARK = "#0F172A"  # Crisp B-Rep topological edges
COLOR_GHOST_GRAY = "#94A3B8"  # Ghost cutaway context & centerlines
COLOR_BG_LIGHT = "#F8FAFC"  # High-contrast clean sheet background

# Absolute FDM Wall Thickness Thresholds [mm] (for 0.4 mm nozzle)
WALL_CRIT_MM = 0.8  # < 0.8 mm (< 2 nozzle perimeters) -> RED
WALL_WARN_MM = 1.6  # 0.8 - 1.6 mm (2 to 4 perimeters) -> AMBER, > 1.6 mm -> OK GRAY

# Minimum cluster surface area [mm^2] to suppress single-triangle corner/tessellation noise
MIN_WALL_CLUSTER_MM2 = 4.0
MIN_OVERHANG_CLUSTER_MM2 = 6.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class CheckResult:
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefectMarker:
    id: str  # "D1", "D2", ...
    category: str  # "thin_wall_critical", "overhang_critical", etc.
    severity: str  # "FAIL" or "WARN"
    short_label: str  # Concise label for 3D Set-of-Mark callout badge
    centroid_mm: tuple[float, float, float]
    area_mm2: float
    area_cm2: float
    dominant_normal: tuple[float, float, float]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "short_label": self.short_label,
            "centroid_mm": [round(x, 2) for x in self.centroid_mm],
            "area_mm2": round(self.area_mm2, 2),
            "area_cm2": round(self.area_cm2, 3),
            "dominant_normal": [round(n, 3) for n in self.dominant_normal],
            "description": self.description,
        }


@dataclass
class SectionCutSpec:
    axis_name: str  # "X" (YZ plane) or "Y" (XZ plane)
    coord_mm: float  # Cut coordinate along axis_name
    reason: str  # e.g. "Defect D1 Focus" or "Center Symmetry / Internal Bore"
    plane: Plane  # build123d Plane for exact B-Rep sectioning
    normal: tuple[float, float, float]
    origin: tuple[float, float, float]
    u_label: str  # 2D horizontal axis name ("Y" or "X")
    v_label: str  # 2D vertical axis name ("Z")


@dataclass
class CadReport:
    name: str
    script_path: str
    timestamp: str
    overall_status: str
    # Geometry & B-Rep
    valid_brep: bool
    solid_count: int
    face_count: int
    edge_count: int
    volume_mm3: float
    volume_cm3: float
    surface_area_mm2: float
    surface_area_cm2: float
    # Bounding Box & Build Volume
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    build_volume_mm: tuple[float, float, float]
    build_volume_margin_mm: tuple[float, float, float]
    build_volume_occupancy_pct: tuple[float, float, float]
    near_or_exceeds_build_volume: bool
    # Bed Contact & Z=0 Audit
    z_min_mm: float
    bed_contact_area_mm2: float
    bed_contact_area_cm2: float
    bed_flatness_ok: bool
    bottom_fillet_faces_count: int
    bottom_fillet_radii_mm: list[float]
    # Overhangs
    overhang_amber_45_60_mm2: float
    overhang_amber_45_60_cm2: float
    overhang_red_gt_60_mm2: float
    overhang_red_gt_60_cm2: float
    # Wall Thickness
    wall_min_mm: float
    wall_p5_mm: float
    wall_p10_mm: float
    wall_p50_mm: float
    wall_p90_mm: float
    wall_max_mm: float
    wall_red_lt_08_pct: float
    wall_amber_08_16_pct: float
    wall_ok_gt_16_pct: float
    # Stability & Center of Mass
    center_of_mass_mm: tuple[float, float, float]
    support_footprint_bounds_xy: tuple[float, float, float, float]
    stability_tip_margin_mm: float
    # Set-of-Mark Defect Markers & Section Cut Metadata
    defect_markers: list[DefectMarker]
    section_cut: dict[str, Any]
    # Feature Catalog & Deterministic Checks
    features: dict[str, Any]
    checks: dict[str, CheckResult]
    # Optional Slicer CLI Telemetry
    slicer: dict[str, Any] | None = None
    generated_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_name": self.name,
            "script_path": self.script_path,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "checks": {
                k: {
                    "status": v.status,
                    "message": v.message,
                    "metrics": v.metrics,
                }
                for k, v in self.checks.items()
            },
            "defect_markers": [m.to_dict() for m in self.defect_markers],
            "section_cut_a_a": self.section_cut,
            "geometry": {
                "valid_brep": self.valid_brep,
                "solid_count": self.solid_count,
                "face_count": self.face_count,
                "edge_count": self.edge_count,
                "volume_mm3": round(self.volume_mm3, 2),
                "volume_cm3": round(self.volume_cm3, 3),
                "surface_area_mm2": round(self.surface_area_mm2, 2),
                "surface_area_cm2": round(self.surface_area_cm2, 3),
            },
            "bounding_box": {
                "min_mm": [round(x, 3) for x in self.bbox_min],
                "max_mm": [round(x, 3) for x in self.bbox_max],
                "size_mm": [round(x, 3) for x in self.bbox_size],
                "printer_name": PRINTER_NAME,
                "printer_build_volume_mm": list(self.build_volume_mm),
                "margin_to_limits_mm": [
                    round(x, 2) for x in self.build_volume_margin_mm
                ],
                "occupancy_pct": [round(x, 1) for x in self.build_volume_occupancy_pct],
                "near_or_exceeds_limits": self.near_or_exceeds_build_volume,
            },
            "bed_adhesion": {
                "z_min_mm": round(self.z_min_mm, 3),
                "bed_flatness_at_z0": self.bed_flatness_ok,
                "contact_area_mm2": round(self.bed_contact_area_mm2, 2),
                "contact_area_cm2": round(self.bed_contact_area_cm2, 3),
                "fillets_touching_build_plate_count": self.bottom_fillet_faces_count,
                "fillets_touching_build_plate_radii_mm": self.bottom_fillet_radii_mm,
            },
            "overhangs": {
                "moderate_45_to_60_deg_mm2": round(self.overhang_amber_45_60_mm2, 2),
                "moderate_45_to_60_deg_cm2": round(self.overhang_amber_45_60_cm2, 3),
                "critical_gt_60_deg_mm2": round(self.overhang_red_gt_60_mm2, 2),
                "critical_gt_60_deg_cm2": round(self.overhang_red_gt_60_cm2, 3),
            },
            "wall_thickness": {
                "thresholds_mm": {
                    "critical_red_below": WALL_CRIT_MM,
                    "warning_amber_below": WALL_WARN_MM,
                },
                "min_mm": round(self.wall_min_mm, 3),
                "percentiles_mm": {
                    "p5": round(self.wall_p5_mm, 3),
                    "p10": round(self.wall_p10_mm, 3),
                    "p50_median": round(self.wall_p50_mm, 3),
                    "p90": round(self.wall_p90_mm, 3),
                    "max": round(self.wall_max_mm, 3),
                },
                "distribution_area_pct": {
                    "red_lt_0.8mm": round(self.wall_red_lt_08_pct, 2),
                    "amber_0.8_to_1.6mm": round(self.wall_amber_08_16_pct, 2),
                    "ok_gt_1.6mm": round(self.wall_ok_gt_16_pct, 2),
                },
            },
            "stability": {
                "center_of_mass_mm": [round(x, 3) for x in self.center_of_mass_mm],
                "footprint_bounds_xy_mm": [
                    round(x, 2) for x in self.support_footprint_bounds_xy
                ],
                "min_tipping_margin_mm": round(self.stability_tip_margin_mm, 2),
            },
            "features": self.features,
            "slicer": self.slicer,
            "generated_files": self.generated_files,
        }


# =============================================================================
# MODEL LOADING & IMAGE POST-PROCESSING
# =============================================================================
def _load_shape(script_path: Path, var_name: str | None) -> Shape:
    """Execute a build123d script with viewer and file-export calls stubbed out."""
    import ocp_vscode

    shown: list[Shape | Builder] = []

    def capture_show(*args: Any, **kwargs: Any) -> None:
        for arg in args:
            if isinstance(arg, (Shape, Builder)):
                shown.append(arg)

    orig_show = getattr(ocp_vscode, "show", None)
    orig_show_object = getattr(ocp_vscode, "show_object", None)
    orig_show_all = getattr(ocp_vscode, "show_all", None)
    orig_export_step = getattr(b3d, "export_step", None)
    orig_export_stl = getattr(b3d, "export_stl", None)
    orig_cwd = Path.cwd()

    ocp_vscode.show = capture_show
    ocp_vscode.show_object = capture_show
    ocp_vscode.show_all = lambda *args, **kwargs: None
    b3d.export_step = lambda *args, **kwargs: None
    b3d.export_stl = lambda *args, **kwargs: None

    resolved_script = script_path.resolve()
    try:
        os.chdir(resolved_script.parent)
        ns = runpy.run_path(str(resolved_script), run_name="__main__")
    finally:
        os.chdir(orig_cwd)
        if orig_show is not None:
            ocp_vscode.show = orig_show
        if orig_show_object is not None:
            ocp_vscode.show_object = orig_show_object
        if orig_show_all is not None:
            ocp_vscode.show_all = orig_show_all
        if orig_export_step is not None:
            b3d.export_step = orig_export_step
        if orig_export_stl is not None:
            b3d.export_stl = orig_export_stl

    if var_name:
        if var_name not in ns:
            raise KeyError(f"Variable {var_name!r} not found in {script_path}")
        candidate = ns[var_name]
    elif shown:
        candidate = shown[-1]
    else:
        candidates = [
            v
            for k, v in ns.items()
            if not k.startswith("_") and isinstance(v, (Shape, Builder))
        ]
        if not candidates:
            raise ValueError(f"No build123d Shape or Builder found in {script_path}")
        candidate = candidates[-1]

    if isinstance(candidate, Builder):
        return candidate._obj  # type: ignore[return-value]
    return candidate


def _crop_whitespace_png(image_path: Path, pad_px: int = 18) -> None:
    """Losslessly trim excess outer background margins to maximize VLM pixel density."""
    if not image_path.exists():
        return
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
        diff = ImageChops.difference(rgb, bg)
        bbox = diff.getbbox()
        if not bbox:
            return
        left = max(0, bbox[0] - pad_px)
        top = max(0, bbox[1] - pad_px)
        right = min(rgb.width, bbox[2] + pad_px)
        bottom = min(rgb.height, bbox[3] + pad_px)
        cropped = img.crop((left, top, right, bottom))
        cropped.save(image_path, format="PNG", optimize=True)


# =============================================================================
# GEOMETRY, CLUSTERED DEFECT DETECTION & SMART CUT-PLANE SELECTION
# =============================================================================
def _extract_spatial_clusters(
    mask: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    cluster_radius_mm: float,
    min_area_mm2: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Group flagged mesh cells into connected spatial clusters and filter micro-noise.

    Returns:
      - list of cluster dicts sorted descending by surface area
      - filtered boolean mask containing only cells belonging to valid clusters
    """
    idx = np.where(mask)[0]
    filtered_mask = np.zeros_like(mask, dtype=bool)
    if len(idx) == 0:
        return [], filtered_mask

    pts = centers[idx]
    visited = np.zeros(len(idx), dtype=bool)
    clusters: list[dict[str, Any]] = []
    r2 = cluster_radius_mm * cluster_radius_mm

    for i in range(len(idx)):
        if visited[i]:
            continue
        visited[i] = True
        queue = [i]
        comp_local = [i]
        while queue:
            curr = queue.pop()
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            diffs = pts[unvisited] - pts[curr]
            d2 = np.sum(diffs * diffs, axis=1)
            neighbors = unvisited[d2 <= r2]
            if len(neighbors) > 0:
                visited[neighbors] = True
                queue.extend(neighbors.tolist())
                comp_local.extend(neighbors.tolist())

        comp_cells = idx[np.array(comp_local, dtype=int)]
        comp_area = float(np.sum(areas[comp_cells]))
        if comp_area >= min_area_mm2:
            filtered_mask[comp_cells] = True
            weights = np.maximum(areas[comp_cells], 1e-6)
            centroid = np.average(centers[comp_cells], axis=0, weights=weights)
            mean_norm = np.average(normals[comp_cells], axis=0, weights=weights)
            n_len = float(np.linalg.norm(mean_norm))
            if n_len > 1e-6:
                mean_norm = mean_norm / n_len
            clusters.append(
                {
                    "cell_indices": comp_cells,
                    "area_mm2": comp_area,
                    "area_cm2": comp_area / 100.0,
                    "centroid_mm": (
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    "normal": (
                        float(mean_norm[0]),
                        float(mean_norm[1]),
                        float(mean_norm[2]),
                    ),
                }
            )

    # Sort primarily by area (rounded to 0.5 mm^2 so symmetric twins tie),
    # breaking ties in favor of the (+X, -Y, +Z) primary camera octant.
    clusters.sort(
        key=lambda c: (
            round(c["area_mm2"] * 2.0) / 2.0,
            c["centroid_mm"][0] - c["centroid_mm"][1] + c["centroid_mm"][2],
        ),
        reverse=True,
    )
    return clusters, filtered_mask


def _compute_wall_thickness(
    poly: pv.PolyData,
    max_ray: float,
    cluster_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true opposing-wall thickness [mm] via inward VTK OBBTree ray-casting.

    Enforces Rule 9:
      - Strict antiparallel normal threshold `dot(n_entry, n_hit) < -0.75` (>138.6 deg)
        so rays grazing 45-degree build-plate chamfers or 90-degree corners are rejected.
      - Connected-component area filtering (`>= MIN_WALL_CLUSTER_MM2`) so isolated
        corner/fillet tessellation slivers do not create fake 0.15 mm thin-wall alerts.
    """
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(poly)
    obb.BuildLocator()

    centers = poly.cell_centers().points
    normals = poly.cell_data["Normals"]
    areas = poly.cell_data["Area"]
    n_cells = len(centers)
    thickness = np.full(n_cells, np.nan, dtype=float)

    p_hit = vtk.vtkPoints()
    c_ids = vtk.vtkIdList()
    eps = 0.08

    for i in range(n_cells):
        n_vec = normals[i]
        p0 = centers[i] - n_vec * eps
        p1 = centers[i] - n_vec * max_ray
        obb.IntersectWithLine(p0, p1, p_hit, c_ids)
        n_hits = p_hit.GetNumberOfPoints()
        for h_idx in range(n_hits):
            hit_cell_id = int(c_ids.GetId(h_idx))
            n_hit = normals[hit_cell_id]
            # True opposing structural wall faces nearly opposite (> 138.6 deg)
            if float(np.dot(n_vec, n_hit)) < -0.75:
                hp = np.array(p_hit.GetPoint(h_idx))
                dist = float(np.linalg.norm(hp - centers[i]))
                if dist > 0.22:
                    thickness[i] = dist
                    break

    valid_mask = ~np.isnan(thickness)
    valid_vals = thickness[valid_mask]
    fallback = float(np.median(valid_vals)) if len(valid_vals) > 0 else 10.0
    thickness_filled = np.where(valid_mask, thickness, fallback)

    # Filter out isolated corner tessellation slivers below WALL_WARN_MM
    raw_thin_mask = thickness_filled < WALL_WARN_MM
    if np.any(raw_thin_mask):
        _, valid_thin_mask = _extract_spatial_clusters(
            raw_thin_mask,
            centers,
            areas,
            normals,
            cluster_radius_mm=cluster_radius_mm,
            min_area_mm2=MIN_WALL_CLUSTER_MM2,
        )
        noise_slivers = raw_thin_mask & (~valid_thin_mask)
        if np.any(noise_slivers):
            thickness_filled[noise_slivers] = fallback
            valid_mask[noise_slivers] = False

    return thickness_filled, valid_mask


def _detect_bottom_fillets(shape: Shape, zmin: float) -> tuple[int, list[float]]:
    """Detect curved fillet faces touching the build plate at Z = zmin (chamfers preferred for FDM)."""
    bottom_fillet_count = 0
    radii: set[float] = set()
    tol = 0.08

    for f in shape.faces():
        f_bb = f.bounding_box()
        if abs(f_bb.min.Z - zmin) > tol:
            continue
        height_z = f_bb.max.Z - f_bb.min.Z
        if 0.15 <= height_z <= 15.0 and f.geom_type in (
            GeomType.CYLINDER,
            GeomType.TORUS,
            GeomType.BSPLINE,
        ):
            try:
                n_vec = f.normal_at(f.position_at(0.5, 0.5))
                if n_vec.Z < -0.15:
                    bottom_fillet_count += 1
                    if hasattr(f, "radius") and f.radius is not None:
                        radii.add(round(float(f.radius), 2))
                    else:
                        radii.add(round(height_z, 2))
            except Exception:
                pass

    return bottom_fillet_count, sorted(radii)


def _extract_feature_catalog(shape: Shape) -> dict[str, Any]:
    """Extract cylindrical hole/shaft diameters, fillet radii, and planar Z levels for report.json."""
    cyl_holes_diameters: dict[float, int] = {}
    fillet_radii: dict[float, int] = {}

    for f in shape.faces():
        if (
            f.geom_type == GeomType.CYLINDER
            and hasattr(f, "radius")
            and f.radius is not None
        ):
            r = round(float(f.radius), 2)
            circ_edges = [e for e in f.edges() if e.geom_type == GeomType.CIRCLE]
            is_full_circle = any(
                abs(e.length - 2.0 * math.pi * r) < 0.25 for e in circ_edges
            )
            if is_full_circle:
                d = round(2.0 * r, 2)
                cyl_holes_diameters[d] = cyl_holes_diameters.get(d, 0) + 1
            else:
                fillet_radii[r] = fillet_radii.get(r, 0) + 1

    edge_radii: set[float] = set()
    for e in shape.edges():
        if (
            e.geom_type == GeomType.CIRCLE
            and hasattr(e, "radius")
            and e.radius is not None
        ):
            edge_radii.add(round(float(e.radius), 2))

    z_levels: set[float] = set()
    for f in shape.faces():
        if f.geom_type == GeomType.PLANE:
            f_bb = f.bounding_box()
            if abs(f_bb.max.Z - f_bb.min.Z) < 1e-2:
                z_levels.add(round(float(f_bb.min.Z), 2))

    return {
        "cylindrical_diameters_mm": [
            {"diameter_mm": d, "face_count": c}
            for d, c in sorted(cyl_holes_diameters.items())
        ],
        "fillet_and_arc_radii_mm": [
            {"radius_mm": r, "face_count": c} for r, c in sorted(fillet_radii.items())
        ],
        "distinct_circular_edge_radii_mm": sorted(edge_radii),
        "horizontal_planar_z_levels_mm": sorted(z_levels),
    }


def _select_smart_section_cut(
    shape: Shape,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    defect_markers: list[DefectMarker],
) -> SectionCutSpec:
    """Choose the optimal vertical section plane `A-A` for both 3D Cutaway and 2D Hatched Section.

    Priority 1: Slice directly through the primary defect marker `D1` along the plane
                that best exposes its cross-sectional profile.
    Priority 2: Slice through the primary internal feature plane / axis of symmetry,
                automatically stepping off center if exact center is an open air gap.
    """
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    sx = max(xmax - xmin, 1.0)
    sy = max(ymax - ymin, 1.0)

    if defect_markers:
        d1 = defect_markers[0]
        dx, dy, _ = d1.centroid_mm
        nx, ny, _ = d1.dominant_normal
        # Clamp within inner 6%..94% of bounding box so the section always slices solid material
        dx_clamped = float(np.clip(dx, xmin + 0.06 * sx, xmax - 0.06 * sx))
        dy_clamped = float(np.clip(dy, ymin + 0.06 * sy, ymax - 0.06 * sy))

        # If normal points primarily in Y (or defect is offset along X, like tower cutouts),
        # a YZ section plane at X = dx exposes the YZ profile of the defect.
        use_x_cut = (abs(ny) >= abs(nx)) and (abs(dx - cx) >= 0.08 * sx)
        if use_x_cut:
            return SectionCutSpec(
                axis_name="X",
                coord_mm=dx_clamped,
                reason=f"Defect {d1.id} ({d1.short_label})",
                plane=Plane.YZ.offset(dx_clamped),
                normal=(1.0, 0.0, 0.0),
                origin=(dx_clamped, cy, cz),
                u_label="Y",
                v_label="Z",
            )
        return SectionCutSpec(
            axis_name="Y",
            coord_mm=dy_clamped,
            reason=f"Defect {d1.id} ({d1.short_label})",
            plane=Plane.XZ.offset(dy_clamped),
            normal=(0.0, 1.0, 0.0),
            origin=(cx, dy_clamped, cz),
            u_label="X",
            v_label="Z",
        )

    # No defects (PASS): slice along the longer horizontal dimension to reveal internal bores/ribs
    if sx >= sy:
        # XZ section plane at Y = y_cut
        y_cut = cy
        reason = (
            "Center Plane (Y = 0)" if abs(cy) < 1e-2 else f"Mid-Plane (Y = {cy:.1f})"
        )
        try:
            sec = b3d.section(shape, section_by=Plane.XZ.offset(y_cut))
            if not sec.faces() or float(sec.area) < 1.0:
                y_cut = cy - 0.22 * sy
                reason = f"Feature Plane (Y = {y_cut:.1f})"
        except Exception:
            pass
        return SectionCutSpec(
            axis_name="Y",
            coord_mm=float(y_cut),
            reason=reason,
            plane=Plane.XZ.offset(float(y_cut)),
            normal=(0.0, 1.0, 0.0),
            origin=(cx, float(y_cut), cz),
            u_label="X",
            v_label="Z",
        )
    else:
        # YZ section plane at X = x_cut
        x_cut = cx
        reason = (
            "Center Plane (X = 0)" if abs(cx) < 1e-2 else f"Mid-Plane (X = {cx:.1f})"
        )
        try:
            sec = b3d.section(shape, section_by=Plane.YZ.offset(x_cut))
            if not sec.faces() or float(sec.area) < 1.0:
                x_cut = cx + 0.22 * sx
                reason = f"Feature Plane (X = {x_cut:.1f})"
        except Exception:
            pass
        return SectionCutSpec(
            axis_name="X",
            coord_mm=float(x_cut),
            reason=reason,
            plane=Plane.YZ.offset(float(x_cut)),
            normal=(1.0, 0.0, 0.0),
            origin=(float(x_cut), cy, cz),
            u_label="Y",
            v_label="Z",
        )


def analyze_model(
    shape: Shape,
    full_poly: pv.PolyData,
    name: str,
    script_path: Path,
) -> tuple[CadReport, pv.PolyData, SectionCutSpec]:
    """Run all geometric, DFM, overhang, stability, wall-thickness, and SoM cluster checks."""
    bb = shape.bounding_box()
    xmin, ymin, zmin = float(bb.min.X), float(bb.min.Y), float(bb.min.Z)
    xmax, ymax, zmax = float(bb.max.X), float(bb.max.Y), float(bb.max.Z)
    sx, sy, sz = float(bb.size.X), float(bb.size.Y), float(bb.size.Z)

    valid_brep = bool(shape.is_valid)
    solid_count = len(shape.solids())
    face_count = len(shape.faces())
    edge_count = len(shape.edges())
    vol_mm3 = float(shape.volume)
    vol_cm3 = vol_mm3 / 1000.0
    area_mm2 = float(shape.area)
    area_cm2 = area_mm2 / 100.0

    # Build volume check vs Bambu Lab P1S (256 x 256 x 256 mm)
    bw, bd, bh = BUILD_VOL_MM
    margins = (bw - sx, bd - sy, bh - sz)
    occ_pct = (sx / bw * 100.0, sy / bd * 100.0, sz / bh * 100.0)
    exceeds_build = any(m < 0.0 for m in margins)
    near_build = any(o >= 95.0 for o in occ_pct)

    # Bed contact area and flatness at Z = zmin
    bed_area_mm2 = 0.0
    bed_xmin, bed_xmax = xmax, xmin
    bed_ymin, bed_ymax = ymax, ymin
    for f in shape.faces():
        f_bb = f.bounding_box()
        if abs(f_bb.min.Z - zmin) < 1e-2 and abs(f_bb.max.Z - zmin) < 1e-2:
            bed_area_mm2 += float(f.area)
            bed_xmin = min(bed_xmin, float(f_bb.min.X))
            bed_xmax = max(bed_xmax, float(f_bb.max.X))
            bed_ymin = min(bed_ymin, float(f_bb.min.Y))
            bed_ymax = max(bed_ymax, float(f_bb.max.Y))

    bed_area_cm2 = bed_area_mm2 / 100.0
    bed_at_z0 = abs(zmin) < 1e-2
    bed_flatness_ok = bed_area_mm2 > 25.0 and bed_at_z0

    # Detect fillets touching build plate
    bottom_fillet_count, bottom_fillet_radii = _detect_bottom_fillets(shape, zmin)

    # Center of mass & tipping stability margin
    try:
        com = shape.center()
        com_xyz = (float(com.X), float(com.Y), float(com.Z))
    except Exception:
        com_xyz = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)

    if bed_area_mm2 > 0:
        min_tip_margin_mm = min(
            com_xyz[0] - bed_xmin,
            bed_xmax - com_xyz[0],
            com_xyz[1] - bed_ymin,
            bed_ymax - com_xyz[1],
        )
        footprint_xy = (bed_xmin, bed_xmax, bed_ymin, bed_ymax)
    else:
        min_tip_margin_mm = -999.0
        footprint_xy = (xmin, xmax, ymin, ymax)

    # Overhang analysis on mesh cells with smooth vertex normals for CAD shading
    poly = full_poly.compute_normals(
        cell_normals=True,
        point_normals=True,
        split_vertices=True,
        feature_angle=35.0,
    )
    sized = poly.compute_cell_sizes(length=False, area=True, volume=False)
    poly.cell_data["Area"] = sized.cell_data["Area"]
    normals = poly.cell_data["Normals"]
    centers = poly.cell_centers().points
    areas = poly.cell_data["Area"]

    nz = normals[:, 2]
    cell_z = centers[:, 2]
    cluster_radius = max(min(sx, sy, sz) * 0.10, 4.5)

    # 0 = Bed contact (Blue), 1 = Safe <=45 deg (Steel-Blue), 2 = 45-60 deg (Amber), 3 = >60 deg (Red)
    is_bed = (np.abs(cell_z - zmin) < 0.25) & (nz < -0.92)
    raw_amber = (~is_bed) & (nz < -0.7254) & (nz >= -0.8660)
    raw_red = (~is_bed) & (nz < -0.8660)

    red_clusters, is_red = _extract_spatial_clusters(
        raw_red,
        centers,
        areas,
        normals,
        cluster_radius_mm=cluster_radius,
        min_area_mm2=MIN_OVERHANG_CLUSTER_MM2,
    )
    amber_clusters, is_amber = _extract_spatial_clusters(
        raw_amber,
        centers,
        areas,
        normals,
        cluster_radius_mm=cluster_radius,
        min_area_mm2=MIN_OVERHANG_CLUSTER_MM2,
    )

    overhang_class = np.ones(poly.n_cells, dtype=float)
    overhang_class[is_bed] = 0.0
    overhang_class[is_amber] = 2.0
    overhang_class[is_red] = 3.0
    poly.cell_data["OverhangClass"] = overhang_class

    amber_mm2 = float(np.sum(areas[is_amber]))
    red_mm2 = float(np.sum(areas[is_red]))
    amber_cm2 = amber_mm2 / 100.0
    red_cm2 = red_mm2 / 100.0

    # Cluster-filtered opposing-wall thickness analysis
    max_ray = max(sx, sy, sz)
    thickness, valid_mask = _compute_wall_thickness(
        poly, max_ray=max_ray, cluster_radius_mm=cluster_radius
    )
    poly.cell_data["Thickness_mm"] = thickness

    # 0 = OK (> 1.6 mm, Gray), 1 = WARN (0.8 - 1.6 mm, Amber), 2 = CRIT (< 0.8 mm, Red)
    is_wall_crit = thickness < WALL_CRIT_MM
    is_wall_warn = (thickness >= WALL_CRIT_MM) & (thickness < WALL_WARN_MM)

    wall_crit_clusters, is_wall_crit = _extract_spatial_clusters(
        is_wall_crit,
        centers,
        areas,
        normals,
        cluster_radius_mm=cluster_radius,
        min_area_mm2=MIN_WALL_CLUSTER_MM2,
    )
    wall_warn_clusters, is_wall_warn = _extract_spatial_clusters(
        is_wall_warn,
        centers,
        areas,
        normals,
        cluster_radius_mm=cluster_radius,
        min_area_mm2=MIN_WALL_CLUSTER_MM2,
    )

    wall_class = np.zeros(poly.n_cells, dtype=float)
    wall_class[is_wall_warn] = 1.0
    wall_class[is_wall_crit] = 2.0
    poly.cell_data["WallClass"] = wall_class

    valid_th = thickness[valid_mask] if np.any(valid_mask) else thickness
    wall_min = float(np.min(valid_th))
    wall_p5 = float(np.percentile(valid_th, 5))
    wall_p10 = float(np.percentile(valid_th, 10))
    wall_p50 = float(np.percentile(valid_th, 50))
    wall_p90 = float(np.percentile(valid_th, 90))
    wall_max = float(np.max(valid_th))

    total_cell_area = max(float(np.sum(areas)), 1e-6)
    wall_red_pct = float(np.sum(areas[is_wall_crit])) / total_cell_area * 100.0
    wall_amber_pct = float(np.sum(areas[is_wall_warn])) / total_cell_area * 100.0
    wall_ok_pct = max(0.0, 100.0 - wall_red_pct - wall_amber_pct)

    # Build Set-of-Mark (SoM) Defect Markers [D1..D3] for significant defect clusters
    defect_markers: list[DefectMarker] = []

    def _add_marker(
        category: str,
        severity: str,
        short_label: str,
        cluster: dict[str, Any],
        desc_prefix: str,
    ) -> None:
        if len(defect_markers) >= 3:
            return
        m_id = f"D{len(defect_markers) + 1}"
        cx_m, cy_m, cz_m = cluster["centroid_mm"]
        defect_markers.append(
            DefectMarker(
                id=m_id,
                category=category,
                severity=severity,
                short_label=short_label,
                centroid_mm=cluster["centroid_mm"],
                area_mm2=cluster["area_mm2"],
                area_cm2=cluster["area_cm2"],
                dominant_normal=cluster["normal"],
                description=(
                    f"{desc_prefix} at ({cx_m:.1f}, {cy_m:.1f}, {cz_m:.1f}) mm "
                    f"(cluster area {cluster['area_cm2']:.2f} cm^2)."
                ),
            )
        )

    for cl in wall_crit_clusters[:2]:
        _add_marker(
            category="thin_wall_critical",
            severity="WARN",
            short_label="Wall <0.8mm",
            cluster=cl,
            desc_prefix="Critical thin wall <0.8 mm",
        )
    for cl in red_clusters[:2]:
        if cl["area_cm2"] >= 0.10:
            _add_marker(
                category="overhang_critical",
                severity="WARN",
                short_label="Overhang >60°",
                cluster=cl,
                desc_prefix="Critical overhang/bridge >60 deg",
            )
    if not defect_markers:
        for cl in wall_warn_clusters[:1]:
            if wall_amber_pct > 10.0 or wall_p5 < WALL_WARN_MM:
                _add_marker(
                    category="thin_wall_moderate",
                    severity="WARN",
                    short_label="Wall 0.8-1.6mm",
                    cluster=cl,
                    desc_prefix="Moderate thin wall 0.8-1.6 mm",
                )
        for cl in amber_clusters[:1]:
            if amber_cm2 > 5.0:
                _add_marker(
                    category="overhang_moderate",
                    severity="WARN",
                    short_label="Overhang 45-60°",
                    cluster=cl,
                    desc_prefix="Moderate overhang 45-60 deg",
                )

    # Smart section cut plane selection (driven by D1 or internal feature plane)
    section_spec = _select_smart_section_cut(
        shape=shape,
        bbox_min=(xmin, ymin, zmin),
        bbox_max=(xmax, ymax, zmax),
        defect_markers=defect_markers,
    )

    features = _extract_feature_catalog(shape)

    # Deterministic PASS / WARN / FAIL checks
    checks: dict[str, CheckResult] = {}

    # 1. Geometry sanity check
    if not valid_brep or solid_count != 1 or vol_mm3 <= 1e-3:
        checks["geometry_sanity"] = CheckResult(
            status="FAIL",
            message=f"Invalid geometry: valid_brep={valid_brep}, solid_count={solid_count}, volume={vol_mm3:.2f} mm^3",
            metrics={
                "valid_brep": valid_brep,
                "solid_count": solid_count,
                "volume_mm3": round(vol_mm3, 2),
            },
        )
    else:
        checks["geometry_sanity"] = CheckResult(
            status="PASS",
            message="Valid single-solid manifold B-Rep with non-zero volume.",
            metrics={
                "valid_brep": True,
                "solid_count": 1,
                "volume_mm3": round(vol_mm3, 2),
            },
        )

    # 2. Build volume check
    if exceeds_build:
        checks["build_volume"] = CheckResult(
            status="FAIL",
            message=f"Part ({sx:.1f}x{sy:.1f}x{sz:.1f} mm) exceeds {PRINTER_NAME} build volume ({bw:.0f}x{bd:.0f}x{bh:.0f} mm).",
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "margins_mm": [round(m, 2) for m in margins],
            },
        )
    elif near_build:
        checks["build_volume"] = CheckResult(
            status="WARN",
            message=f"Part is within 5% of {PRINTER_NAME} build volume limits (max occupancy {max(occ_pct):.1f}%).",
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "occupancy_pct": [round(o, 1) for o in occ_pct],
            },
        )
    else:
        checks["build_volume"] = CheckResult(
            status="PASS",
            message=f"Fits comfortably inside {PRINTER_NAME} build volume (max occupancy {max(occ_pct):.1f}%).",
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "occupancy_pct": [round(o, 1) for o in occ_pct],
            },
        )

    # 3. Bed adhesion & Z=0 flatness check
    if bed_area_cm2 < 0.25:
        checks["bed_adhesion"] = CheckResult(
            status="FAIL",
            message=f"Insufficient bed contact area ({bed_area_cm2:.2f} cm^2) at Z_min={zmin:.2f} mm.",
            metrics={
                "contact_area_cm2": round(bed_area_cm2, 3),
                "z_min_mm": round(zmin, 3),
            },
        )
    elif not bed_at_z0 or bed_area_cm2 < 1.0:
        checks["bed_adhesion"] = CheckResult(
            status="WARN",
            message=f"Small bed contact area ({bed_area_cm2:.2f} cm^2) or part bottom Z_min={zmin:.2f} mm != 0.0.",
            metrics={
                "contact_area_cm2": round(bed_area_cm2, 3),
                "z_min_mm": round(zmin, 3),
            },
        )
    else:
        checks["bed_adhesion"] = CheckResult(
            status="PASS",
            message=f"Flat bottom at Z={zmin:.2f} mm with {bed_area_cm2:.2f} cm^2 bed contact area.",
            metrics={
                "contact_area_cm2": round(bed_area_cm2, 3),
                "z_min_mm": round(zmin, 3),
            },
        )

    # 4. Bottom fillets touching build plate check
    if bottom_fillet_count > 0:
        checks["build_plate_fillets"] = CheckResult(
            status="WARN",
            message=(
                f"Detected {bottom_fillet_count} curved fillet face(s) touching Z_min "
                f"(radii: {bottom_fillet_radii} mm). 45-deg chamfers are preferred on the build plate for FDM."
            ),
            metrics={
                "fillet_face_count": bottom_fillet_count,
                "radii_mm": bottom_fillet_radii,
            },
        )
    else:
        checks["build_plate_fillets"] = CheckResult(
            status="PASS",
            message="No overhanging fillets detected on the Z_min build plate interface.",
            metrics={"fillet_face_count": 0},
        )

    # 5. Overhang check
    if red_cm2 > 2.5:
        checks["overhangs"] = CheckResult(
            status="WARN",
            message=f"Critical overhangs/bridges >60 deg detected ({red_cm2:.2f} cm^2 red, {amber_cm2:.2f} cm^2 amber).",
            metrics={
                "red_gt_60_cm2": round(red_cm2, 3),
                "amber_45_60_cm2": round(amber_cm2, 3),
            },
        )
    elif red_cm2 > 0.15 or amber_cm2 > 5.0:
        checks["overhangs"] = CheckResult(
            status="WARN",
            message=f"Moderate overhangs/bridges present ({red_cm2:.2f} cm^2 >60 deg, {amber_cm2:.2f} cm^2 45-60 deg).",
            metrics={
                "red_gt_60_cm2": round(red_cm2, 3),
                "amber_45_60_cm2": round(amber_cm2, 3),
            },
        )
    else:
        checks["overhangs"] = CheckResult(
            status="PASS",
            message=f"Support-free FDM geometry ({red_cm2:.2f} cm^2 >60 deg, {amber_cm2:.2f} cm^2 45-60 deg).",
            metrics={
                "red_gt_60_cm2": round(red_cm2, 3),
                "amber_45_60_cm2": round(amber_cm2, 3),
            },
        )

    # 6. Wall thickness check
    if wall_p5 < WALL_CRIT_MM or wall_red_pct > 2.0 or len(wall_crit_clusters) > 0:
        checks["wall_thickness"] = CheckResult(
            status="WARN",
            message=f"Thin walls <{WALL_CRIT_MM} mm detected (min={wall_min:.2f} mm, P5={wall_p5:.2f} mm, {wall_red_pct:.1f}% area <0.8 mm).",
            metrics={
                "min_mm": round(wall_min, 2),
                "p5_mm": round(wall_p5, 2),
                "red_area_pct": round(wall_red_pct, 2),
            },
        )
    elif wall_p5 < WALL_WARN_MM or wall_amber_pct > 10.0:
        checks["wall_thickness"] = CheckResult(
            status="WARN",
            message=f"Walls in 0.8-1.6 mm warning range (P5={wall_p5:.2f} mm, {wall_amber_pct:.1f}% area in 0.8-1.6 mm).",
            metrics={
                "p5_mm": round(wall_p5, 2),
                "amber_area_pct": round(wall_amber_pct, 2),
            },
        )
    else:
        checks["wall_thickness"] = CheckResult(
            status="PASS",
            message=f"Robust wall thickness (min={wall_min:.2f} mm, P5={wall_p5:.2f} mm, median={wall_p50:.2f} mm).",
            metrics={
                "min_mm": round(wall_min, 2),
                "p5_mm": round(wall_p5, 2),
                "p50_mm": round(wall_p50, 2),
            },
        )

    # 7. Stability / Center of Mass check
    if min_tip_margin_mm < 0.0:
        checks["stability"] = CheckResult(
            status="FAIL",
            message=f"Center of mass projects outside bed footprint (tip margin: {min_tip_margin_mm:.1f} mm)!",
            metrics={"tip_margin_mm": round(min_tip_margin_mm, 2)},
        )
    elif min_tip_margin_mm < 3.0:
        checks["stability"] = CheckResult(
            status="WARN",
            message=f"Low tipping stability margin ({min_tip_margin_mm:.1f} mm).",
            metrics={"tip_margin_mm": round(min_tip_margin_mm, 2)},
        )
    else:
        checks["stability"] = CheckResult(
            status="PASS",
            message=f"Stable upright posture (tip margin: {min_tip_margin_mm:.1f} mm).",
            metrics={"tip_margin_mm": round(min_tip_margin_mm, 2)},
        )

    statuses = {c.status for c in checks.values()}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    report = CadReport(
        name=name,
        script_path=str(script_path.resolve()),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        overall_status=overall,
        valid_brep=valid_brep,
        solid_count=solid_count,
        face_count=face_count,
        edge_count=edge_count,
        volume_mm3=vol_mm3,
        volume_cm3=vol_cm3,
        surface_area_mm2=area_mm2,
        surface_area_cm2=area_cm2,
        bbox_min=(xmin, ymin, zmin),
        bbox_max=(xmax, ymax, zmax),
        bbox_size=(sx, sy, sz),
        build_volume_mm=BUILD_VOL_MM,
        build_volume_margin_mm=margins,
        build_volume_occupancy_pct=occ_pct,
        near_or_exceeds_build_volume=exceeds_build or near_build,
        z_min_mm=zmin,
        bed_contact_area_mm2=bed_area_mm2,
        bed_contact_area_cm2=bed_area_cm2,
        bed_flatness_ok=bed_flatness_ok,
        bottom_fillet_faces_count=bottom_fillet_count,
        bottom_fillet_radii_mm=bottom_fillet_radii,
        overhang_amber_45_60_mm2=amber_mm2,
        overhang_amber_45_60_cm2=amber_cm2,
        overhang_red_gt_60_mm2=red_mm2,
        overhang_red_gt_60_cm2=red_cm2,
        wall_min_mm=wall_min,
        wall_p5_mm=wall_p5,
        wall_p10_mm=wall_p10,
        wall_p50_mm=wall_p50,
        wall_p90_mm=wall_p90,
        wall_max_mm=wall_max,
        wall_red_lt_08_pct=wall_red_pct,
        wall_amber_08_16_pct=wall_amber_pct,
        wall_ok_gt_16_pct=wall_ok_pct,
        center_of_mass_mm=com_xyz,
        support_footprint_bounds_xy=footprint_xy,
        stability_tip_margin_mm=min_tip_margin_mm,
        defect_markers=defect_markers,
        section_cut={
            "plane_axis": section_spec.axis_name,
            "coordinate_mm": round(section_spec.coord_mm, 2),
            "projection_axes": f"{section_spec.u_label}{section_spec.v_label}",
            "selection_reason": section_spec.reason,
        },
        features=features,
        checks=checks,
    )
    return report, poly, section_spec


# =============================================================================
# HEADLESS SLICER CLI HOOK (OrcaSlicer / Bambu Studio / PrusaSlicer)
# =============================================================================
def _resolve_orca_profile(base_dir: Path, folder: str, name: str) -> dict[str, Any]:
    """Recursively flatten OrcaSlicer / Bambu Studio JSON profile inheritance for CLI mode."""
    p = base_dir / folder / f"{name}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    parent = data.get("inherits")
    if parent:
        merged = _resolve_orca_profile(base_dir, folder, parent)
        merged.update(data)
        return merged
    return data


def run_headless_slicer(
    shape: Shape, slicer_override: str | None = None
) -> dict[str, Any]:
    """Invoke headless OrcaSlicer / Bambu Studio / PrusaSlicer CLI and parse real G-code metrics."""
    candidates = (
        [Path(slicer_override)]
        if slicer_override
        else [
            Path(r"C:\Program Files\OrcaSlicer\orca-slicer.exe"),
            Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe"),
            Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"),
        ]
    )
    slicer_exe: Path | None = None
    for cand in candidates:
        if cand.exists():
            slicer_exe = cand
            break
        found = shutil.which(str(cand))
        if found:
            slicer_exe = Path(found)
            break

    if slicer_exe is None:
        return {
            "status": "UNAVAILABLE",
            "error": "No headless slicer executable found.",
        }

    with tempfile.TemporaryDirectory(prefix="codecad_slice_") as tmp_str:
        tmp_dir = Path(tmp_str)
        stl_path = tmp_dir / "temp_part.stl"
        export_stl(shape, str(stl_path))

        profile_roots = [
            slicer_exe.parent / "resources" / "profiles" / "BBL",
            Path(r"C:\Program Files\OrcaSlicer\resources\profiles\BBL"),
            Path(r"C:\Program Files\Bambu Studio\resources\profiles\BBL"),
        ]
        bbl_root = next((r for r in profile_roots if r.exists()), None)

        cmd = [str(slicer_exe)]
        if bbl_root is not None:
            for fname, folder, prof_name in [
                ("machine.json", "machine", "Bambu Lab P1S 0.4 nozzle"),
                ("process.json", "process", "0.20mm Standard @BBL X1C"),
                ("filament.json", "filament", "Bambu PLA Basic @BBL X1C"),
            ]:
                flat = _resolve_orca_profile(bbl_root, folder, prof_name)
                flat.pop("inherits", None)
                for k in (
                    "extruder_variant_list",
                    "printer_extruder_variant",
                    "print_extruder_variant",
                    "filament_extruder_variant",
                ):
                    flat.pop(k, None)
                (tmp_dir / fname).write_text(
                    json.dumps(flat, indent=2), encoding="utf-8"
                )

            cmd.extend(
                [
                    "--load-settings",
                    f"{tmp_dir / 'machine.json'};{tmp_dir / 'process.json'}",
                    "--load-filaments",
                    str(tmp_dir / "filament.json"),
                ]
            )

        cmd.extend(["--slice", "0", "--outputdir", str(tmp_dir), str(stl_path)])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        gcodes = list(tmp_dir.glob("*.gcode"))
        if proc.returncode != 0 or not gcodes:
            return {
                "status": "ERROR",
                "slicer_executable": str(slicer_exe),
                "returncode": proc.returncode,
                "stderr": (proc.stderr or proc.stdout)[-500:],
            }

        gcode_text = gcodes[0].read_text(encoding="utf-8", errors="replace")

    model_time_str = None
    total_time_str = None
    total_layers = None
    filament_mm = None
    filament_cm3 = None
    filament_g = None

    m_time = re.search(
        r";\s*model printing time:\s*([^;]+);\s*total estimated time:\s*(.+)",
        gcode_text,
    )
    if m_time:
        model_time_str = m_time.group(1).strip()
        total_time_str = m_time.group(2).strip()

    m_layers = re.search(r";\s*total layer number:\s*(\d+)", gcode_text)
    if m_layers:
        total_layers = int(m_layers.group(1))

    for pat, attr in [
        (
            r";\s*(?:filament used \[mm\]\s*=|total filament length \[mm\]\s*:)\s*([\d.]+)",
            "mm",
        ),
        (
            r";\s*(?:filament used \[cm3\]\s*=|total filament volume \[cm\^3\]\s*:)\s*([\d.]+)",
            "cm3",
        ),
        (
            r";\s*(?:filament used \[g\]\s*=|total filament weight \[g\]\s*:)\s*([\d.]+)",
            "g",
        ),
    ]:
        matches = re.findall(pat, gcode_text)
        if matches:
            val = float(matches[-1])
            if attr == "mm":
                filament_mm = val
            elif attr == "cm3":
                filament_cm3 = val
            elif attr == "g":
                filament_g = val

    feature_counts: dict[str, int] = {}
    bridge_extrusions = 0
    floating_shell_blocks = 0
    overhang_wall_blocks = 0

    current_feature = ""
    for line in gcode_text.splitlines():
        if line.startswith("; FEATURE:"):
            current_feature = line.split(":", 1)[1].strip()
            feature_counts[current_feature] = feature_counts.get(current_feature, 0) + 1
            low = current_feature.lower()
            if "floating" in low:
                floating_shell_blocks += 1
            elif "overhang" in low:
                overhang_wall_blocks += 1
        elif "bridge" in current_feature.lower() and (
            line.startswith("G1 ") or line.startswith("G0 ")
        ):
            if " E" in line and " E-" not in line:
                bridge_extrusions += 1

    bridge_blocks = sum(
        cnt for feat, cnt in feature_counts.items() if "bridge" in feat.lower()
    )

    return {
        "status": "OK",
        "slicer_executable": slicer_exe.name,
        "printer_profile": "Bambu Lab P1S 0.4 nozzle (0.20mm Standard PLA)",
        "model_print_time": model_time_str,
        "total_estimated_time": total_time_str,
        "total_layers": total_layers,
        "filament_used_g": filament_g,
        "filament_used_cm3": filament_cm3,
        "filament_used_mm": filament_mm,
        "bridge_feature_blocks": bridge_blocks,
        "bridge_extrusion_segments": bridge_extrusions,
        "floating_vertical_shell_blocks": floating_shell_blocks,
        "overhang_wall_blocks": overhang_wall_blocks,
        "feature_block_counts": feature_counts,
    }


# =============================================================================
# 3D RENDERING HELPERS (PARALLEL PROJECTION, B-REP EDGES & SET-OF-MARK BADGES)
# =============================================================================
def _build_edge_polydata(shape: Shape) -> pv.PolyData | None:
    """Sample all exact build123d B-Rep edges into a single PyVista PolyData."""
    all_points: list[tuple[float, float, float]] = []
    all_lines: list[int] = []

    for edge in shape.edges():
        if edge.length < 1e-4:
            continue
        n = (
            2
            if edge.geom_type == GeomType.LINE
            else max(20, min(80, int(edge.length / 0.22) + 2))
        )
        pts = edge.positions([i / (n - 1) for i in range(n)])
        start_idx = len(all_points)
        all_points.extend((p.X, p.Y, p.Z) for p in pts)
        all_lines.append(n)
        all_lines.extend(range(start_idx, start_idx + n))

    if not all_points:
        return None
    return pv.PolyData(
        np.array(all_points, dtype=float), lines=np.array(all_lines, dtype=int)
    )


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
    """Configure an orthographic parallel-projection camera (Rule 3)."""
    xmin, ymin, zmin = report.bbox_min
    xmax, ymax, zmax = report.bbox_max
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    dist = max(diag * 2.2, 10.0)
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    pos = (
        cx + dist * dx / norm,
        cy + dist * dy / norm,
        cz + dist * dz / norm,
    )
    plotter.enable_parallel_projection()
    plotter.camera_position = [pos, (cx, cy, cz), up]
    plotter.reset_camera(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
    plotter.camera.zoom(zoom)


def _add_conditional_build_box(plotter: pv.Plotter, report: CadReport) -> None:
    """Render printer build-volume wireframe ONLY if the part exceeds or is within 5% of limits."""
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


def _add_3d_som_markers(
    plotter: pv.Plotter,
    report: CadReport,
) -> None:
    """Overlay Set-of-Mark (SoM) 3D callout badges `[D1]`, `[D2]` at defect centroids (Rule 8)."""
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


# =============================================================================
# SHEET 1: 3D SPATIAL, DFM & SMART CUTAWAY SHEET (`preview.png` — 2x2)
# =============================================================================
def render_sheet1_3d_overview(
    shape: Shape,
    analyzed_poly: pv.PolyData,
    edge_mesh: pv.PolyData | None,
    report: CadReport,
    section_spec: SectionCutSpec,
    out_path: Path,
) -> None:
    """Render Sheet 1 (`preview.png`): 2x2 Parallel-Projection 3D Spatial, DFM & Smart Cutaway."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(
        shape=(2, 2),
        off_screen=True,
        window_size=(1800, 1500),
        lighting="light_kit",
    )
    plotter.set_background(COLOR_BG_LIGHT)

    xmin, ymin, zmin = report.bbox_min
    xmax, ymax, _ = report.bbox_max
    sx, sy, _ = report.bbox_size
    cx, cy, cz = report.center_of_mass_mm

    # 10 mm pitch reference build-plate grid at Z = zmin
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
        i_resolution=max(4, min(32, int(round(span_x / grid_step)))),
        j_resolution=max(4, min(32, int(round(span_y / grid_step)))),
    )

    # Unified 4-state DFM colormap: 0=Bed Blue, 1=Safe CAD Steel-Blue, 2=Amber 45-60°, 3=Red >60°
    dfm_cmap = ListedColormap(
        [COLOR_BED_BLUE, COLOR_SAFE_CAD, COLOR_WARN_AMBER, COLOR_CRIT_RED]
    )

    # (0, 0) & (0, 1): Two opposite top isometric views with unified CAD + DFM coloring
    top_iso_views = [
        ((0, 0), "1. Iso Front-Right-Top (+X -Y +Z) — CAD & DFM", (1.05, -1.15, 0.80)),
        (
            (0, 1),
            "2. Iso Back-Left-Top (-X +Y +Z) — Opposite Side",
            (-1.05, 1.15, 0.80),
        ),
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
        plotter.add_text(
            title, position="upper_left", font_size=12, color=COLOR_EDGE_DARK
        )
        plotter.add_axes()
        _set_parallel_iso_camera(plotter, report, direction=cam_dir)

    # (1, 0): Underside view (-Z) exposing Bed Contact (Blue) & Downward Overhangs (Amber/Red)
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
        font_size=11.5,
        color=COLOR_EDGE_DARK,
    )
    plotter.add_axes()
    _set_parallel_iso_camera(plotter, report, direction=(-1.05, -1.15, -0.75))

    # (1, 1): 4-in-1 Smart 3D Cutaway + Wall Thickness + Ghost Context + CoM Plumb Line
    plotter.subplot(1, 1)
    wall_cmap = ListedColormap([COLOR_OK_GRAY, COLOR_WARN_AMBER, COLOR_CRIT_RED])

    # Determine clip direction so the cut face faces toward the Front-Right-Top camera (+X, -Y, +Z)
    if section_spec.axis_name == "X":
        # Camera sits at +X; clip away the x > x_cut half so the cut face at x_cut is visible
        clip_normal = (1.0, 0.0, 0.0)
        invert_clip = True
    else:
        # Camera sits at -Y; clip away the y < y_cut half so the cut face at y_cut is visible
        clip_normal = (0.0, -1.0, 0.0)
        invert_clip = True

    clipped_solid = analyzed_poly.clip(
        normal=clip_normal, origin=section_spec.origin, invert=invert_clip
    )
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

    # Full whole-part ghost wireframe + faint shell so the cutaway never floats in a void
    plotter.add_mesh(
        analyzed_poly,
        color="#94A3B8",
        opacity=0.07,
        smooth_shading=True,
        lighting=False,
    )
    _add_edges(plotter, edge_mesh, color=COLOR_GHOST_GRAY, line_width=1.0, opacity=0.22)

    # Crisp edges on the kept solid half
    if edge_mesh is not None and edge_mesh.n_points > 0:
        kept_edges = edge_mesh.clip(
            normal=clip_normal, origin=section_spec.origin, invert=invert_clip
        )
        if kept_edges.n_points > 0:
            _add_edges(
                plotter, kept_edges, color=COLOR_EDGE_DARK, line_width=1.5, opacity=0.90
            )

    # High-contrast B-Rep section cap on the cut plane A-A
    try:
        sec_shape = b3d.section(shape, section_by=section_spec.plane)
        sec_faces = sec_shape.faces() if hasattr(sec_shape, "faces") else []
        if sec_faces:
            pyvista_cad.add_cad(
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
        slc = analyzed_poly.slice(
            normal=section_spec.normal, origin=section_spec.origin
        )
        if slc.n_cells > 0:
            plotter.add_mesh(slc, color=COLOR_CAP_CYAN, line_width=3.0, lighting=False)

    # Center of Mass sphere, vertical gravity plumb line & tipping support footprint
    r_marker = max(min(report.bbox_size) * 0.032, 0.85)
    com_sphere = pv.Sphere(radius=r_marker, center=(cx, cy, cz))
    floor_proj = pv.Sphere(radius=r_marker * 0.72, center=(cx, cy, zmin))
    plumb_line = pv.Line((cx, cy, cz), (cx, cy, zmin))
    plotter.add_mesh(com_sphere, color=COLOR_BED_BLUE, lighting=False)
    plotter.add_mesh(floor_proj, color=COLOR_BED_BLUE, lighting=False)
    plotter.add_mesh(plumb_line, color=COLOR_BED_BLUE, line_width=2.6, lighting=False)

    fxmin, fxmax, fymin, fymax = report.support_footprint_bounds_xy
    if fxmax > fxmin and fymax > fymin:
        footprint_rect = pv.Rectangle(
            [(fxmin, fymin, zmin), (fxmax, fymin, zmin), (fxmax, fymax, zmin)]
        )
        fp_color = (
            COLOR_BED_BLUE
            if report.stability_tip_margin_mm >= 3.0
            else COLOR_WARN_AMBER
        )
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
    plotter.add_text(
        cut_title, position="upper_left", font_size=11, color=COLOR_EDGE_DARK
    )
    plotter.add_axes()
    _set_parallel_iso_camera(plotter, report, direction=(1.05, -1.15, 0.80))

    plotter.screenshot(str(out_path))
    plotter.close()
    _crop_whitespace_png(out_path)


# =============================================================================
# SHEET 2: 2D VECTOR BLUEPRINT & HATCHED SECTION A-A (`preview_engineering.png`)
# =============================================================================
def _sample_2d_projected_edge(
    edge: Edge, u_offset: float = 0.0, v_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    n = (
        2
        if edge.geom_type == GeomType.LINE
        else max(24, min(72, int(edge.length / 0.35) + 2))
    )
    pts = edge.positions([i / (n - 1) for i in range(n)])
    xs = np.array([p.X + u_offset for p in pts], dtype=float)
    ys = np.array([p.Y + v_offset for p in pts], dtype=float)
    return xs, ys


def _draw_outer_dim_h(
    ax: plt.Axes,
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
    ax.plot(
        [x1, x1],
        [y_ref + sign * gap, y_dim + sign * overshoot],
        color=color,
        lw=0.9,
        alpha=0.85,
    )
    ax.plot(
        [x2, x2],
        [y_ref + sign * gap, y_dim + sign * overshoot],
        color=color,
        lw=0.9,
        alpha=0.85,
    )
    ax.annotate(
        "",
        xy=(x1, y_dim),
        xytext=(x2, y_dim),
        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.2, mutation_scale=11),
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
        bbox=dict(boxstyle="square,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.92),
    )


def _draw_outer_dim_v(
    ax: plt.Axes,
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
    ax.plot(
        [x_ref + sign * gap, x_dim + sign * overshoot],
        [y1, y1],
        color=color,
        lw=0.9,
        alpha=0.85,
    )
    ax.plot(
        [x_ref + sign * gap, x_dim + sign * overshoot],
        [y2, y2],
        color=color,
        lw=0.9,
        alpha=0.85,
    )
    ax.annotate(
        "",
        xy=(x_dim, y1),
        xytext=(x_dim, y2),
        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.2, mutation_scale=11),
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
        bbox=dict(boxstyle="square,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.92),
    )


def _draw_ordinate_z_levels(
    ax: plt.Axes,
    u_max: float,
    v_min: float,
    v_max: float,
    max_span: float,
    z_levels: list[float],
) -> None:
    """Draw non-overlapping ordinate Z-step level callouts on the right margin of XZ/YZ views."""
    if not z_levels:
        return
    min_gap = max(max_span * 0.075, (v_max - v_min) * 0.085, 1.2)
    selected: list[float] = []
    for z in sorted(z_levels):
        if not selected or abs(z - selected[-1]) >= min_gap:
            selected.append(z)
        elif abs(z - v_max) < 1e-2:
            selected[-1] = z

    x_tick_start = u_max + max_span * 0.02
    x_tick_end = u_max + max_span * 0.065
    for z in selected:
        ax.plot(
            [x_tick_start, x_tick_end],
            [z, z],
            color="#475569",
            lw=0.85,
            ls="-",
            alpha=0.85,
            zorder=5,
        )
        ax.text(
            x_tick_end + max_span * 0.01,
            z,
            f"Z={z:.1f}",
            fontsize=9.0,
            fontfamily="monospace",
            fontweight="bold",
            color="#334155",
            ha="left",
            va="center",
            bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.88),
            zorder=6,
        )


def _draw_2d_som_markers(
    ax: plt.Axes,
    report: CadReport,
    u_label: str,
    v_label: str,
    filter_section: SectionCutSpec | None = None,
) -> None:
    """Project Set-of-Mark `[D1]`, `[D2]` defect pins onto a 2D (u, v) orthographic axis.

    Coincident projected markers (e.g. symmetric D1/D2 viewed from the side) are merged
    into a single badge (`D1,D2`) to prevent overlapping labels.
    """
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
            bbox=dict(boxstyle="round,pad=0.22", fc=badge_color, ec="#FFFFFF", lw=0.9),
            zorder=10,
        )


def _wire_to_ordered_2d_polygon(wire: Wire, axis_name: str) -> np.ndarray | None:
    """Sample a closed build123d Wire on an X or Y section plane into an ordered 2D polygon."""
    edge_chains: list[np.ndarray] = []
    for e in wire.edges():
        if e.length < 1e-4:
            continue
        n = (
            2
            if e.geom_type == GeomType.LINE
            else max(18, min(64, int(e.length / 0.3) + 2))
        )
        pts = e.positions([i / (n - 1) for i in range(n)])
        if axis_name == "X":
            # YZ plane -> (Y, Z)
            arr = np.array([[p.Y, p.Z] for p in pts], dtype=float)
        else:
            # XZ plane -> (X, Z)
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


def _face_to_compound_mpl_path(face: Face, axis_name: str) -> MplPath | None:
    """Convert a planar B-Rep Face (with optional inner holes) into a compound Matplotlib Path."""
    outer_poly = _wire_to_ordered_2d_polygon(face.outer_wire(), axis_name)
    if outer_poly is None:
        return None
    # Ensure outer ring is counter-clockwise (signed area > 0)
    if _signed_area_2d(outer_poly) < 0:
        outer_poly = outer_poly[::-1]

    rings = [outer_poly]
    for iw in face.inner_wires():
        inner_poly = _wire_to_ordered_2d_polygon(iw, axis_name)
        if inner_poly is not None:
            # Ensure inner hole ring is clockwise (signed area < 0) so PathPatch leaves it unhatched
            if _signed_area_2d(inner_poly) > 0:
                inner_poly = inner_poly[::-1]
            rings.append(inner_poly)

    verts_list: list[np.ndarray] = []
    codes_list: list[int] = []
    for r in rings:
        closed = np.vstack([r, r[0]])
        verts_list.append(closed)
        codes_list.append(MplPath.MOVETO)
        codes_list.extend([MplPath.LINETO] * (len(r) - 1))
        codes_list.append(MplPath.CLOSEPOLY)

    return MplPath(np.vstack(verts_list), codes_list)


def _render_clean_hlr_view(
    ax: plt.Axes,
    shape: Shape,
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
        xs, ys = _sample_2d_projected_edge(e, u_offset, v_offset)
        ax.plot(
            xs,
            ys,
            color=COLOR_OK_GRAY,
            lw=0.85,
            ls=(0, (4.0, 2.5)),
            alpha=0.62,
            zorder=2,
        )

    for e in vis:
        if e.length < 0.05:
            continue
        xs, ys = _sample_2d_projected_edge(e, u_offset, v_offset)
        all_u.extend(xs.tolist())
        all_v.extend(ys.tolist())
        ax.plot(
            xs,
            ys,
            color=COLOR_EDGE_DARK,
            lw=1.5,
            solid_capstyle="round",
            zorder=4,
        )

    if not all_u or not all_v:
        return

    u_min, u_max = min(all_u), max(all_u)
    v_min, v_max = min(all_v), max(all_v)
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

    # Symmetry centerlines if origin lies inside view
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

    # Draw Section Cut Line A-A on Top View (XY Plane)
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
            for y_end, va_align in [
                (v_max + cut_ext, "bottom"),
                (v_min - cut_ext, "top"),
            ]:
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
                    bbox=dict(boxstyle="square,pad=0.18", fc=COLOR_BED_BLUE, ec="none"),
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
            for x_end, ha_align in [
                (u_min - cut_ext, "right"),
                (u_max + cut_ext, "left"),
            ]:
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
                    bbox=dict(boxstyle="square,pad=0.18", fc=COLOR_BED_BLUE, ec="none"),
                    zorder=8,
                )

    # Main outer bounding dimensions
    dim_offset = max_span * 0.13
    _draw_outer_dim_h(
        ax,
        u_min,
        u_max,
        v_min,
        v_min - dim_offset,
        f"{u_span:.2f} mm ({u_label})",
    )
    _draw_outer_dim_v(
        ax,
        v_min,
        v_max,
        u_min,
        u_min - dim_offset,
        f"{v_span:.2f} mm ({v_label})",
    )

    # Optional right-margin ordinate Z-step levels
    z_levels = report.features.get("horizontal_planar_z_levels_mm", [])
    if show_z_levels and v_label == "Z":
        _draw_ordinate_z_levels(ax, u_max, v_min, v_max, max_span, z_levels)

    # Set-of-Mark [D1] defect badges
    _draw_2d_som_markers(ax, report, u_label=u_label, v_label=v_label)

    pad_u = max_span * 0.24
    pad_v = max_span * 0.24
    right_mult = 0.85 if (show_z_levels and v_label == "Z") else 0.65
    ax.set_xlim(u_min - pad_u, u_max + pad_u * right_mult)
    ax.set_ylim(v_min - pad_v, v_max + pad_v * 0.65)
    ax.set_xlabel(
        f"{u_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155"
    )
    ax.set_ylabel(
        f"{v_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155"
    )
    ax.tick_params(labelsize=10, colors="#334155")


def _render_2d_hatched_section_view(
    ax: plt.Axes,
    shape: Shape,
    report: CadReport,
    section_spec: SectionCutSpec,
) -> None:
    """Render Panel 2 of Sheet 2: True 2D Hatched B-Rep Cross-Section A-A on a `[mm]` grid."""
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

    # 1. Faint background silhouette of full part for context
    try:
        vis_bg, _ = shape.project_to_viewport(
            viewport_origin=vp_origin,
            viewport_up=(0, 0, 1),
            look_at=center,
        )
        for e in vis_bg:
            if e.length < 0.1:
                continue
            xs, ys = _sample_2d_projected_edge(e, u_offset, v_offset)
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

    # 2. Exact B-Rep section faces with 45-degree technical cross-hatching (`///`)
    all_u: list[float] = [u_min_bb, u_max_bb]
    all_v: list[float] = [v_min_bb, v_max_bb]
    try:
        sec_shape = b3d.section(shape, section_by=section_spec.plane)
        sec_faces = sec_shape.faces() if hasattr(sec_shape, "faces") else []
        for f in sec_faces:
            cpath = _face_to_compound_mpl_path(f, section_spec.axis_name)
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
                verts = cpath.vertices
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
    _draw_outer_dim_h(
        ax,
        u_min,
        u_max,
        v_min,
        v_min - dim_offset,
        f"{u_span:.2f} mm ({u_label})",
    )
    _draw_outer_dim_v(
        ax,
        v_min,
        v_max,
        u_min,
        u_min - dim_offset,
        f"{v_span:.2f} mm ({v_label})",
    )

    z_levels = report.features.get("horizontal_planar_z_levels_mm", [])
    _draw_ordinate_z_levels(ax, u_max, v_min, v_max, max_span, z_levels)
    _draw_2d_som_markers(
        ax, report, u_label=u_label, v_label=v_label, filter_section=section_spec
    )

    # Compact legend badge inside section view
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
        bbox=dict(
            boxstyle="round,pad=0.28", fc="#F8FAFC", ec="#CBD5E1", lw=0.9, alpha=0.92
        ),
        zorder=8,
    )

    pad_u = max_span * 0.24
    pad_v = max_span * 0.24
    ax.set_xlim(u_min - pad_u, u_max + pad_u * 0.85)
    ax.set_ylim(v_min - pad_v, v_max + pad_v * 0.65)
    ax.set_xlabel(
        f"{u_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155"
    )
    ax.set_ylabel(
        f"{v_label} [mm]", fontsize=11, fontfamily="monospace", color="#334155"
    )
    ax.tick_params(labelsize=10, colors="#334155")


def render_sheet2_2d_engineering(
    shape: Shape,
    report: CadReport,
    section_spec: SectionCutSpec,
    out_path: Path,
) -> None:
    """Render Sheet 2 (`preview_engineering.png`): 2x2 Pure 2D Vector HLR & Hatched Section A-A."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cx = (report.bbox_min[0] + report.bbox_max[0]) / 2.0
    cy = (report.bbox_min[1] + report.bbox_max[1]) / 2.0
    cz = (report.bbox_min[2] + report.bbox_max[2]) / 2.0
    center = (cx, cy, cz)
    dist = max(report.bbox_size) * 5.0 + 100.0

    fig, axes = plt.subplots(
        2, 2, figsize=(18, 14.5), dpi=130, facecolor=COLOR_BG_LIGHT
    )
    plt.subplots_adjust(
        left=0.06, right=0.96, bottom=0.06, top=0.94, wspace=0.18, hspace=0.22
    )

    ax_top = axes[0, 0]
    ax_section = axes[0, 1]
    ax_front = axes[1, 0]
    ax_right = axes[1, 1]

    for ax in (ax_top, ax_section, ax_front, ax_right):
        ax.set_facecolor("#FFFFFF")

    # (0, 0): Top HLR View (XY Plane) with Section Cut Line A-A
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

    # (0, 1): True 2D Hatched Cross-Section A-A
    _render_2d_hatched_section_view(
        ax_section,
        shape,
        report,
        section_spec=section_spec,
    )

    # (1, 0): Front HLR View (XZ Plane) aligned below Top View
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

    # (1, 1): Right HLR View (YZ Plane) aligned right of Front View
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
    _crop_whitespace_png(out_path)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOTA Zero-Tier Visual & Symbolic CAD + FDM DFM Inspection Pipeline."
    )
    parser.add_argument(
        "script", nargs="?", default="test_part.py", help="Path to build123d script"
    )
    parser.add_argument("--var", default=None, help="Variable name to render")
    parser.add_argument(
        "--out",
        default=None,
        help="Output main PNG path (defaults to <script_dir>/.preview/preview.png)",
    )
    parser.add_argument(
        "--slice",
        action="store_true",
        help="Invoke headless slicer CLI (OrcaSlicer / Bambu Studio) and record G-code telemetry in report.json",
    )
    parser.add_argument(
        "--slicer-path",
        default=None,
        help="Optional explicit path to headless slicer executable",
    )
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    shape = _load_shape(script_path, args.var)
    part_name = f"{script_path.stem}" + (f" ({args.var})" if args.var else "")

    out_path = (
        Path(args.out)
        if args.out is not None
        else script_path.resolve().parent / ".preview" / "preview.png"
    )
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_path.stem
    engineering_path = out_dir / f"{stem}_engineering.png"
    report_path = out_dir / "report.json"

    # Clean up any legacy multi-tier sheets if present
    for legacy_name in (
        f"{stem}_diagnostics.png",
        f"{stem}_surface_audit.png",
        f"{stem}_blueprint.png",
        f"{stem}_sections.png",
        f"{stem}_dfm.png",
    ):
        legacy_file = out_dir / legacy_name
        if legacy_file.exists():
            try:
                legacy_file.unlink()
            except OSError:
                pass

    # High-fidelity tessellation for smooth normal shading and accurate ray-casting
    raw_mesh = pyvista_cad.from_build123d(
        shape, linear_deflection=0.025, angular_deflection=0.10
    )
    full_poly = (
        raw_mesh
        if isinstance(raw_mesh, pv.PolyData)
        else pyvista_cad.flatten_to_cad_polydata(raw_mesh)
    )
    edge_mesh = _build_edge_polydata(shape)

    # 1. Analyze geometry, DFM checks, SoM defect clusters, and smart section cut plane
    report, analyzed_poly, section_spec = analyze_model(
        shape, full_poly, part_name, script_path
    )

    # 2. Optional headless slicer CLI hook
    if args.slice:
        report.slicer = run_headless_slicer(shape, slicer_override=args.slicer_path)

    # 3. Sheet 1: 3D Spatial, DFM & Smart Cutaway (`preview.png`)
    render_sheet1_3d_overview(
        shape, analyzed_poly, edge_mesh, report, section_spec, out_path
    )
    report.generated_files.append(str(out_path))

    # 4. Sheet 2: 2D Vector Orthographic HLR & Hatched Section A-A (`preview_engineering.png`)
    render_sheet2_2d_engineering(shape, report, section_spec, engineering_path)
    report.generated_files.append(str(engineering_path))

    # 5. Symbolic Ground Truth (`report.json`)
    report.generated_files.append(str(report_path))
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Concise terminal summary
    sx, sy, sz = report.bbox_size
    print(
        f"[{report.overall_status}] {report.name} | "
        f"Size: {sx:.2f} x {sy:.2f} x {sz:.2f} mm | "
        f"Vol: {report.volume_cm3:.2f} cm^3 | "
        f"Bed: {report.bed_contact_area_cm2:.2f} cm^2 | "
        f"Wall min/P5: {report.wall_min_mm:.2f}/{report.wall_p5_mm:.2f} mm | "
        f"Overhang >60°: {report.overhang_red_gt_60_cm2:.2f} cm^2"
    )
    for chk_name, chk in report.checks.items():
        if chk.status != "PASS":
            print(f"  -> [{chk.status}] {chk_name}: {chk.message}")
    for dm in report.defect_markers:
        print(f"  -> [SoM {dm.id}] {dm.description}")
    if report.slicer and report.slicer.get("status") == "OK":
        s = report.slicer
        print(
            f"  -> [SLICER: {s['slicer_executable']}] "
            f"Time: {s['model_print_time']} (Total: {s['total_estimated_time']}) | "
            f"Filament: {s['filament_used_g']} g | "
            f"Bridges: {s['bridge_feature_blocks']} blocks | "
            f"Floating shells: {s['floating_vertical_shell_blocks']}"
        )
    for gf in report.generated_files:
        print(f"Saved: {gf}")


if __name__ == "__main__":
    main()
