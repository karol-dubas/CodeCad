"""Geometric ray-casting, spatial defect clustering, feature extraction, and section selection."""

from __future__ import annotations

import math
from typing import Any

import build123d as b3d
import numpy as np
import pyvista as pv
import vtk
from build123d import GeomType, Part, Plane

from preview_pipeline.models import (
    MIN_WALL_CLUSTER_MM2,
    WALL_WARN_MM,
    DefectMarker,
    SectionCutSpec,
)


def extract_spatial_clusters(
    mask: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    cluster_radius_mm: float,
    min_area_mm2: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Group flagged mesh cells into connected spatial clusters and filter micro-noise."""
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

    clusters.sort(
        key=lambda c: (
            round(c["area_mm2"] * 2.0) / 2.0,
            c["centroid_mm"][0] - c["centroid_mm"][1] + c["centroid_mm"][2],
        ),
        reverse=True,
    )
    return clusters, filtered_mask


def compute_wall_thickness(
    poly: pv.PolyData,
    max_ray: float,
    cluster_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true opposing-wall thickness [mm] via inward VTK OBBTree ray-casting."""
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(poly)
    obb.BuildLocator()

    centers = np.asarray(poly.cell_centers().points, dtype=float)
    normals = np.asarray(poly.cell_data["Normals"], dtype=float)
    areas = np.asarray(poly.cell_data["Area"], dtype=float)
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
            if float(np.dot(n_vec, n_hit)) < -0.75:
                hp = np.array(p_hit.GetPoint(h_idx), dtype=float)
                dist = float(np.linalg.norm(hp - centers[i]))
                if dist > 0.22:
                    thickness[i] = dist
                    break

    valid_mask = ~np.isnan(thickness)
    valid_vals = thickness[valid_mask]
    fallback = float(np.median(valid_vals)) if len(valid_vals) > 0 else 10.0
    thickness_filled = np.where(valid_mask, thickness, fallback)

    raw_thin_mask = thickness_filled < WALL_WARN_MM
    if np.any(raw_thin_mask):
        _, valid_thin_mask = extract_spatial_clusters(
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


def detect_bottom_fillets(shape: Part, zmin: float) -> tuple[int, list[float]]:
    """Detect curved fillet faces touching the build plate at Z = zmin."""
    bottom_fillet_count = 0
    radii: set[float] = set()
    tol = 0.08

    for f in shape.faces():
        f_bb = f.bounding_box()
        if abs(f_bb.min.Z - zmin) > tol:
            continue
        height_z = float(f_bb.max.Z - f_bb.min.Z)
        if 0.15 <= height_z <= 15.0 and f.geom_type in (
            GeomType.CYLINDER,
            GeomType.TORUS,
            GeomType.BSPLINE,
        ):
            try:
                n_vec = f.normal_at(f.position_at(0.5, 0.5))
                if n_vec.Z < -0.15:
                    bottom_fillet_count += 1
                    f_radius = getattr(f, "radius", None)
                    if f_radius is not None:
                        radii.add(round(float(f_radius), 2))
                    else:
                        radii.add(round(height_z, 2))
            except Exception:
                pass

    return bottom_fillet_count, sorted(radii)


def extract_feature_catalog(shape: Part) -> dict[str, Any]:
    """Extract cylindrical hole/shaft diameters, fillet radii, and planar Z levels."""
    cyl_holes_diameters: dict[float, int] = {}
    fillet_radii: dict[float, int] = {}

    for f in shape.faces():
        if f.geom_type == GeomType.CYLINDER:
            f_radius = getattr(f, "radius", None)
            if f_radius is None:
                continue
            r = round(float(f_radius), 2)
            circ_edges = [e for e in f.edges() if e.geom_type == GeomType.CIRCLE]
            is_full_circle = any(abs(e.length - 2.0 * math.pi * r) < 0.25 for e in circ_edges)
            if is_full_circle:
                d = round(2.0 * r, 2)
                cyl_holes_diameters[d] = cyl_holes_diameters.get(d, 0) + 1
            else:
                fillet_radii[r] = fillet_radii.get(r, 0) + 1

    edge_radii: set[float] = set()
    for e in shape.edges():
        if e.geom_type == GeomType.CIRCLE:
            e_radius = getattr(e, "radius", None)
            if e_radius is not None:
                edge_radii.add(round(float(e_radius), 2))

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


def _clean_zero(val: float) -> float:
    return 0.0 if abs(val) < 1e-4 else float(val)


def _score_section_candidate(
    shape: Part,
    plane: Plane,
    total_height: float,
    is_center: bool,
) -> tuple[int, int, int, float, float]:
    """Score a vertical section plane; prefer full-height single-component cuts."""
    try:
        sec = b3d.section(shape, section_by=plane)
        faces = sec.faces() if hasattr(sec, "faces") else []
        total_area = float(sec.area)
        if not faces or total_area < 1.0:
            return (-1, 0, 0, 0.0, 0.0)
        sec_bb = sec.bounding_box()
        z_span_ratio = float(sec_bb.max.Z - sec_bb.min.Z) / max(total_height, 1e-3)
        full_height = 1 if z_span_ratio >= 0.90 else 0
        single_piece = 1 if len(faces) == 1 else 0
        center_bonus = 1 if is_center else 0
        max_face_area = max(float(f.area) for f in faces)
        cohesion = max_face_area / max(total_area, 1e-6)
        return (full_height, single_piece, center_bonus, cohesion, total_area)
    except Exception:
        return (-1, 0, 0, 0.0, 0.0)


def select_smart_section_cut(
    shape: Part,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    defect_markers: list[DefectMarker],
) -> SectionCutSpec:
    """Choose optimal vertical section plane A-A for 3D cutaway and 2D hatched section."""
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    cx = _clean_zero((xmin + xmax) / 2.0)
    cy = _clean_zero((ymin + ymax) / 2.0)
    cz = _clean_zero((zmin + zmax) / 2.0)
    sx = max(xmax - xmin, 1.0)
    sy = max(ymax - ymin, 1.0)
    sz = max(zmax - zmin, 1.0)

    if defect_markers:
        d1 = defect_markers[0]
        dx, dy, _ = d1.centroid_mm
        nx, ny, _ = d1.dominant_normal
        dx_clamped = _clean_zero(float(np.clip(dx, xmin + 0.06 * sx, xmax - 0.06 * sx)))
        dy_clamped = _clean_zero(float(np.clip(dy, ymin + 0.06 * sy, ymax - 0.06 * sy)))

        if abs(nx) < 0.25 and abs(ny) < 0.25:
            use_x_cut = (abs(dx - cx) / sx) >= (abs(dy - cy) / sy)
        else:
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

    candidates: list[tuple[str, float, str, bool]] = [
        ("Y", cy, "Center Plane (Y = 0)" if abs(cy) < 1e-2 else f"Mid-Plane (Y = {cy:.1f})", True),
        ("X", cx, "Center Plane (X = 0)" if abs(cx) < 1e-2 else f"Mid-Plane (X = {cx:.1f})", True),
    ]
    for f in shape.faces():
        f_bb = f.bounding_box()
        if zmax - 1.0 <= f_bb.max.Z and float(f.area) > 4.0:
            fc = f.center()
            fx = _clean_zero(float(np.clip(fc.X, xmin + 0.06 * sx, xmax - 0.06 * sx)))
            fy = _clean_zero(float(np.clip(fc.Y, ymin + 0.06 * sy, ymax - 0.06 * sy)))
            candidates.append(("X", fx, f"Feature Plane (X = {fx:.1f})", abs(fx - cx) < 1e-2))
            candidates.append(("Y", fy, f"Feature Plane (Y = {fy:.1f})", abs(fy - cy) < 1e-2))

    best_score: tuple[int, int, int, float, float] = (-2, 0, 0, 0.0, 0.0)
    best_cand = candidates[0] if sx >= sy else candidates[1]
    seen: set[tuple[str, float]] = set()
    for axis_name, coord, reason, is_center in candidates:
        key = (axis_name, round(coord, 1))
        if key in seen:
            continue
        seen.add(key)
        plane = Plane.YZ.offset(coord) if axis_name == "X" else Plane.XZ.offset(coord)
        score = _score_section_candidate(shape, plane, sz, is_center)
        if score > best_score:
            best_score = score
            best_cand = (axis_name, coord, reason, is_center)

    axis_name, coord, reason, _ = best_cand
    if axis_name == "X":
        return SectionCutSpec(
            axis_name="X",
            coord_mm=coord,
            reason=reason,
            plane=Plane.YZ.offset(coord),
            normal=(1.0, 0.0, 0.0),
            origin=(coord, cy, cz),
            u_label="Y",
            v_label="Z",
        )
    return SectionCutSpec(
        axis_name="Y",
        coord_mm=coord,
        reason=reason,
        plane=Plane.XZ.offset(coord),
        normal=(0.0, 1.0, 0.0),
        origin=(cx, coord, cz),
        u_label="X",
        v_label="Z",
    )
