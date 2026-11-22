"""High-level B-Rep & FDM DFM model analyzer orchestrating geometry metrics and checks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv
from build123d import Part

from preview_pipeline.checks import build_defect_markers, evaluate_checks
from preview_pipeline.geometry import (
    compute_wall_thickness,
    detect_bottom_fillets,
    extract_feature_catalog,
    extract_spatial_clusters,
    select_smart_section_cut,
)
from preview_pipeline.loader import to_repo_relative_posix
from preview_pipeline.models import (
    BUILD_VOL_MM,
    MIN_OVERHANG_CLUSTER_MM2,
    MIN_WALL_CLUSTER_MM2,
    WALL_CRIT_MM,
    WALL_WARN_MM,
    CadReport,
    SectionCutSpec,
)


def analyze_model(
    shape: Part,
    full_poly: pv.PolyData,
    name: str,
    script_path: Path,
) -> tuple[CadReport, pv.PolyData, SectionCutSpec]:
    """Run all geometric, DFM, overhang, stability, wall-thickness, and SoM checks."""
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

    bw, bd, bh = BUILD_VOL_MM
    margins = (bw - sx, bd - sy, bh - sz)
    occ_pct = (sx / bw * 100.0, sy / bd * 100.0, sz / bh * 100.0)
    exceeds_build = any(m < 0.0 for m in margins)
    near_build = any(o >= 95.0 for o in occ_pct)

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
    bottom_fillet_count, bottom_fillet_radii = detect_bottom_fillets(shape, zmin)

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

    poly = cast(
        pv.PolyData,
        full_poly.compute_normals(
            cell_normals=True, point_normals=True, split_vertices=True, feature_angle=35.0
        ),
    )
    sized = cast(pv.PolyData, poly.compute_cell_sizes(length=False, area=True, volume=False))
    poly.cell_data["Area"] = sized.cell_data["Area"]
    normals = np.asarray(poly.cell_data["Normals"], dtype=float)
    centers = np.asarray(poly.cell_centers().points, dtype=float)
    areas = np.asarray(poly.cell_data["Area"], dtype=float)

    nz = normals[:, 2]
    cell_z = centers[:, 2]
    cluster_radius = max(min(sx, sy, sz) * 0.10, 4.5)

    is_bed = (np.abs(cell_z - zmin) < 0.25) & (nz < -0.92)
    raw_amber = (~is_bed) & (nz < -0.7254) & (nz >= -0.8660)
    raw_red = (~is_bed) & (nz < -0.8660)

    red_clusters, is_red = extract_spatial_clusters(
        raw_red, centers, areas, normals, cluster_radius, MIN_OVERHANG_CLUSTER_MM2
    )
    amber_clusters, is_amber = extract_spatial_clusters(
        raw_amber, centers, areas, normals, cluster_radius, MIN_OVERHANG_CLUSTER_MM2
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

    thickness, valid_mask = compute_wall_thickness(poly, max(sx, sy, sz), cluster_radius)
    poly.cell_data["Thickness_mm"] = thickness

    wall_crit_clusters, is_wall_crit = extract_spatial_clusters(
        thickness < WALL_CRIT_MM, centers, areas, normals, cluster_radius, MIN_WALL_CLUSTER_MM2
    )
    wall_warn_clusters, is_wall_warn = extract_spatial_clusters(
        (thickness >= WALL_CRIT_MM) & (thickness < WALL_WARN_MM),
        centers,
        areas,
        normals,
        cluster_radius,
        MIN_WALL_CLUSTER_MM2,
    )

    wall_class = np.zeros(poly.n_cells, dtype=float)
    wall_class[is_wall_warn] = 1.0
    wall_class[is_wall_crit] = 2.0
    poly.cell_data["WallClass"] = wall_class

    overhang_defect = np.zeros(poly.n_cells, dtype=float)
    overhang_defect[is_amber] = 1.0
    overhang_defect[is_red] = 2.0
    poly.cell_data["CutawayClass"] = np.maximum(wall_class, overhang_defect)

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

    defect_markers = build_defect_markers(
        wall_crit_clusters,
        red_clusters,
        wall_warn_clusters,
        amber_clusters,
        wall_amber_pct,
        wall_p5,
        amber_cm2,
        wall_red_pct=wall_red_pct,
        red_cm2=red_cm2,
    )
    section_spec = select_smart_section_cut(
        shape, (xmin, ymin, zmin), (xmax, ymax, zmax), defect_markers
    )
    features = extract_feature_catalog(shape)
    checks = evaluate_checks(
        valid_brep,
        solid_count,
        vol_mm3,
        sx,
        sy,
        sz,
        margins,
        occ_pct,
        exceeds_build,
        near_build,
        zmin,
        bed_at_z0,
        bed_area_cm2,
        bottom_fillet_count,
        bottom_fillet_radii,
        red_cm2,
        amber_cm2,
        wall_min,
        wall_p5,
        wall_p50,
        wall_red_pct,
        wall_amber_pct,
        wall_crit_clusters,
        min_tip_margin_mm,
    )

    statuses = {c.status for c in checks.values()}
    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    report = CadReport(
        name=name,
        script_path=to_repo_relative_posix(script_path),
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
