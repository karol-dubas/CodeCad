"""Deterministic DFM PASS/WARN/FAIL evaluators and Set-of-Mark defect marker builders."""

from __future__ import annotations

from typing import Any

from preview_pipeline.models import (
    BUILD_VOL_MM,
    PRINTER_NAME,
    WALL_CRIT_MM,
    WALL_WARN_MM,
    CheckResult,
    DefectMarker,
)


def build_defect_markers(
    wall_crit_clusters: list[dict[str, Any]],
    red_clusters: list[dict[str, Any]],
    wall_warn_clusters: list[dict[str, Any]],
    amber_clusters: list[dict[str, Any]],
    wall_amber_pct: float,
    wall_p5: float,
    amber_cm2: float,
) -> list[DefectMarker]:
    markers: list[DefectMarker] = []

    def _add(
        category: str,
        severity: str,
        short_label: str,
        cluster: dict[str, Any],
        desc_prefix: str,
    ) -> None:
        if len(markers) >= 3:
            return
        m_id = f"D{len(markers) + 1}"
        cx_m, cy_m, cz_m = cluster["centroid_mm"]
        markers.append(
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
        _add("thin_wall_critical", "WARN", "Wall <0.8mm", cl, "Critical thin wall <0.8 mm")
    for cl in red_clusters[:2]:
        if cl["area_cm2"] >= 0.10:
            _add("overhang_critical", "WARN", "Overhang >60°", cl, "Critical overhang/bridge >60 deg")
    if not markers:
        for cl in wall_warn_clusters[:1]:
            if wall_amber_pct > 10.0 or wall_p5 < WALL_WARN_MM:
                _add(
                    "thin_wall_moderate",
                    "WARN",
                    "Wall 0.8-1.6mm",
                    cl,
                    "Moderate thin wall 0.8-1.6 mm",
                )
        for cl in amber_clusters[:1]:
            if amber_cm2 > 5.0:
                _add("overhang_moderate", "WARN", "Overhang 45-60°", cl, "Moderate overhang 45-60 deg")
    return markers


def evaluate_checks(
    valid_brep: bool,
    solid_count: int,
    vol_mm3: float,
    sx: float,
    sy: float,
    sz: float,
    margins: tuple[float, float, float],
    occ_pct: tuple[float, float, float],
    exceeds_build: bool,
    near_build: bool,
    zmin: float,
    bed_at_z0: bool,
    bed_area_cm2: float,
    bottom_fillet_count: int,
    bottom_fillet_radii: list[float],
    red_cm2: float,
    amber_cm2: float,
    wall_min: float,
    wall_p5: float,
    wall_p50: float,
    wall_red_pct: float,
    wall_amber_pct: float,
    wall_crit_clusters: list[dict[str, Any]],
    min_tip_margin_mm: float,
) -> dict[str, CheckResult]:
    bw, bd, bh = BUILD_VOL_MM
    checks: dict[str, CheckResult] = {}

    if not valid_brep or solid_count != 1 or vol_mm3 <= 1e-3:
        checks["geometry_sanity"] = CheckResult(
            status="FAIL",
            message=(
                f"Invalid geometry: valid_brep={valid_brep}, "
                f"solid_count={solid_count}, volume={vol_mm3:.2f} mm^3"
            ),
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
            metrics={"valid_brep": True, "solid_count": 1, "volume_mm3": round(vol_mm3, 2)},
        )

    if exceeds_build:
        checks["build_volume"] = CheckResult(
            status="FAIL",
            message=(
                f"Part ({sx:.1f}x{sy:.1f}x{sz:.1f} mm) exceeds {PRINTER_NAME} "
                f"build volume ({bw:.0f}x{bd:.0f}x{bh:.0f} mm)."
            ),
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "margins_mm": [round(m, 2) for m in margins],
            },
        )
    elif near_build:
        checks["build_volume"] = CheckResult(
            status="WARN",
            message=(
                f"Part is within 5% of {PRINTER_NAME} build volume limits "
                f"(max occupancy {max(occ_pct):.1f}%)."
            ),
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "occupancy_pct": [round(o, 1) for o in occ_pct],
            },
        )
    else:
        checks["build_volume"] = CheckResult(
            status="PASS",
            message=(
                f"Fits comfortably inside {PRINTER_NAME} build volume "
                f"(max occupancy {max(occ_pct):.1f}%)."
            ),
            metrics={
                "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                "occupancy_pct": [round(o, 1) for o in occ_pct],
            },
        )

    if bed_area_cm2 < 0.25:
        checks["bed_adhesion"] = CheckResult(
            status="FAIL",
            message=f"Insufficient bed contact area ({bed_area_cm2:.2f} cm^2) at Z_min={zmin:.2f} mm.",
            metrics={"contact_area_cm2": round(bed_area_cm2, 3), "z_min_mm": round(zmin, 3)},
        )
    elif not bed_at_z0 or bed_area_cm2 < 1.0:
        checks["bed_adhesion"] = CheckResult(
            status="WARN",
            message=f"Small bed contact area ({bed_area_cm2:.2f} cm^2) or bottom Z_min={zmin:.2f} mm != 0.0.",
            metrics={"contact_area_cm2": round(bed_area_cm2, 3), "z_min_mm": round(zmin, 3)},
        )
    else:
        checks["bed_adhesion"] = CheckResult(
            status="PASS",
            message=f"Flat bottom at Z={zmin:.2f} mm with {bed_area_cm2:.2f} cm^2 bed contact area.",
            metrics={"contact_area_cm2": round(bed_area_cm2, 3), "z_min_mm": round(zmin, 3)},
        )

    if bottom_fillet_count > 0:
        checks["build_plate_fillets"] = CheckResult(
            status="WARN",
            message=(
                f"Detected {bottom_fillet_count} curved fillet face(s) touching Z_min "
                f"(radii: {bottom_fillet_radii} mm). Use 45-deg chamfers on the build plate."
            ),
            metrics={"fillet_face_count": bottom_fillet_count, "radii_mm": bottom_fillet_radii},
        )
    else:
        checks["build_plate_fillets"] = CheckResult(
            status="PASS",
            message="No overhanging fillets detected on the Z_min build plate interface.",
            metrics={"fillet_face_count": 0},
        )

    if red_cm2 > 2.5:
        checks["overhangs"] = CheckResult(
            status="WARN",
            message=f"Critical overhangs/bridges >60 deg detected ({red_cm2:.2f} cm^2 red, {amber_cm2:.2f} cm^2 amber).",
            metrics={"red_gt_60_cm2": round(red_cm2, 3), "amber_45_60_cm2": round(amber_cm2, 3)},
        )
    elif red_cm2 > 0.15 or amber_cm2 > 5.0:
        checks["overhangs"] = CheckResult(
            status="WARN",
            message=f"Moderate overhangs/bridges present ({red_cm2:.2f} cm^2 >60 deg, {amber_cm2:.2f} cm^2 45-60 deg).",
            metrics={"red_gt_60_cm2": round(red_cm2, 3), "amber_45_60_cm2": round(amber_cm2, 3)},
        )
    else:
        checks["overhangs"] = CheckResult(
            status="PASS",
            message=f"Support-free FDM geometry ({red_cm2:.2f} cm^2 >60 deg, {amber_cm2:.2f} cm^2 45-60 deg).",
            metrics={"red_gt_60_cm2": round(red_cm2, 3), "amber_45_60_cm2": round(amber_cm2, 3)},
        )

    if wall_p5 < WALL_CRIT_MM or wall_red_pct > 2.0 or len(wall_crit_clusters) > 0:
        checks["wall_thickness"] = CheckResult(
            status="WARN",
            message=(
                f"Thin walls <{WALL_CRIT_MM} mm detected "
                f"(min={wall_min:.2f} mm, P5={wall_p5:.2f} mm, {wall_red_pct:.1f}% area <0.8 mm)."
            ),
            metrics={
                "min_mm": round(wall_min, 2),
                "p5_mm": round(wall_p5, 2),
                "red_area_pct": round(wall_red_pct, 2),
            },
        )
    elif wall_p5 < WALL_WARN_MM or wall_amber_pct > 10.0:
        checks["wall_thickness"] = CheckResult(
            status="WARN",
            message=(
                f"Walls in 0.8-1.6 mm warning range "
                f"(P5={wall_p5:.2f} mm, {wall_amber_pct:.1f}% area in 0.8-1.6 mm)."
            ),
            metrics={"p5_mm": round(wall_p5, 2), "amber_area_pct": round(wall_amber_pct, 2)},
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

    return checks
