"""Single Source of Truth (SSoT) constants, thresholds, and report dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from build123d import Plane

# Printer & build volume constants (SSoT)
PRINTER_NAME = "Bambu Lab P1S"
BUILD_VOL_MM: tuple[float, float, float] = (256.0, 256.0, 256.0)

# Global semantic DFM & CAD color palette (SSoT)
COLOR_SAFE_CAD = "#5B7C99"  # Safe <= 45 deg overhang / normal CAD surface
COLOR_OK_GRAY = "#64748B"  # Safe > 1.6 mm wall thickness
COLOR_BED_BLUE = "#0284C7"  # Z = Z_min build-plate contact & dimension accents
COLOR_CAP_CYAN = "#38BDF8"  # Cross-section cut cap fill
COLOR_WARN_AMBER = "#F59E0B"  # Warning: 45-60 deg overhang or 0.8-1.6 mm wall
COLOR_CRIT_RED = "#EF4444"  # Critical: > 60 deg overhang/bridge or < 0.8 mm wall
COLOR_BADGE_RED = "#DC2626"  # SoM critical badge background
COLOR_BADGE_AMBER = "#D97706"  # SoM warning badge background
COLOR_EDGE_DARK = "#0F172A"  # Crisp B-Rep topological edges
COLOR_GHOST_GRAY = "#94A3B8"  # Ghost cutaway context & centerlines
COLOR_BG_LIGHT = "#F8FAFC"  # High-contrast clean sheet background

# Absolute FDM wall thickness thresholds [mm] (0.4 mm nozzle)
WALL_CRIT_MM = 0.8  # < 0.8 mm (< 2 nozzle perimeters) -> RED
WALL_WARN_MM = 1.6  # 0.8 - 1.6 mm (2 to 4 perimeters) -> AMBER, > 1.6 mm -> OK

# Minimum cluster surface area [mm^2] to suppress single-triangle corner/tessellation noise
MIN_WALL_CLUSTER_MM2 = 4.0
MIN_OVERHANG_CLUSTER_MM2 = 6.0


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
    reason: str
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
    valid_brep: bool
    solid_count: int
    face_count: int
    edge_count: int
    volume_mm3: float
    volume_cm3: float
    surface_area_mm2: float
    surface_area_cm2: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    build_volume_mm: tuple[float, float, float]
    build_volume_margin_mm: tuple[float, float, float]
    build_volume_occupancy_pct: tuple[float, float, float]
    near_or_exceeds_build_volume: bool
    z_min_mm: float
    bed_contact_area_mm2: float
    bed_contact_area_cm2: float
    bed_flatness_ok: bool
    bottom_fillet_faces_count: int
    bottom_fillet_radii_mm: list[float]
    overhang_amber_45_60_mm2: float
    overhang_amber_45_60_cm2: float
    overhang_red_gt_60_mm2: float
    overhang_red_gt_60_cm2: float
    wall_min_mm: float
    wall_p5_mm: float
    wall_p10_mm: float
    wall_p50_mm: float
    wall_p90_mm: float
    wall_max_mm: float
    wall_red_lt_08_pct: float
    wall_amber_08_16_pct: float
    wall_ok_gt_16_pct: float
    center_of_mass_mm: tuple[float, float, float]
    support_footprint_bounds_xy: tuple[float, float, float, float]
    stability_tip_margin_mm: float
    defect_markers: list[DefectMarker]
    section_cut: dict[str, Any]
    features: dict[str, Any]
    checks: dict[str, CheckResult]
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
                "margin_to_limits_mm": [round(x, 2) for x in self.build_volume_margin_mm],
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
                "footprint_bounds_xy_mm": [round(x, 2) for x in self.support_footprint_bounds_xy],
                "min_tipping_margin_mm": round(self.stability_tip_margin_mm, 2),
            },
            "features": self.features,
            "slicer": self.slicer,
            "generated_files": self.generated_files,
        }
