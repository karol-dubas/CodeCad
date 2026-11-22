"""CLI entry point for the CodeCad visual & symbolic FDM DFM inspection pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import pyvista as pv
import pyvista_cad

from preview_pipeline import (
    analyze_model,
    build_edge_polydata,
    generate_root_readme,
    load_shape,
    render_sheet1_3d_overview,
    render_sheet2_2d_engineering,
    run_headless_slicer,
    to_repo_relative_posix,
)


def _cleanup_legacy_sheets(out_dir: Path, stem: str) -> None:
    for suffix in ("diagnostics", "surface_audit", "blueprint", "sections", "dfm"):
        legacy_file = out_dir / f"{stem}_{suffix}.png"
        if legacy_file.exists():
            with contextlib.suppress(OSError):
                legacy_file.unlink()


def main() -> None:
    # Treat repository & CAD script runtime warnings as errors (ignore third-party noise if any)
    warnings.filterwarnings("error", category=DeprecationWarning, module=r"^preview_pipeline")

    parser = argparse.ArgumentParser(
        description="Visual & Symbolic CAD + FDM DFM Inspection Pipeline."
    )
    parser.add_argument("script", help="Path to build123d script")
    parser.add_argument("--var", default=None, help="Variable name to render")
    parser.add_argument("--out", default=None, help="Output main PNG path")
    parser.add_argument(
        "--slice",
        action="store_true",
        help="Invoke headless slicer CLI and record G-code telemetry in report.json",
    )
    parser.add_argument(
        "--slicer-path",
        default=None,
        help="Optional explicit path to headless slicer executable",
    )
    parser.add_argument(
        "--allow-warn",
        action="store_true",
        help="Exit 0 even if report.json has WARN status (default treats WARN as error)",
    )
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    shape = load_shape(script_path, args.var)
    part_name = f"{script_path.stem}" + (f" ({args.var})" if args.var else "")

    out_path = (
        Path(args.out)
        if args.out is not None
        else script_path.parent / ".preview" / "preview.png"
    )
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = out_path.stem
    engineering_path = out_dir / f"{stem}_engineering.png"
    report_path = out_dir / "report.json"
    _cleanup_legacy_sheets(out_dir, stem)

    pyvista_cad_dyn: Any = pyvista_cad
    raw_mesh: Any = pyvista_cad_dyn.from_build123d(
        shape, linear_deflection=0.025, angular_deflection=0.10
    )
    full_poly = (
        raw_mesh
        if isinstance(raw_mesh, pv.PolyData)
        else cast(pv.PolyData, pyvista_cad_dyn.flatten_to_cad_polydata(raw_mesh))
    )
    edge_mesh = build_edge_polydata(shape)

    report, analyzed_poly, section_spec = analyze_model(shape, full_poly, part_name, script_path)
    if args.slice:
        report.slicer = run_headless_slicer(shape, slicer_override=args.slicer_path)

    render_sheet1_3d_overview(shape, analyzed_poly, edge_mesh, report, section_spec, out_path)
    report.generated_files.append(to_repo_relative_posix(out_path))

    render_sheet2_2d_engineering(shape, report, section_spec, engineering_path)
    report.generated_files.append(to_repo_relative_posix(engineering_path))

    report.generated_files.append(to_repo_relative_posix(report_path))
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme_path = generate_root_readme()
    report.generated_files.append(to_repo_relative_posix(readme_path))

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

    if report.overall_status == "FAIL" or (
        report.overall_status == "WARN" and not args.allow_warn
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
