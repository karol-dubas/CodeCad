"""Headless OrcaSlicer / Bambu Studio / PrusaSlicer CLI integration and G-code telemetry."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from build123d import Part, export_stl


def _resolve_orca_profile(base_dir: Path, folder: str, name: str) -> dict[str, Any]:
    """Recursively flatten OrcaSlicer / Bambu Studio JSON profile inheritance for CLI mode."""
    p = base_dir / folder / f"{name}.json"
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    parent = data.get("inherits")
    if isinstance(parent, str) and parent:
        merged = _resolve_orca_profile(base_dir, folder, parent)
        merged.update(data)
        return merged
    return data


def _find_slicer_executable(slicer_override: str | None) -> Path | None:
    candidates = (
        [Path(slicer_override)]
        if slicer_override
        else [
            Path(r"C:\Program Files\OrcaSlicer\orca-slicer.exe"),
            Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe"),
            Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"),
        ]
    )
    for cand in candidates:
        if cand.exists():
            return cand
        found = shutil.which(str(cand))
        if found:
            return Path(found)
    return None


def _parse_gcode_telemetry(gcode_text: str, slicer_name: str) -> dict[str, Any]:
    model_time_str: str | None = None
    total_time_str: str | None = None
    total_layers: int | None = None
    filament_mm: float | None = None
    filament_cm3: float | None = None
    filament_g: float | None = None

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
        (r";\s*(?:filament used \[mm\]\s*=|total filament length \[mm\]\s*:)\s*([\d.]+)", "mm"),
        (r";\s*(?:filament used \[cm3\]\s*=|total filament volume \[cm\^3\]\s*:)\s*([\d.]+)", "cm3"),
        (r";\s*(?:filament used \[g\]\s*=|total filament weight \[g\]\s*:)\s*([\d.]+)", "g"),
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
        elif (
            "bridge" in current_feature.lower()
            and (line.startswith("G1 ") or line.startswith("G0 "))
            and " E" in line
            and " E-" not in line
        ):
            bridge_extrusions += 1

    bridge_blocks = sum(
        cnt for feat, cnt in feature_counts.items() if "bridge" in feat.lower()
    )

    return {
        "status": "OK",
        "slicer_executable": slicer_name,
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


def run_headless_slicer(
    shape: Part, slicer_override: str | None = None
) -> dict[str, Any]:
    """Invoke headless slicer CLI on a temporary mesh and parse real G-code metrics."""
    slicer_exe = _find_slicer_executable(slicer_override)
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
                for key in (
                    "extruder_variant_list",
                    "printer_extruder_variant",
                    "print_extruder_variant",
                    "filament_extruder_variant",
                ):
                    flat.pop(key, None)
                (tmp_dir / fname).write_text(json.dumps(flat, indent=2), encoding="utf-8")

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

    return _parse_gcode_telemetry(gcode_text, slicer_exe.name)
