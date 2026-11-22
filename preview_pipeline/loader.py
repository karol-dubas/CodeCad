"""Headless build123d script loader and lossless PNG whitespace cropper."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any

import build123d as b3d
from build123d import Part, Solid
from PIL import Image, ImageChops


def _coerce_to_part(candidate: Any, script_path: Path) -> Part:
    if hasattr(candidate, "_obj"):
        candidate = candidate._obj
    if isinstance(candidate, Part):
        return candidate
    if isinstance(candidate, Solid):
        return Part(candidate)
    raise TypeError(f"Selected object in {script_path} is not a Part/Solid: {type(candidate)}")


def load_shape(script_path: Path, var_name: str | None) -> Part:
    """Execute a build123d script with viewer and file-export calls stubbed out."""
    import ocp_vscode

    shown: list[Any] = []

    def capture_show(*args: Any, **kwargs: Any) -> None:
        del kwargs
        for arg in args:
            if isinstance(arg, (Part, Solid)) or hasattr(arg, "_obj"):
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
            if not k.startswith("_") and (isinstance(v, (Part, Solid)) or hasattr(v, "_obj"))
        ]
        if not candidates:
            raise ValueError(f"No build123d Part/Solid found in {script_path}")
        candidate = candidates[-1]

    return _coerce_to_part(candidate, script_path)


def crop_whitespace_png(image_path: Path, pad_px: int = 18) -> None:
    """Losslessly trim excess outer background margins to maximize VLM pixel density."""
    if not image_path.exists():
        return
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        corner_pixel = rgb.getpixel((0, 0))
        if not isinstance(corner_pixel, tuple):
            return
        bg = Image.new("RGB", rgb.size, corner_pixel)
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
