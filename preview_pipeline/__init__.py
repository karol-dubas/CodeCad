"""Modular CAD & FDM DFM inspection pipeline for build123d."""

from preview_pipeline.analyzer import analyze_model
from preview_pipeline.loader import load_shape, to_repo_relative_posix
from preview_pipeline.models import CadReport, CheckResult, DefectMarker, SectionCutSpec
from preview_pipeline.readme_gen import generate_root_readme
from preview_pipeline.render_2d import render_sheet2_2d_engineering
from preview_pipeline.render_3d import build_edge_polydata, render_sheet1_3d_overview
from preview_pipeline.slicer import run_headless_slicer

__all__ = [
    "CadReport",
    "CheckResult",
    "DefectMarker",
    "SectionCutSpec",
    "analyze_model",
    "build_edge_polydata",
    "generate_root_readme",
    "load_shape",
    "render_sheet1_3d_overview",
    "render_sheet2_2d_engineering",
    "run_headless_slicer",
    "to_repo_relative_posix",
]
