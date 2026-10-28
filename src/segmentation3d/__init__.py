"""Compatibility path to 3D segmentation in stage 2."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "2_3d_segmentation" / "src" / "segmentation3d"))
