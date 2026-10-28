"""Compatibility path to 2D segmentation in stage 3."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "3_2d_segmentation" / "src" / "segmentation2d"))
