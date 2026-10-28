"""Compatibility path to 2D classification in stage 4."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "4_2d_classification" / "src" / "classification"))
