"""Adapters and utilities for the integrated MRI workflow."""
"""Cross-stage adapter modules used by the optional Streamlit app."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "7_gradcam"))
