"""Compatibility path to dataset loaders in stage 1."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "1_data_and_preprocessing" / "src" / "data"))
