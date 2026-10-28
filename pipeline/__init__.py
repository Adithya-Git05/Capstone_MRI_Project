"""Compatibility path to surgical-risk code in stage 5."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "5_surgical_risk_mapping" / "pipeline"))
