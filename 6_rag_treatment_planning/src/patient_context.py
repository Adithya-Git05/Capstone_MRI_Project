import json
from typing import Any
from .config import PATIENT_PATH
from .schemas import PatientContext


def load_patient_context(case_id: str = "DEMO_001", path=PATIENT_PATH) -> dict[str, Any]:
    patient = json.loads(path.read_text(encoding="utf-8"))
    if patient.get("case_id") != case_id:
        raise KeyError(f"Unknown case_id: {case_id}")
    return PatientContext.model_validate(patient).model_dump(exclude_none=True)


def patient_warnings(patient: dict[str, Any]) -> list[str]:
    warnings = []
    if patient.get("data_type", "").upper().startswith("SYNTHETIC"):
        warnings.append(
            "SYNTHETIC/DEMO PATIENT HISTORY: this fictional history is not an MRI finding."
        )
    if patient.get("missing_modalities"):
        warnings.append("Missing modalities: " + ", ".join(patient["missing_modalities"]) + ".")
    return warnings
