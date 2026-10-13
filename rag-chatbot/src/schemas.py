from typing import Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    case_id: str = "DEMO_001"

class PatientContext(BaseModel):
    model_config = {"extra": "allow"}

    case_id: str
    data_type: str
    provenance: dict[str, Any] | None = None
    tumor_type: str | None = None
    tumor_volume_ml: float | None = None
    enhancing_tumor_volume_ml: float | None = None
    classification: str | None = None
    classification_probability: float | None = None
    classification_provenance: dict[str, Any] | None = None
    risk_score: float | None = None
    risk_uncertainty: float | None = None
    risk_provenance: dict[str, Any] | None = None
    critical_structures: list[Any] = Field(default_factory=list)
    candidate_corridors: list[dict[str, Any]] = Field(default_factory=list)
    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] | None = None
    quality_indicators: dict[str, Any] | None = None

class Evidence(BaseModel):
    chunk_id: str
    paper_id: str
    passage: str
    title: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    chunk_id: str | None = None
    paper_id: str | None = None
    title: str | None = None
    status: str = "unverified"
    raw: Any | None = None
    reason: str | None = None

class QueryResponse(BaseModel):
    answer: dict[str, Any]
    citations: list[Citation]
    retrieved_evidence: list[Evidence]
    patient_context: dict[str, Any]
    warnings: list[str]
    verification: dict[str, Any] = Field(default_factory=dict)
    numerical_checks: dict[str, Any] = Field(default_factory=dict)
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
