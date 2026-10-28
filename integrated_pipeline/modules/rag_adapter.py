from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import numpy as np


def validate_rag_evidence(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("retrieved_evidence", [])
    citations = result.get("citations", [])
    if not isinstance(evidence, list) or not isinstance(citations, list):
        raise ValueError("RAG response has invalid evidence or citation lists.")
    if not evidence:
        if citations:
            raise ValueError("RAG response has citations without retrieved evidence.")
        return {
            **result,
            "evidence_status": "insufficient_evidence",
            "evidence_warning": "No retrieved evidence supports a case-specific interpretation.",
        }
    evidence_chunk_ids = {
        item["chunk_id"]
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
    }
    evidence_paper_ids = {
        item["paper_id"]
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str)
    }
    for citation in citations:
        if not isinstance(citation, dict) or not (
            (
                isinstance(citation.get("chunk_id"), str)
                and citation["chunk_id"] in evidence_chunk_ids
            )
            or (
                isinstance(citation.get("paper_id"), str)
                and citation["paper_id"] in evidence_paper_ids
            )
        ):
            raise ValueError("RAG response contains a citation not present in retrieved evidence.")
    return {**result, "evidence_status": "retrieved"}


def query_analysis_context(
    base_url: str,
    case_id: str,
    question: str,
    patient_context: dict[str, Any],
) -> dict[str, Any]:
    """Send explicit structured model outputs to RAG; raw MRI data is never included."""
    if not case_id.strip():
        raise ValueError("A non-empty study ID is required.")
    if len(question.strip()) < 3:
        raise ValueError("Enter a question of at least three characters.")
    if patient_context.get("case_id") != case_id:
        raise ValueError("RAG context case_id must match the study ID.")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("RAG API URL must be an absolute HTTP(S) URL.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("Do not embed credentials in the RAG API URL.")
    if parsed_url.scheme != "https" and parsed_url.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Use HTTPS when sending analysis context to a remote RAG service.")
    try:
        import requests
    except ImportError as error:
        raise RuntimeError("RAG API calls require requests.") from error
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/rag/query",
            json={
                "case_id": case_id.strip(),
                "question": question.strip(),
                "patient_context": patient_context,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"RAG service request failed: {error}") from error
    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError("RAG service returned invalid JSON.") from error
    if not isinstance(result, dict) or not isinstance(result.get("answer"), dict):
        raise ValueError("RAG service returned an unexpected response.")
    return validate_rag_evidence(result)


def build_patient_context(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build RAG context only from measured or explicitly labeled model outputs."""
    masks = result.get("region_masks", {})
    whole_tumor = masks.get("WT")
    voxel_spacing = result.get("output_spacing_mm", [])
    tumor_volume_ml = None
    if whole_tumor is not None and len(voxel_spacing) == 3:
        tumor_volume_ml = float(
            whole_tumor.sum() * float(np.prod(voxel_spacing)) / 1000.0
        )
    risk = result.get("risk_mapping")
    risk_stage = result.get("stages", {}).get("surgical risk mapping", {})
    synthseg_provided = risk_stage.get("status") in {"completed", "failed"}
    context: dict[str, Any] = {
        "case_id": case_id,
        "data_type": "RESEARCH_MODEL_OUTPUTS",
        "provenance": {
            "source": "current integrated pipeline run",
            "raw_mri_sent": False,
            "tumor_volume": "estimated from predicted WT mask; not ground truth",
        },
        "tumor_volume_ml": tumor_volume_ml,
        "available_modalities": (
            list(result.get("modalities", []))
            + (["SynthSeg anatomical labels"] if synthseg_provided else [])
        ),
        "missing_modalities": ["DTI FA", "white-matter tract masks"],
        "model_info": {
            "segmentation_3d": result.get("model_3d"),
            "configuration": result.get("model_configuration", {}),
            "ontology": result.get("region_definitions", {}),
        },
        "quality_indicators": {
            "clinical_validation": False,
            "3d_label_provenance_verified": False,
            "2d_mri_domain_compatibility": "unvalidated",
            "risk_mapping_status": risk_stage.get("status", "not recorded"),
        },
    }
    classification = result.get("classification")
    if classification:
        context["classification"] = classification.get("predicted_class")
        context["classification_probability"] = classification.get("probabilities", {}).get(
            classification.get("predicted_class")
        )
        context["classification_provenance"] = {
            "model_checkpoint": classification.get("checkpoint"),
            "training_domain": "BRISC 2D images",
            "application_domain": "MRI axial slice",
            "domain_compatibility_validated": False,
            "interpretation": "experimental prediction only, not a diagnosis",
        }
    if risk:
        risk_values = np.asarray(risk["risk_map"], dtype=np.float32)
        brain = np.asarray(risk.get("brain_mask", risk_values > 0), dtype=bool)
        context["risk_score"] = (
            float(risk_values[brain].mean()) if np.any(brain) else None
        )
        context["risk_provenance"] = {
            "model": "Capstone_MRI_Project V8 research prototype",
            "score_interpretation": "normalized geometric/anatomical composite; not injury probability",
            "active_component_weights": risk.get("active_weights", {}),
            "evidence_status": risk.get("evidence_status", {}),
            "strict_spatial_qc": risk.get("qc", {}),
            "unavailable_inputs": ["DTI FA", "tract masks"],
        }
        context["critical_structures"] = risk.get("structures", [])
        context["candidate_corridors"] = risk.get("corridors", [])
    else:
        if not synthseg_provided:
            context["missing_modalities"].append("registered SynthSeg anatomical labels")
    return context
