from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


RAG_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "6_rag_treatment_planning"
RAG_CHUNKS_PATH = RAG_PROJECT_ROOT / "knowledge_base" / "chunks.json"
RAG_VECTOR_MANIFEST = (
    RAG_PROJECT_ROOT / "knowledge_base" / "vector_db" / "manifest.json"
)


def rag_readiness(api_url: str = "http://127.0.0.1:8000") -> dict[str, Any]:
    """Report local RAG prerequisites and API health without downloading assets."""
    missing: list[str] = []
    setup: list[str] = []
    if not RAG_CHUNKS_PATH.is_file():
        missing.append(f"Knowledge-base chunks are missing: {RAG_CHUNKS_PATH}")
        setup.extend(
            [
                "From 6_rag_treatment_planning, download the approved literature and "
                "build the index: python scripts\\download_literature.py",
                "Then run: python scripts\\build_knowledge_base.py",
            ]
        )
    if not RAG_VECTOR_MANIFEST.is_file():
        missing.append(f"Vector index manifest is missing: {RAG_VECTOR_MANIFEST}")
        if not setup:
            setup.append(
                "From 6_rag_treatment_planning, run: "
                "python scripts\\build_knowledge_base.py"
            )

    required_modules = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "rank_bm25": "rank-bm25",
        "sentence_transformers": "sentence-transformers",
        "dotenv": "python-dotenv",
        "httpx": "httpx",
    }
    missing_modules = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        missing.append(
            "RAG Python dependencies are not installed in the active environment: "
            + ", ".join(missing_modules)
        )
        setup.append(
            "From 6_rag_treatment_planning, install them with: "
            "python -m pip install -r requirements.txt"
        )

    try:
        from dotenv import dotenv_values

        rag_settings = dotenv_values(RAG_PROJECT_ROOT / ".env")
    except ImportError:
        rag_settings = {}
    gemini_api_key = os.environ.get(
        "GEMINI_API_KEY", str(rag_settings.get("GEMINI_API_KEY") or "")
    )
    if gemini_api_key.strip().lower() in {"replace-with-your-key", "your-api-key"}:
        gemini_api_key = ""
    gemini_model = os.environ.get(
        "GEMINI_MODEL", str(rag_settings.get("GEMINI_MODEL") or "gemini-3.5-flash")
    )
    try:
        import requests
    except ImportError:
        missing.append("The Streamlit app requires requests to query the RAG API.")
        setup.append("Install the app dependencies with: python -m pip install -e '.[integrated]'")
        requests = None

    api_parts = urlsplit(api_url)
    if (
        api_parts.scheme not in {"http", "https"}
        or not api_parts.hostname
        or api_parts.username
        or api_parts.password
        or (
            api_parts.scheme != "https"
            and api_parts.hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        missing.append(
            "RAG API URL must be a valid HTTPS URL or a local HTTP URL; no health "
            "request was sent."
        )
        setup.append(
            "Use the local default http://127.0.0.1:8000 or an authorized HTTPS RAG endpoint."
        )

    service_status = "unavailable"
    health: dict[str, Any] | None = None
    if requests is not None and not any(
        "RAG API URL must be" in item for item in missing
    ):
        try:
            response = requests.get(f"{api_url.rstrip('/')}/api/health", timeout=2)
            response.raise_for_status()
            health_response = response.json()
            health = health_response if isinstance(health_response, dict) else None
        except (requests.RequestException, ValueError):
            health = None
        if health is not None:
            service_status = "available"
            if not health.get("knowledge_base_exists"):
                missing.append(
                    "The running RAG API reports that its knowledge-base chunks are missing."
                )
        else:
            missing.append(
                f"RAG API is not reachable at {api_url.rstrip('/')}. Start it from "
                "6_rag_treatment_planning with: "
                "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
            )
            setup.append(
                "First install the RAG requirements and build the knowledge base using "
                "the instructions in 6_rag_treatment_planning\\README.md."
            )

    gemini_status = "configured" if gemini_api_key else "missing API key"
    if not gemini_api_key:
        missing.append(
            "Gemini API key is not configured; set GEMINI_API_KEY in "
            "6_rag_treatment_planning\\.env."
        )
        setup.append(
            "Copy .env.example to .env in 6_rag_treatment_planning and set "
            "GEMINI_API_KEY without committing the key."
        )
    if not RAG_PROJECT_ROOT.joinpath(".env").is_file():
        setup.append(
            "RAG settings are loaded by the API from 6_rag_treatment_planning\\.env. "
            "Copy .env.example to .env and configure GEMINI_API_KEY and retrieval models."
        )
    return {
        "ready": (
            service_status == "available"
            and bool(health and health.get("knowledge_base_exists"))
            and gemini_status == "configured"
            and not missing_modules
        ),
        "api_status": service_status,
        "gemini_status": gemini_status,
        "gemini_model": gemini_model,
        "chunks_path": str(RAG_CHUNKS_PATH),
        "chunks_available": RAG_CHUNKS_PATH.is_file(),
        "vector_index_manifest_available": RAG_VECTOR_MANIFEST.is_file(),
        "missing": missing,
        "setup": list(dict.fromkeys(setup)),
    }


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


def build_patient_context(
    case_id: str,
    result: dict[str, Any],
    volume_result: dict[str, Any] | None = None,
    modality_inputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build RAG context only from measured or explicitly labeled model outputs."""
    if volume_result:
        result = {
            **result,
            "region_masks": volume_result.get("region_masks", result.get("region_masks", {})),
            "risk_mapping": volume_result.get("risk_mapping", result.get("risk_mapping")),
            "segmentation3d_metadata": volume_result.get(
                "segmentation3d_metadata", result.get("segmentation3d_metadata", {})
            ),
            "model_3d": volume_result.get("model_3d", result.get("model_3d")),
            "modalities": volume_result.get("modalities", result.get("modalities", [])),
            "output_spacing_mm": volume_result.get(
                "output_spacing_mm", result.get("output_spacing_mm", [])
            ),
            "stages": {
                **result.get("stages", {}),
                **volume_result.get("stages", {}),
            },
        }
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
        "patient_label": "Patient 001",
        "data_type": "MODEL_INFERENCE_WITH_SYNTHETIC_HISTORY",
        "patient_history": (
            "Fictional demonstration history: intermittent headaches. No real medical "
            "history or verified clinical findings are supplied."
        ),
        "patient_history_provenance": (
            "Synthetic history for demonstration only. Imaging findings and model "
            "outputs come only from the active MRI analysis."
        ),
        "provenance": {
            "source": "current integrated pipeline run",
            "raw_mri_sent": False,
            "model_findings_source": "active MRI inference results only",
            "synthetic_history_is_not_an_inference": True,
            "tumor_volume": (
                "estimated from a predicted WT mask; not ground truth"
                if whole_tumor is not None
                else "unavailable; no 3D tumor mask was produced"
            ),
        },
        "tumor_volume_ml": tumor_volume_ml,
        "available_modalities": (
            list(result.get("modalities", [])) or list(modality_inputs or {})
        )
        + (["SynthSeg anatomical labels"] if synthseg_provided else []),
        "modality_inputs": modality_inputs or {},
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
    input_info = result.get("input", {})
    if input_info:
        context["input_summary"] = {
            "modality": input_info.get("modality"),
            "shape": input_info.get("shape"),
            "slice_index": input_info.get("slice_index"),
        }
    segmentation3d = result.get("segmentation3d_metadata")
    if segmentation3d:
        context["segmentation_3d"] = {
            "model": result.get("model_3d"),
            "region_order": segmentation3d.get("region_order"),
            "mask_voxel_count": segmentation3d.get("mask_voxel_count"),
            "prediction_threshold": segmentation3d.get("prediction_threshold"),
            "clinical_performance": "unvalidated",
        }
    segmentation_2d = result.get("segmentation")
    if segmentation_2d:
        context["segmentation_2d"] = {
            key: segmentation_2d.get(key)
            for key in (
                "status",
                "architecture",
                "checkpoint",
                "classes",
                "foreground_area",
                "inference_time_seconds",
                "limitation",
            )
        }
    classification = result.get("classification")
    if classification:
        context["classification"] = classification.get("predicted_class")
        context["classification_results"] = {
            "status": classification.get("status"),
            "scores": classification.get("scores"),
            "probabilities": classification.get("probabilities"),
        }
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
    context["analysis_results_text"] = _analysis_results_text(context)
    return context


def _analysis_results_text(context: dict[str, Any]) -> str:
    """Render only supplied case context as readable text for the language model."""
    sections = (
        ("PATIENT HISTORY (synthetic demonstration data)", ("patient_label", "patient_history")),
        ("MRI INPUTS", ("available_modalities", "modality_inputs", "input_summary")),
        ("2D SEGMENTATION INFERENCE", ("segmentation_2d",)),
        ("3D SEGMENTATION INFERENCE", ("segmentation_3d",)),
        (
            "CLASSIFICATION INFERENCE",
            ("classification", "classification_results", "classification_provenance"),
        ),
        ("SURGICAL RISK MAPPING INFERENCE", ("risk_score", "risk_provenance", "critical_structures")),
        ("SURGICAL CORRIDOR INFERENCE", ("candidate_corridors",)),
        ("MISSING INPUTS / LIMITATIONS", ("missing_modalities", "quality_indicators")),
    )
    lines: list[str] = []
    for heading, keys in sections:
        values = [(key, context[key]) for key in keys if key in context and context[key] not in (None, {}, [], "")]
        if not values:
            continue
        lines.append(f"{heading}:")
        for key, value in values:
            serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
            lines.append(f"- {key.replace('_', ' ')}: {serialized}")
    return "\n".join(lines)
