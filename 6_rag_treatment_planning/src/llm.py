import json
import httpx
from typing import Any
from .config import GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """You are an evidence-synthesis component for a research prototype, not a doctor. The patient history is explicitly synthetic demonstration data and must be described as such; it is not an MRI finding. Structured segmentation, classification, risk, and corridor fields are outputs from the active MRI analysis when supplied. Report only those supplied outputs, preserve their values, and clearly label them as experimental model outputs rather than verified clinical findings. Never replace missing model outputs with values from an example, prior case, or generic assumption; say they are unavailable. Use only sufficiently relevant supplied evidence and patient values. The input lists the only allowed source identifiers in available_chunk_ids. Cite a literature claim only by copying an exact identifier from that list. Return citations as a list of objects with a chunk_id field. Never output a placeholder, example identifier, bibliography number, or identifier not in that list. If no supplied passage supports a claim, leave citations empty and state that evidence is insufficient. Do not invent citations, numerical values, diagnoses, risk, safety, or surgical decisions. Never label a supplied risk score low, moderate, or high unless a documented validated threshold is supplied. Do not choose a surgical corridor from risk, coverage, or length values. Do not infer age, sex, tumor location, histology, neurological status, or imaging findings. Keep history, model outputs, and general literature findings separate; mention missing data and conflicting evidence, and say when evidence is insufficient. Return valid JSON with keys: summary, patient_findings, risk_interpretation, corridor_comparison, evidence_supported_considerations, uncertainties, missing_information, limitations, citations."""
ANSWER_KEYS = {
    "summary", "patient_findings", "risk_interpretation", "corridor_comparison",
    "evidence_supported_considerations", "uncertainties", "missing_information",
    "limitations", "citations",
}
MAX_PROMPT_PASSAGE_CHARS = 1600


def _prompt_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep retrieved passages bounded before sending the generation prompt."""
    return [
        {
            "chunk_id": item.get("chunk_id"),
            "paper_id": item.get("paper_id"),
            "title": item.get("title"),
            "text": str(item.get("text", ""))[:MAX_PROMPT_PASSAGE_CHARS],
        }
        for item in evidence
    ]


def _parse_answer(payload: Any) -> dict[str, Any]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    raw_value = None
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        raw_value = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if isinstance(raw_value, str):
        value = json.loads(raw_value)
    else:
        value = raw_value
    if isinstance(value, dict) and isinstance(value.get("answer"), dict):
        value = value["answer"]
    if not isinstance(value, dict):
        raise ValueError("Gemini returned no JSON object in its response.")
    if not ANSWER_KEYS.intersection(value):
        raise ValueError("Gemini JSON did not contain the expected answer fields.")
    return value


def normalize_citations(citations: Any, evidence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize model citations without treating bibliography numbers as chunk IDs."""
    by_chunk = {str(item["chunk_id"]): item for item in evidence}
    if citations is None:
        return [], []
    while isinstance(citations, dict):
        container_keys = ("citations", "sources", "references", "evidence")
        values = next((citations[key] for key in container_keys if key in citations), None)
        if values is not None:
            citations = values
            continue
        if any(key in citations for key in ("chunk_id", "citation_id", "source_chunk_id", "source_id", "id", "citation")):
            citations = [citations]
        else:
            citations = [
                ({**value} if isinstance(value, dict) else {"raw": value, "chunk_id": key})
                for key, value in citations.items()
            ]
    if not isinstance(citations, list):
        citations = [citations]
    normalized = []
    warnings = []
    for citation in citations:
        raw = citation
        if isinstance(citation, str):
            candidate = citation.strip()
            candidate = candidate[1:-1].strip() if candidate.startswith("[") and candidate.endswith("]") else candidate
            citation = {"chunk_id": candidate} if candidate in by_chunk else {"raw": raw}
        elif isinstance(citation, (int, float)) and not isinstance(citation, bool):
            citation = {"raw": citation}
        elif not isinstance(citation, dict):
            citation = {"raw": raw}
        chunk_id = (
            citation.get("chunk_id")
            or citation.get("citation_id")
            or citation.get("source_chunk_id")
            or citation.get("source_id")
            or citation.get("id")
            or citation.get("citation")
        )
        if isinstance(chunk_id, str) and chunk_id in by_chunk:
            source = by_chunk[chunk_id]
            normalized.append({
                "chunk_id": chunk_id,
                "paper_id": source.get("paper_id", ""),
                "title": source.get("title", ""),
                "status": "verified",
            })
            continue
        normalized.append({
            "raw": raw,
            "chunk_id": None,
            "paper_id": citation.get("paper_id") if isinstance(citation, dict) else None,
            "title": citation.get("title") if isinstance(citation, dict) else None,
            "status": "unverified",
            "reason": "Citation does not resolve to a retrieved chunk; bibliography numbers are not chunk IDs.",
        })
        warnings.append(f"Unresolved citation {raw!r} was marked unverified; it was not mapped to a retrieved chunk.")
    return normalized, warnings


def generate_response(question: str, patient: dict[str, Any], evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    analysis_text = patient.get("analysis_results_text")
    if not isinstance(analysis_text, str) or not analysis_text.strip():
        analysis_text = json.dumps(patient, indent=2, ensure_ascii=True)
    evidence_text = "\n\n".join(
        "\n".join(
            (
                f"[LITERATURE SOURCE chunk_id={item.get('chunk_id', '')}]",
                f"Paper: {item.get('paper_id', '')}",
                f"Title: {item.get('title', '')}",
                "Passage:",
                str(item.get("text", ""))[:MAX_PROMPT_PASSAGE_CHARS],
            )
        )
        for item in evidence
    )
    available_chunk_ids = ", ".join(
        str(item.get("chunk_id", "")) for item in evidence if item.get("chunk_id")
    )
    prompt = (
        "QUESTION:\n"
        + question
        + "\n\nPATIENT HISTORY AND PRIOR MRI ANALYSIS RESULTS (plain text; source of "
        "patient-specific findings):\n"
        + analysis_text
        + "\n\nRETRIEVED PAPER LITERATURE (use for general evidence; cite only exact "
        "chunk_id values shown here):\n"
        + evidence_text
        + "\n\nALLOWED CITATION CHUNK IDS:\n"
        + available_chunk_ids
        + "\n\nReturn the requested JSON answer. Keep synthetic history, MRI model "
        "outputs, and literature findings clearly separated. Do not infer missing "
        "model results."
    )
    if not GEMINI_API_KEY:
        return {}, "Gemini is not configured; set GEMINI_API_KEY in 6_rag_treatment_planning\\.env."
    try:
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=90,
        )
        response.raise_for_status()
        value = _parse_answer(response.json())
        citations, citation_warnings = normalize_citations(value.get("citations"), evidence)
        if citation_warnings:
            warning = (
                "Generated text was withheld because one or more citations did not "
                "match the retrieved passages."
            )
            details = " ".join(citation_warnings)
            return (
                {
                    "summary": (
                        "The model-generated synthesis was withheld because its citations "
                        "could not be verified against retrieved sources."
                    ),
                    "patient_findings": [],
                    "risk_interpretation": "",
                    "corridor_comparison": [],
                    "evidence_supported_considerations": [],
                    "uncertainties": [
                        "Review the retrieved passages directly; no generated claim is presented as supported."
                    ],
                    "missing_information": [],
                    "limitations": ["Automated literature retrieval is not clinical validation."],
                    "citations": [],
                },
                " ".join(part for part in (warning, details) if part),
            )
        value["citations"] = citations
        if not citations:
            return (
                value,
                "Gemini returned no verifiable citations. Treat its literature synthesis as "
                "unverified and review the retrieved passages.",
            )
        return value, None
    except httpx.HTTPStatusError as exc:
        message = ""
        try:
            error_payload = exc.response.json()
        except ValueError:
            error_payload = {}
        if isinstance(error_payload, dict):
            details = error_payload.get("error", {})
            if isinstance(details, dict):
                message = str(details.get("message", ""))
        message = message.replace(GEMINI_API_KEY, "[redacted]") if GEMINI_API_KEY else message
        detail = f": {message}" if message else ""
        return (
            {},
            f"Gemini API returned HTTP {exc.response.status_code} for model "
            f"'{GEMINI_MODEL}'{detail}",
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {}, f"Gemini response could not be used: {exc}"
