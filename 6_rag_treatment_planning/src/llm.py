import json
import httpx
from typing import Any
from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """You are an evidence-synthesis component for a research prototype, not a doctor. Use only sufficiently relevant supplied evidence and patient values. The input lists the only allowed source identifiers in available_chunk_ids. Cite a literature claim only by copying an exact identifier from that list. Return citations as a list of objects with a chunk_id field. Never output a placeholder, example identifier, bibliography number, or identifier not in that list. If no supplied passage supports a claim, leave citations empty and state that evidence is insufficient. Do not invent citations, numerical values, diagnoses, risk, safety, or surgical decisions. Never label a supplied risk score low, moderate, or high unless a documented validated threshold is supplied. Treat synthetic scores and classification probabilities as illustrative, uncalibrated demo outputs, not clinical recommendations. Do not choose a surgical corridor from synthetic risk, coverage, or length values. Do not infer age, sex, tumor location, histology, neurological status, or imaging findings. Keep patient facts separate from general literature findings, mention missing data and conflicting evidence, and say when evidence is insufficient. Return valid JSON with keys: summary, patient_findings, risk_interpretation, corridor_comparison, evidence_supported_considerations, uncertainties, missing_information, limitations, citations."""
ANSWER_KEYS = {
    "summary", "patient_findings", "risk_interpretation", "corridor_comparison",
    "evidence_supported_considerations", "uncertainties", "missing_information",
    "limitations", "citations",
}
MAX_PROMPT_PASSAGE_CHARS = 1600


def _prompt_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the generation prompt within the local model's context window."""
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
    raw_value = payload.get("response") if isinstance(payload, dict) else None
    if isinstance(raw_value, str):
        value = json.loads(raw_value)
    else:
        value = raw_value
    if isinstance(value, dict) and isinstance(value.get("answer"), dict):
        value = value["answer"]
    if not isinstance(value, dict):
        raise ValueError("Ollama returned no JSON object in the response field.")
    if not ANSWER_KEYS.intersection(value):
        raise ValueError("Ollama JSON did not contain the expected answer fields.")
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
    context = {
        "question": question,
        "patient_context": patient,
        "available_chunk_ids": [item.get("chunk_id") for item in evidence],
        "retrieved_evidence": _prompt_evidence(evidence),
    }
    citation_example = json.dumps(
        {"citations": [{"chunk_id": str(evidence[0].get("chunk_id", ""))}]}
    )
    prompt = (
        SYSTEM_PROMPT
        + "\nCITATION FORMAT EXAMPLE USING A REAL RETRIEVED ID; use it only if the passage supports the claim: "
        + citation_example
        + "\nINPUT:\n"
        + json.dumps(context, indent=2)
    )
    try:
        response = httpx.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}, timeout=90)
        response.raise_for_status()
        value = _parse_answer(response.json())
        citations, citation_warnings = normalize_citations(value.get("citations"), evidence)
        if citation_warnings or not citations:
            warning = (
                "Generated text was withheld because it supplied no verifiable citation "
                "to the retrieved passages."
                if not citations
                else "Generated text was withheld because one or more citations did not "
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
        return value, None
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {}, f"Ollama response could not be used: {exc}"
