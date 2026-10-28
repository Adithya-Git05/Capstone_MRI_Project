import re
from typing import Any

from .bm25_retriever import tokenize

STOPWORDS = {
    "about", "after", "also", "are", "can", "case", "does", "for", "from",
    "how", "important", "into", "more", "need", "patient", "that", "the",
    "this", "what", "when", "which", "with", "your",
}
DOMAIN_TERMS = {
    "glioma", "glioblastoma", "tumor", "neoplasm", "resection", "surgery",
    "surgical", "neurosurgery", "tractography", "diffusion", "dti", "fmri",
    "mapping", "eloquent", "vascular", "artery", "vessel", "corridor",
    "white", "matter", "critical", "structure", "brain",
}
PLANNING_TERMS = {
    "resection", "surgery", "surgical", "neurosurgery", "tractography",
    "mapping", "eloquent", "vascular", "corridor", "critical", "structure",
}
NON_PLANNING_TERMS = {
    "treatment", "therapy", "therapeutic", "molecular", "signaling",
    "survival", "bone", "joint", "nasal", "empathy", "injury",
}
TUMOR_TERMS = {
    "tumor", "tumors", "glioma", "gliomas", "glioblastoma",
    "neoplasm", "neoplasms", "lesion", "lesions",
}
CASE_TERMS = (
    "glioma surgical planning extent of resection critical structures "
    "functional mapping DTI tractography vascular anatomy surgical corridor"
)


def expanded_query(question: str, patient: dict[str, Any]) -> str:
    question_tokens = set(tokenize(question))
    broad = len(question_tokens - STOPWORDS) <= 5 or "this case" in question.lower()
    if not broad:
        return question
    terms = [str(patient.get("tumor_type", "")), CASE_TERMS]
    if patient.get("critical_structures"):
        terms.append("critical structures")
    if patient.get("candidate_corridors"):
        terms.append("corridor planning")
    return f"{question} {' '.join(term for term in terms if term)}"


def _meaningful_tokens(value: str) -> set[str]:
    return {token for token in tokenize(value) if token not in STOPWORDS}


def score_relevance(
    question: str,
    item: dict[str, Any],
    patient: dict[str, Any],
    rerank_score: float,
) -> tuple[float, list[str]]:
    text = f"{item.get('title', '')} {item.get('text', '')}"
    text_tokens = _meaningful_tokens(text)
    query_tokens = _meaningful_tokens(question)
    overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
    domain_overlap = len(text_tokens & DOMAIN_TERMS) / max(len(DOMAIN_TERMS), 1)
    query_domain_terms = query_tokens & DOMAIN_TERMS
    matching_domain_terms = query_domain_terms & text_tokens
    planning_overlap = text_tokens & PLANNING_TERMS
    normalized_rerank = 1.0 / (1.0 + pow(2.718281828, -rerank_score))
    score = 0.55 * overlap + 0.25 * normalized_rerank + 0.20 * min(domain_overlap * 4, 1.0)
    reasons: list[str] = []
    lowered = text.lower()
    title_tokens = _meaningful_tokens(str(item.get("title", "")))
    if overlap == 0:
        score *= 0.2
        reasons.append("no meaningful query-term overlap")
    if not text_tokens & DOMAIN_TERMS:
        score *= 0.25
        reasons.append("no glioma/neurosurgical domain signal")
    minimum_domain_matches = 1 if len(query_domain_terms) <= 2 else 2
    if len(matching_domain_terms) < minimum_domain_matches:
        score *= 0.25
        reasons.append("insufficient overlap with the question's specific MRI topic")
    if query_tokens & TUMOR_TERMS and not text_tokens & TUMOR_TERMS:
        score *= 0.15
        reasons.append("no tumor/neoplasm context for a tumor-focused question")
    planning_question = any(
        term in question.lower()
        for term in ("surg", "corridor", "critical structure", "case")
    )
    if planning_question:
        if not planning_overlap:
            score *= 0.35
            reasons.append("no surgical-planning signal")
        if title_tokens & NON_PLANNING_TERMS and not title_tokens & PLANNING_TERMS:
            score *= 0.2
            reasons.append("title indicates a non-planning topic")
    if "this case" in question.lower() and not patient.get("age"):
        if re.search(r"\b(pediatric|paediatric|children|childhood|adolescent)\b", lowered):
            score *= 0.2
            reasons.append("population-specific evidence while patient age is unknown")
    if score < 0.18:
        reasons.append("below relevance threshold")
    return round(score, 4), reasons


def filter_relevant(
    question: str,
    ranked: list[dict[str, Any]],
    patient: dict[str, Any],
    threshold: float,
    max_chunks_per_paper: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    paper_counts: dict[str, int] = {}
    for item in ranked:
        relevance_score, reasons = score_relevance(
            question, item, patient, float(item.get("rerank_score", 0.0))
        )
        copied = dict(item)
        copied["relevance_score"] = relevance_score
        paper_id = str(item.get("paper_id", ""))
        if relevance_score < threshold:
            reason = reasons or ["below relevance threshold"]
            diagnostics.append({"chunk_id": item.get("chunk_id"), "score": relevance_score, "kept": False, "reasons": reason})
            continue
        if paper_counts.get(paper_id, 0) >= max_chunks_per_paper:
            diagnostics.append({"chunk_id": item.get("chunk_id"), "score": relevance_score, "kept": False, "reasons": ["per-paper diversity limit"]})
            continue
        selected.append(copied)
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        diagnostics.append({"chunk_id": item.get("chunk_id"), "score": relevance_score, "kept": True, "reasons": reasons})
        if len(selected) == top_k:
            break
    return selected, {
        "candidate_count": len(ranked),
        "kept_count": len(selected),
        "threshold": threshold,
        "diagnostics": diagnostics,
    }
