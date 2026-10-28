import math
import json
import re
from typing import Any


def verify_evidence(evidence: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    valid_ids = {chunk["chunk_id"] for chunk in chunks}
    approved = []
    warnings = []
    for item in evidence:
        if item.get("chunk_id") not in valid_ids or not item.get("text", "").strip() or not item.get("title"):
            warnings.append(f"Evidence item {item.get('chunk_id', 'unknown')} failed deterministic source validation.")
            continue
        approved.append(item)
    if not approved:
        warnings.append("No retrieved passage passed deterministic source validation.")
    return approved, warnings


def validate_citations(citations: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[str]:
    valid = {item["chunk_id"] for item in evidence}
    warnings = []
    for item in citations:
        if not isinstance(item, dict):
            warnings.append("Citation entry has an unexpected format and is unverified.")
        elif item.get("chunk_id") not in valid or item.get("status") == "unverified":
            warnings.append(f"Citation {item.get('raw', item.get('chunk_id'))!r} is unverified because it does not resolve to retrieved evidence.")
    return warnings


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", value.lower()))


def _claim_items(answer: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    claims = []
    for field, value in answer.items():
        if field in {"citations", "verification", "numerical_checks"}:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                claims.append((field, item, []))
            elif isinstance(item, dict):
                text = item.get("claim") or item.get("statement") or item.get("text")
                cited = item.get("chunk_ids") or item.get("citations") or item.get("citation_ids") or []
                if isinstance(cited, str):
                    cited = [cited]
                if isinstance(text, str):
                    claims.append((field, text, [str(value) for value in cited]))
    return claims


def _support_score(claim: str, passage: str) -> float:
    claim_tokens = _tokens(claim)
    passage_tokens = _tokens(passage)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & passage_tokens) / len(claim_tokens)


def verify_claims(answer: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Automated lexical support check; it is not clinical or entailment validation."""
    by_id = {item["chunk_id"]: item for item in evidence}
    global_ids = [
        item.get("chunk_id") for item in answer.get("citations", [])
        if isinstance(item, dict) and item.get("chunk_id") in by_id
    ]
    results = []
    for field, claim, cited_ids in _claim_items(answer):
        ids = [chunk_id for chunk_id in cited_ids if chunk_id in by_id] or global_ids
        if not ids:
            results.append({"field": field, "claim": claim, "status": "unverified", "reason": "No valid citation was associated with the claim."})
            continue
        scores = {chunk_id: _support_score(claim, by_id[chunk_id]["text"]) for chunk_id in ids}
        best_id, best_score = max(scores.items(), key=lambda item: item[1])
        methodology_claim = bool(re.search(
            r"\b(calculat|derived|formula|based on|probability of damage|computed from|determined by)\b",
            claim.lower(),
        ))
        status = "supported" if best_score >= 0.35 and not methodology_claim else "unverified"
        reason = (
            "The answer asserts an unsupported calculation or formula; project methodology was not supplied."
            if methodology_claim
            else "Automated lexical overlap only; clinical validation was not performed."
        )
        results.append({
            "field": field,
            "claim": claim,
            "chunk_id": best_id,
            "support_score": round(best_score, 3),
            "status": status,
            "reason": reason,
        })
    return {
        "method": "token-overlap heuristic, threshold=0.35",
        "clinical_validation": False,
        "claims": results,
        "unsupported_claim_count": sum(item["status"] != "supported" for item in results),
        "claim_count": len(results),
    }


def _number_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(value):
        return [float(value)]
    return []


def _numbers_from_value(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number_values(value)
    if isinstance(value, list):
        numbers = []
        for item in value:
            numbers.extend(_numbers_from_value(item))
        return numbers
    if isinstance(value, str):
        return [
            float(item)
            for item in re.findall(
                r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value
            )
        ]
    return []


def _named_numbers(value: Any, aliases: set[str]) -> list[float]:
    """Find numbers under matching structured keys without reading nearby fields."""
    numbers = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in aliases:
                numbers.extend(_numbers_from_value(item))
            numbers.extend(_named_numbers(item, aliases))
    elif isinstance(value, list):
        for item in value:
            numbers.extend(_named_numbers(item, aliases))
    return numbers


def check_numerical_consistency(patient: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Compare only explicitly named patient fields; omitted fields are not discrepancies."""
    text = json.dumps(answer, ensure_ascii=True)
    checks = []
    fields: list[tuple[str, set[str], list[float]]] = []
    for key, value in patient.items():
        if isinstance(value, list) and key == "candidate_corridors":
            for corridor in value:
                if not isinstance(corridor, dict):
                    continue
                corridor_id = str(corridor.get("id", "")).lower()
                for field, corridor_value in corridor.items():
                    expected = _number_values(corridor_value)
                    if expected:
                        fields.append((
                            f"corridor_{corridor_id}_{field}",
                            {f'"{field}"', f"{corridor_id} {field}", f"corridor {corridor_id} {field}"},
                            expected,
                        ))
            continue
        expected_values = _number_values(value)
        if expected_values:
            fields.append((key, {key.replace("_", " ").lower(), key.lower()}, expected_values))
    lowered = text.lower()
    for key, aliases, expected_values in fields:
        if key.startswith("corridor_"):
            _, corridor_id, field_name = key.split("_", 2)
            found = []
            for item in _iter_dicts(answer):
                if str(item.get("id", "")).lower() == corridor_id:
                    found.extend(_numbers_from_value(item.get(field_name)))
        else:
            found = _named_numbers(answer, {alias.replace(" ", "_") for alias in aliases})
            if not found:
                prose_aliases = "|".join(
                    re.escape(alias)
                    .replace(r"\ ", r"[\s_]+")
                    .replace("_", r"[\s_]+")
                    for alias in aliases
                )
                found = [
                    float(item)
                    for item in re.findall(
                        rf"(?:{prose_aliases})\s*(?:is|was|:|=)\s*"
                        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                        lowered,
                    )
                ]
        if not found:
            checks.append({"field": key, "expected": expected_values, "status": "omitted"})
            continue
        matches = any(
            abs(candidate - expected) <= max(
                1e-6,
                abs(expected) * 1e-4,
                0.5 if candidate.is_integer() else max(0.005, abs(expected) * 0.05),
            )
            for expected in expected_values
            for candidate in found
        )
        checks.append({
            "field": key,
            "expected": expected_values,
            "generated": found,
            "status": "consistent" if matches else "discrepancy",
            "reason": "Named patient value matches source." if matches else "Named patient value differs from source.",
        })
    discrepancies = [item for item in checks if item["status"] == "discrepancy"]
    return {
        "source_of_truth": "patient_context",
        "checks": checks,
        "discrepancy_count": len(discrepancies),
        "warning": "This is deterministic consistency checking, not clinical validation.",
    }


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def validate_patient_interpretations(answer: dict[str, Any], patient: dict[str, Any]) -> list[str]:
    """Flag clinical labels that are not justified by the supplied demo schema."""
    text = str(answer).lower()
    warnings = []
    if patient.get("risk_score") is not None and re.search(
        r"\brisk(?: score)?\b[^.]{0,120}\b(low|moderate|high|safe|unsafe)\b", text
    ):
        warnings.append("Risk labels are not validated because no documented risk scale or threshold was supplied.")
    if patient.get("classification_probability") is not None and re.search(
        r"\b(probability|confidence)\b[^.]{0,120}\b(calibrated|diagnostic|certain)\b", text
    ):
        warnings.append("The supplied classification probability is an uncalibrated synthetic demonstration value.")
    if patient.get("candidate_corridors") and re.search(
        r"\b(corridor|route|approach)\b[^.]{0,120}\b(safest|best|recommend|preferred)\b", text
    ):
        warnings.append("Synthetic corridor metrics cannot establish a safest or recommended surgical route.")
    if re.search(r"\b(calculat|derived|formula|based on|probability of damage|computed from)\b", text):
        warnings.append("A generated calculation or formula claim is unverified because no project methodology defines it.")
    return warnings
