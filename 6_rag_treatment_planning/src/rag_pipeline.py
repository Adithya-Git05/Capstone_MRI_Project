import json
from typing import Any
from .config import CHUNKS_PATH, EMBEDDING_MODEL, MAX_CHUNKS_PER_PAPER, RERANKER_MODEL, RELEVANCE_THRESHOLD, TOP_K
from .hybrid_retriever import HybridRetriever
from .llm import generate_response
from .patient_context import load_patient_context, patient_warnings
from .reranker import Reranker
from .relevance import expanded_query, filter_relevant
from .schemas import PatientContext
from .verification import check_numerical_consistency, validate_citations, validate_patient_interpretations, verify_claims, verify_evidence

class RAGPipeline:
    def __init__(self, chunks: list[dict[str, Any]] | None = None):
        self.chunks = chunks if chunks is not None else json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.retriever = HybridRetriever(self.chunks, EMBEDDING_MODEL)
        self.reranker = Reranker(RERANKER_MODEL)

    def query(
        self,
        question: str,
        case_id: str,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if patient_context is None:
            patient = load_patient_context(case_id)
        else:
            validated = PatientContext.model_validate(patient_context)
            if validated.case_id != case_id:
                raise ValueError("RAG context case_id does not match the query case_id.")
            patient = validated.model_dump(exclude_none=True)
        retrieval_query = expanded_query(question, patient)
        candidates = self.retriever.search(retrieval_query, top_k=max(TOP_K * 4, 20))
        ranked = self.reranker.rerank(retrieval_query, candidates, top_k=len(candidates))
        selected, retrieval_diagnostics = filter_relevant(
            retrieval_query, ranked, patient, RELEVANCE_THRESHOLD, MAX_CHUNKS_PER_PAPER, TOP_K
        )
        evidence, warnings = verify_evidence(selected, self.chunks)
        response_evidence = [
            dict(
                item,
                passage=item["text"],
                metadata={
                    key: item[key]
                    for key in (
                        "authors", "year", "journal", "doi", "pmid", "pmcid",
                        "evidence_type", "section", "source_url",
                    )
                    if item.get(key)
                },
            )
            for item in evidence
        ]
        if not evidence:
            answer = {
                "summary": "Insufficient sufficiently relevant evidence was retrieved for this question.",
                "patient_findings": [],
                "risk_interpretation": "No patient-specific risk interpretation was generated.",
                "corridor_comparison": [],
                "evidence_supported_considerations": [],
                "uncertainties": ["The retrieved literature did not meet the configured relevance threshold."],
                "missing_information": patient.get("missing_modalities", []),
                "limitations": ["This automated retrieval result is not clinical validation."],
                "citations": [],
            }
            llm_warning = "Ollama was not called because no sufficiently relevant evidence passed filtering."
        else:
            answer, llm_warning = generate_response(question, patient, evidence)
        warnings.extend(patient_warnings(patient))
        if llm_warning:
            warnings.append(llm_warning)
        citations = answer.get("citations", []) if isinstance(answer, dict) else []
        warnings.extend(validate_citations(citations, evidence))
        claim_verification = verify_claims(answer, evidence) if isinstance(answer, dict) else {"claims": [], "clinical_validation": False}
        numerical_checks = check_numerical_consistency(patient, answer) if isinstance(answer, dict) else {"checks": [], "discrepancy_count": 0}
        if claim_verification.get("unsupported_claim_count"):
            warnings.append("One or more generated claims could not be automatically supported by cited passages.")
        if numerical_checks.get("discrepancy_count"):
            warnings.append("One or more patient-specific numerical values differ from the supplied patient context.")
        warnings.extend(validate_patient_interpretations(answer, patient))
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_evidence": response_evidence,
            "patient_context": patient,
            "warnings": warnings,
            "verification": claim_verification,
            "numerical_checks": numerical_checks,
            "retrieval_diagnostics": retrieval_diagnostics,
        }
