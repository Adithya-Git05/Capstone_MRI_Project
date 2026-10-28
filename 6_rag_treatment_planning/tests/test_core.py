import json
from pathlib import Path
from unittest.mock import patch
from src.bm25_retriever import BM25Retriever
from src.chunking import chunk_text, make_chunks
from src.patient_context import load_patient_context, patient_warnings
from src.reranker import Reranker
from src.verification import check_numerical_consistency, validate_citations, verify_claims, verify_evidence
from src.llm import generate_response, normalize_citations
from src.relevance import expanded_query, filter_relevant
from src.verification import validate_patient_interpretations

CHUNKS = json.loads(Path('knowledge_base/chunks.json').read_text())

def test_chunking_preserves_sentences():
    chunks = chunk_text('One sentence. Two sentence. Three sentence.', target_words=3, overlap_words=1)
    assert chunks and all(chunk.endswith('.') for chunk in chunks)

def test_metadata_preserved():
    result = make_chunks('Evidence sentence.', {'paper_id':'P1','title':'Title','authors':['A'],'year':2024,'journal':'J','evidence_type':'review'})[0]
    assert result['paper_id'] == 'P1' and result['title'] == 'Title'

def test_bm25_retrieval():
    results = BM25Retriever(CHUNKS).search('DTI tractography', 1)
    assert results and results[0]['text']

def test_reranking_and_verification():
    results = Reranker().rerank('tractography', CHUNKS, 1)
    approved, warnings = verify_evidence(results, CHUNKS)
    assert approved and not warnings
    assert validate_citations([{'chunk_id':'bad'}], approved)

def test_patient_context_and_missing_data():
    patient = load_patient_context()
    assert patient['data_type'] == 'SYNTHETIC_DEMO'
    assert 'DTI' in patient_warnings(patient)[1]

def test_claim_verification_requires_supporting_citation():
    evidence = [{'chunk_id': 'P1_C001', 'title': 'Source', 'text': 'DTI tractography is sensitive to acquisition quality.'}]
    result = verify_claims({'summary': 'DTI tractography is sensitive to acquisition quality.', 'citations': [{'chunk_id': 'P1_C001'}]}, evidence)
    assert result['claims'][0]['status'] == 'supported'

def test_numerical_consistency_flags_patient_discrepancy():
    patient = {'tumor_volume_ml': 42.3, 'risk_score': 0.31}
    result = check_numerical_consistency(patient, {'summary': 'tumor_volume_ml is 99.0 and risk_score is 0.31'})
    assert result['discrepancy_count'] == 1

def test_relevance_filter_removes_unrelated_evidence_and_keeps_relevant():
    patient = {"tumor_type": "glioma", "candidate_corridors": [{"id": "A"}]}
    ranked = [
        {"chunk_id": "good", "paper_id": "P1", "title": "Glioma surgery", "text": "Glioma resection and tractography guide surgical planning.", "rerank_score": 2.0},
        {"chunk_id": "bad", "paper_id": "P2", "title": "Empathy after acquired brain injury", "text": "Empathy outcomes after acquired brain injury.", "rerank_score": 2.0},
    ]
    selected, diagnostics = filter_relevant("What is the role of tractography in glioma surgery?", ranked, patient, 0.18, 2, 5)
    assert [item["chunk_id"] for item in selected] == ["good"]
    assert diagnostics["kept_count"] == 1

def test_relevance_filter_rejects_generic_brain_overlap_for_tractography_query():
    ranked = [
        {
            "chunk_id": "irrelevant",
            "paper_id": "P2",
            "title": "Nose-brain axis",
            "text": "The nasal cavity connects to the central nervous system.",
            "rerank_score": 5.0,
        },
        {
            "chunk_id": "relevant",
            "paper_id": "P1",
            "title": "Diffusion MRI tractography in brain tumors",
            "text": "Diffusion MRI tractography can have limitations near brain tumors.",
            "rerank_score": 0.1,
        },
    ]
    selected, diagnostics = filter_relevant(
        "What are diffusion MRI tractography limitations near brain tumors?",
        ranked,
        {},
        0.35,
        2,
        5,
    )

    assert [item["chunk_id"] for item in selected] == ["relevant"]
    rejected = next(
        item for item in diagnostics["diagnostics"] if item["chunk_id"] == "irrelevant"
    )
    assert "insufficient overlap with the question's specific MRI topic" in rejected["reasons"]
    assert "no tumor/neoplasm context for a tumor-focused question" in rejected["reasons"]

def test_broad_query_expands_using_case_context():
    expanded = expanded_query("What are the important considerations for this case?", {"tumor_type": "glioma", "critical_structures": ["x"]})
    assert "glioma" in expanded and "tractography" in expanded

def test_unsupported_risk_labels_are_flagged():
    warnings = validate_patient_interpretations({"summary": "The risk score is moderate and corridor A is safest."}, {"risk_score": 0.31, "candidate_corridors": [{"id": "A"}]})
    assert len(warnings) == 2

def test_citations_normalize_valid_chunk_and_metadata():
    citations, warnings = normalize_citations([{"chunk_id": "P1_C001"}], [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source"}])
    assert citations[0]["status"] == "verified"
    assert citations[0]["title"] == "Source"
    assert warnings == []

def test_citations_accept_explicit_id_alias_only_when_it_matches_a_chunk():
    citations, warnings = normalize_citations([{"id": "P1_C001"}], [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source"}])
    assert citations[0]["chunk_id"] == "P1_C001"
    assert citations[0]["status"] == "verified"
    assert warnings == []

def test_citations_unpack_dictionary_collection_and_validate_each_record():
    evidence = [
        {"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source 1"},
        {"chunk_id": "P2_C001", "paper_id": "P2", "title": "Source 2"},
    ]
    raw = {"citations": {"first": {"id": "P1_C001"}, "second": {"id": "P2_C001"}}}
    citations, warnings = normalize_citations(raw, evidence)
    assert [item["chunk_id"] for item in citations] == ["P1_C001", "P2_C001"]
    assert all(item["status"] == "verified" for item in citations)
    assert warnings == []

def test_citations_keep_missing_and_unknown_dictionary_entries_unverified():
    raw = {"sources": {"missing": {"id": None}, "unknown": {"id": "P9_C001"}, "bad": "not-a-record"}}
    citations, warnings = normalize_citations(raw, [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source"}])
    assert len(citations) == 3
    assert all(item["status"] == "unverified" for item in citations)
    assert len(warnings) == 3

def test_citations_mark_missing_none_malformed_and_numeric_unverified():
    evidence = [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source"}]
    citations, warnings = normalize_citations([{"chunk_id": None}, None, 9, {"chunk_id": "missing"}], evidence)
    assert len(citations) == 4
    assert all(item["status"] == "unverified" for item in citations)
    assert len(warnings) == 4

def test_llm_withholds_generated_claims_when_citation_is_not_a_retrieved_chunk():
    evidence = [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source", "text": "Supported passage."}]
    generated = {
        "summary": "Unsupported generated claim.",
        "patient_findings": [],
        "risk_interpretation": "",
        "corridor_comparison": [],
        "evidence_supported_considerations": [],
        "uncertainties": [],
        "missing_information": [],
        "limitations": [],
        "citations": [{"chunk_id": "EXACT_ID"}],
    }
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": json.dumps(generated)},
        },
    )()

    with patch("src.llm.httpx.post", return_value=response):
        answer, warning = generate_response("Question", {}, evidence)

    assert answer["summary"].startswith("The model-generated synthesis was withheld")
    assert answer["citations"] == []
    assert warning and "EXACT_ID" in warning

def test_llm_accepts_citations_only_for_exact_retrieved_chunk_ids():
    evidence = [{"chunk_id": "P1_C001", "paper_id": "P1", "title": "Source", "text": "DTI tractography depends on acquisition quality."}]
    generated = {
        "summary": "DTI tractography depends on acquisition quality.",
        "patient_findings": [],
        "risk_interpretation": "",
        "corridor_comparison": [],
        "evidence_supported_considerations": [],
        "uncertainties": [],
        "missing_information": [],
        "limitations": [],
        "citations": [{"chunk_id": "P1_C001"}],
    }
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": json.dumps(generated)},
        },
    )()

    with patch("src.llm.httpx.post", return_value=response):
        answer, warning = generate_response("Question", {}, evidence)

    assert answer["summary"] == generated["summary"]
    assert answer["citations"][0]["chunk_id"] == "P1_C001"
    assert answer["citations"][0]["status"] == "verified"
    assert warning is None

def test_numeric_checks_omit_missing_and_accept_matching_or_rounded_values():
    patient = {"tumor_volume_ml": 42.3, "risk_score": 0.31}
    matching = check_numerical_consistency(patient, {"summary": "tumor_volume_ml: 42.3"})
    assert matching["discrepancy_count"] == 0
    assert [item["status"] for item in matching["checks"]] == ["consistent", "omitted"]
    rounded = check_numerical_consistency(patient, {"summary": "tumor_volume_ml: 42"})
    assert rounded["checks"][0]["status"] == "consistent"

def test_numeric_checks_report_field_expected_generated_and_reason():
    result = check_numerical_consistency({"risk_score": 0.31}, {"summary": "risk_score: 0.91"})
    check = result["checks"][0]
    assert check["status"] == "discrepancy"
    assert check["field"] == "risk_score"
    assert check["expected"] == [0.31]
    assert check["generated"] == [0.91]
    assert check["reason"]

def test_numeric_checks_handle_all_corridor_values_without_false_mismatch():
    patient = {
        "risk_score": 0.31,
        "candidate_corridors": [
            {"id": "A", "risk": 0.18, "coverage": 0.91, "length_mm": 44},
            {"id": "B", "risk": 0.24, "coverage": 0.96, "length_mm": 37},
            {"id": "C", "risk": 0.31, "coverage": 0.98, "length_mm": 29},
        ],
    }
    answer = {
        "summary": "No corridor is established as safest.",
        "corridor_comparison": [
            {"id": "A", "risk": 0.18, "coverage": 0.91, "length_mm": 44},
            {"id": "B", "risk": 0.24, "coverage": 0.96, "length_mm": 37},
            {"id": "C", "risk": 0.31, "coverage": 0.98, "length_mm": 29},
        ],
    }
    result = check_numerical_consistency(patient, answer)
    assert result["discrepancy_count"] == 0
    assert all(item["status"] in {"consistent", "omitted"} for item in result["checks"])

def test_formula_claim_is_unverified_even_with_valid_citation():
    evidence = [{"chunk_id": "P1_C001", "title": "Source", "text": "Risk is reported for the case."}]
    result = verify_claims({"summary": "Risk is calculated from the probability of damage.", "citations": [{"chunk_id": "P1_C001"}]}, evidence)
    assert result["claims"][0]["status"] == "unverified"
