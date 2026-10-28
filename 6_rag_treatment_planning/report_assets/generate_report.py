from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Capstone_RAG_Chatbot_Implementation_Report.pdf"


class WorkflowDiagram(Flowable):
    def __init__(self, title: str, steps: list[str], width: float = 170 * mm, height: float = 82 * mm):
        super().__init__()
        self.title = title
        self.steps = steps
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#264653"))
        canvas.setFillColor(colors.HexColor("#264653"))
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(0, self.height - 12, self.title)
        box_w = 35 * mm
        box_h = 17 * mm
        x_gap = 8 * mm
        y_gap = 11 * mm
        usable = self.width
        rows = [self.steps[i : i + 3] for i in range(0, len(self.steps), 3)]
        for row_index, row in enumerate(rows):
            y = self.height - 30 - row_index * (box_h + y_gap)
            for col_index, label in enumerate(row):
                x = col_index * (box_w + x_gap)
                if row_index % 2:
                    x = usable - box_w - col_index * (box_w + x_gap)
                canvas.setFillColor(colors.HexColor("#e8f1f2"))
                canvas.setStrokeColor(colors.HexColor("#2a9d8f"))
                canvas.roundRect(x, y, box_w, box_h, 4, fill=1, stroke=1)
                canvas.setFillColor(colors.HexColor("#173042"))
                canvas.setFont("Helvetica", 7.8)
                words = label.split()
                lines: list[str] = []
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip()
                    if len(candidate) > 22 and current:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                start_y = y + box_h / 2 + (len(lines) - 1) * 4
                for line in lines:
                    canvas.drawCentredString(x + box_w / 2, start_y, line)
                    start_y -= 8
                if col_index < len(row) - 1:
                    arrow_x = x + box_w
                    canvas.setStrokeColor(colors.HexColor("#e76f51"))
                    canvas.line(arrow_x, y + box_h / 2, arrow_x + x_gap - 2, y + box_h / 2)
                    canvas.line(arrow_x + x_gap - 2, y + box_h / 2, arrow_x + x_gap - 5, y + box_h / 2 + 3)
                    canvas.line(arrow_x + x_gap - 2, y + box_h / 2, arrow_x + x_gap - 5, y + box_h / 2 - 3)
        canvas.restoreState()


class IngestionDiagram(Flowable):
    def __init__(self, width: float = 170 * mm, height: float = 58 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.HexColor("#264653"))
        c.drawString(0, self.height - 12, "Figure 2. Literature acquisition and indexing pipeline")
        labels = [
            "PMC / NCBI search",
            "Verified PDF + metadata",
            "PyMuPDF extraction",
            "Clean + chunk",
            "Embeddings + BM25",
            "Persisted artifacts",
        ]
        box_w = 25 * mm
        gap = 3.5 * mm
        y = 18 * mm
        for i, label in enumerate(labels):
            x = i * (box_w + gap)
            c.setFillColor(colors.HexColor("#f3ede3"))
            c.setStrokeColor(colors.HexColor("#f4a261"))
            c.roundRect(x, y, box_w, 18 * mm, 4, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#173042"))
            c.setFont("Helvetica", 7)
            parts = label.split()
            lines = []
            current = ""
            for word in parts:
                if len(current + " " + word) > 16 and current:
                    lines.append(current)
                    current = word
                else:
                    current = (current + " " + word).strip()
            if current:
                lines.append(current)
            start = y + 13 * mm
            for line in lines:
                c.drawCentredString(x + box_w / 2, start, line)
                start -= 8
            if i < len(labels) - 1:
                c.setStrokeColor(colors.HexColor("#e76f51"))
                c.line(x + box_w, y + 9 * mm, x + box_w + gap - 1, y + 9 * mm)
                c.line(x + box_w + gap - 1, y + 9 * mm, x + box_w + gap - 4, y + 9 * mm + 3)
                c.line(x + box_w + gap - 1, y + 9 * mm, x + box_w + gap - 4, y + 9 * mm - 3)
        c.restoreState()


def load_data() -> dict:
    audit = json.loads((ROOT / "data" / "corpus_audit.json").read_text(encoding="utf-8"))
    patient = json.loads((ROOT / "data" / "demo_patient.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "knowledge_base" / "vector_db" / "manifest.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "data" / "evaluation_questions.json").read_text(encoding="utf-8"))
    metadata = list(csv.DictReader((ROOT / "data" / "literature_metadata.csv").open(encoding="utf-8", newline="")))
    return {"audit": audit, "patient": patient, "manifest": manifest, "questions": questions, "metadata": metadata}


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, header=True, small=False):
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653") if header else colors.HexColor("#f3ede3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.HexColor("#173042")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c4c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
    ]
    if small:
        style.extend([("FONTSIZE", (0, 0), (-1, -1), 7.4), ("LEADING", (0, 0), (-1, -1), 9)])
    else:
        style.extend([("FONTSIZE", (0, 0), (-1, -1), 8.2), ("LEADING", (0, 0), (-1, -1), 10)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    t.setStyle(TableStyle(style))
    return t


def build_story(data: dict, styles: dict[str, ParagraphStyle]) -> list:
    audit = data["audit"]
    patient = data["patient"]
    manifest = data["manifest"]
    questions = data["questions"]
    metadata = data["metadata"]
    story: list = []

    story += [
        Spacer(1, 22 * mm),
        p("CAPSTONE IMPLEMENTATION REPORT", styles["eyebrow"]),
        Spacer(1, 5 * mm),
        p("Brain-Tumor Surgical-Planning<br/>Knowledge-Base Chatbot", styles["title"]),
        Spacer(1, 8 * mm),
        p("Technical implementation, verification status, and reproducibility dossier", styles["subtitle"]),
        Spacer(1, 20 * mm),
        Table(
            [
                ["Project type", "Undergraduate Capstone research prototype"],
                ["Implementation scope", "Local literature-grounded conversational retrieval system"],
                ["Primary runtime", "Python, FastAPI, Ollama, SentenceTransformers"],
                ["Demonstration case", "DEMO_001 — explicitly synthetic"],
                ["Report date", "10 October 2026"],
            ],
            colWidths=[42 * mm, 112 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#264653")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f3ede3")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c4c8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 18 * mm),
        p("Scope boundary", styles["callout_title"]),
        p(
            "This report documents the implementation present in the workspace. The system is a research and education prototype, not a clinically validated surgical decision-support system. It retrieves supplied literature and preserves supplied demonstration values; it does not perform segmentation, DTI/fMRI processing, vascular modeling, risk-field computation, or surgical corridor optimization.",
            styles["callout"],
        ),
        PageBreak(),
        p("Contents", styles["h1"]),
        p("1. Executive summary and objectives", styles["toc"]),
        p("2. System architecture and workflow", styles["toc"]),
        p("3. Technology stack and configuration", styles["toc"]),
        p("4. Literature acquisition and knowledge-base construction", styles["toc"]),
        p("5. Retrieval, fusion, reranking, and diversity", styles["toc"]),
        p("6. Ollama generation and response handling", styles["toc"]),
        p("7. Citations, verification, and numerical consistency", styles["toc"]),
        p("8. Backend API and frontend integration", styles["toc"]),
        p("9. Synthetic demonstration case", styles["toc"]),
        p("10. Testing and evaluation status", styles["toc"]),
        p("11. Reproducibility", styles["toc"]),
        p("12. Implementation status, limitations, and future work", styles["toc"]),
        p("13. File-by-file implementation map", styles["toc"]),
        PageBreak(),
        p("1. Executive summary and objectives", styles["h1"]),
        p(
            "The Brain-Tumor Surgical-Planning Knowledge-Base Chatbot is a local retrieval-augmented generation prototype. Its purpose is to expose an auditable workflow in which a user question is matched against a literature corpus, relevant passages are selected, and a local language model produces a structured answer constrained by retrieved evidence and a synthetic patient context.",
            styles["body"],
        ),
        p(
            "The implemented objective is not autonomous surgical planning. Instead, the project demonstrates document acquisition, metadata-aware indexing, hybrid retrieval, reranking, source-aware response construction, citation-ID validation, deterministic numerical consistency checks, and visible uncertainty handling.",
            styles["body"],
        ),
        table(
            [
                ["Measured workspace fact", "Current evidence"],
                ["Literature files", f"{audit['pdf_files']} PDFs and {audit['metadata_rows']} metadata rows"],
                ["Indexed corpus", f"{audit['indexed_papers']} papers and {audit['indexed_chunks']} chunks"],
                ["Persisted vector artifact", "Present; NumPy format with compatibility manifest"],
                ["Automated tests", "23 passed, 1 warning in the latest run"],
                ["Evaluation dataset", f"{len(questions)} questions; all pending human annotation"],
            ],
            [55 * mm, 99 * mm],
            small=True,
        ),
        Spacer(1, 6 * mm),
        p("Implementation classification", styles["h2"]),
        p(
            "Core application flow, schemas, retrieval components, indexing artifacts, and automated tests are implemented. Claim verification is heuristic rather than semantic or clinical. Corpus counts are auditable, but corpus relevance and evidence-type labels require manual review. No clinical performance or retrieval-quality metric is reported.",
            styles["body"],
        ),
        PageBreak(),
        p("2. System architecture and workflow", styles["h1"]),
        p(
            "The runtime path begins in the browser and terminates in a structured JSON response rendered by the frontend. The orchestration method is RAGPipeline.query() in src/rag_pipeline.py.",
            styles["body"],
        ),
        WorkflowDiagram(
            "Runtime query workflow",
            [
                "Browser frontend",
                "FastAPI API",
                "Patient context",
                "BM25 + vector",
                "RRF fusion",
                "CrossEncoder",
                "Relevance + diversity",
                "Evidence validation",
                "Ollama JSON",
                "Citation normalization",
                "Claim + numeric checks",
                "Structured response",
            ],
        ),
        p("Figure 1. Implemented runtime query workflow.", styles["caption"]),
        p(
            "For broad case questions, src/relevance.py may expand the retrieval query using known case topics. The expansion does not infer unprovided anatomy, age, imaging, or clinical findings. Candidate passages are retrieved, fused, reranked, filtered by relevance, and limited to at most two chunks per paper by default.",
            styles["body"],
        ),
        IngestionDiagram(),
        p("The ingestion path is separate from the runtime query path. scripts/build_knowledge_base.py regenerates chunks and saves the vector artifact; the API rebuild route performs local ingestion and reconstructs the in-memory pipeline.", styles["body"]),
        p("Figure 2. Implemented literature acquisition and indexing pipeline.", styles["caption"]),
        PageBreak(),
        p("3. Technology stack and configuration", styles["h1"]),
        p(
            "The project uses Python 3.13.2 in the workspace virtual environment. The dependency manifest is requirements.txt. FastAPI and Uvicorn provide the local HTTP service; Pydantic defines response and patient schemas; PyMuPDF extracts PDF text; NumPy stores dense vector artifacts; SentenceTransformers supplies embeddings and reranking models; HTTPX communicates with Ollama.",
            styles["body"],
        ),
        table(
            [
                ["Concern", "Implementation / configured value", "Source"],
                ["Backend", "FastAPI + Uvicorn", "api/main.py"],
                ["Local LLM", "Ollama; llama3.2:3b", "src/config.py"],
                ["Embeddings", "sentence-transformers/all-MiniLM-L6-v2", "src/config.py"],
                ["Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", "src/config.py"],
                ["Lexical retrieval", "Custom BM25-style scorer", "src/bm25_retriever.py"],
                ["Vector storage", "NumPy embeddings.npy + manifest.json", "src/vector_retriever.py"],
                ["Fusion", "Reciprocal Rank Fusion; RRF_K=60", "src/hybrid_retriever.py"],
                ["Final top-k", "5", "src/config.py"],
                ["Per-paper diversity", "Maximum 2 chunks per paper", "src/config.py"],
                ["Relevance threshold", "0.35", "src/config.py"],
            ],
            [34 * mm, 82 * mm, 38 * mm],
            small=True,
        ),
        p(
            "FAISS and rank-bm25 appear in requirements.txt, but the inspected runtime source uses NumPy vectors and a custom BM25 implementation rather than directly importing those packages. The application has lexical fallbacks if the transformer models cannot be loaded.",
            styles["body"],
        ),
        PageBreak(),
        p("4. Literature acquisition and knowledge-base construction", styles["h1"]),
        p(
            "scripts/download_literature.py searches PMC using NCBI E-utilities, restricts queries to open-access records, deduplicates PMCID values, retrieves summary metadata, and obtains permitted PDF URLs through the PMC Open Data mechanism. It verifies a PDF signature and minimum size before replacing the destination file. Failed records retain an error message and manual-review flag.",
            styles["body"],
        ),
        table(
            [
                ["Corpus audit field", "Value"],
                ["Metadata records", str(audit["metadata_rows"])],
                ["PDF files", str(audit["pdf_files"])],
                ["Indexed papers", str(audit["indexed_papers"])],
                ["Indexed chunks", str(audit["indexed_chunks"])],
                ["Duplicate PMCIDs", str(len(audit["duplicate_pmcids"]))],
                ["Duplicate filenames", str(len(audit["duplicate_local_filenames"]))],
                ["Metadata/PDF mismatches", str(len(audit["metadata_without_pdf"]) + len(audit["pdf_without_metadata"]))],
                ["Unavailable license fields", str(audit["missing_fields"]["license"])],
            ],
            [75 * mm, 79 * mm],
            small=True,
        ),
        p(
            "src/ingestion.py uses PyMuPDF through extract_pdf(), removes hyphenated line breaks and repeated whitespace, and skips extracted documents containing fewer than 50 words. src/chunking.py splits on sentence punctuation, targets approximately 400 words per chunk, and retains a 60-word overlap.",
            styles["body"],
        ),
        p(
            "Each chunk preserves paper_id, title, authors, year, journal, DOI, PMID, PMCID, evidence_type, section, and text. Chunk identifiers use the format {paper_id}_C{index:03d}, for example 13320052_C003.",
            styles["body"],
        ),
        p("Quality boundary", styles["callout_title"]),
        p(
            "The audit establishes file and metadata consistency, not medical relevance. Inspection of sample metadata shows records whose titles concern unrelated biomedical topics. Evidence-type labels are derived from search-query categories and are explicitly marked for manual review. Twenty-five license fields remain unavailable rather than being inferred.",
            styles["callout"],
        ),
        PageBreak(),
        p("5. Retrieval, fusion, reranking, and diversity", styles["h1"]),
        p(
            "The retrieval layer combines two complementary signals. BM25Retriever in src/bm25_retriever.py tokenizes each chunk and applies a custom term-frequency and length-normalized scoring method. VectorRetriever in src/vector_retriever.py encodes chunks with SentenceTransformer when available and compares normalized vectors using dot product.",
            styles["body"],
        ),
        p(
            "HybridRetriever retrieves a larger candidate pool from both systems, merges results by chunk_id, and calculates reciprocal-rank fusion using the configured constant 60. Reranker then applies the CrossEncoder to query/passage pairs. If the model is unavailable, it uses token overlap as a deterministic fallback.",
            styles["body"],
        ),
        p(
            "After reranking, filter_relevant() computes a relevance score using query overlap, reranker score, domain signal, and planning-topic signal. It applies population and non-planning-topic penalties where relevant. It stops after five selected passages by default and rejects additional passages from a paper after two selected chunks.",
            styles["body"],
        ),
        table(
            [
                ["Stage", "Output / safeguard"],
                ["BM25", "Lexical candidates with BM25-style score"],
                ["Vector search", "Semantic candidates with embedding similarity"],
                ["RRF", "Merged rank score keyed by chunk_id"],
                ["CrossEncoder", "Rerank score for query/passage pairs"],
                ["Relevance gate", "Minimum score 0.35 by default"],
                ["Diversity", "Maximum two selected chunks per paper"],
                ["Diagnostics", "Candidate count, kept count, reasons, scores"],
            ],
            [43 * mm, 111 * mm],
            small=True,
        ),
        p(
            "The persisted vector manifest records the embedding model, corpus hash, ordered chunk IDs, and format. Startup reuse is allowed only when those values and the vector count remain compatible; otherwise the implementation regenerates vectors.",
            styles["body"],
        ),
        PageBreak(),
        p("6. Ollama generation and response handling", styles["h1"]),
        p(
            "src/llm.py constructs a JSON-oriented prompt containing the original question, patient context, available retrieved chunk IDs, paper IDs, titles, and passage excerpts. Passage text is bounded to 1,600 characters per item to keep the prompt within the local model’s context budget.",
            styles["body"],
        ),
        p(
            "The system prompt instructs Ollama to use exact retrieved chunk IDs, avoid bibliography-number citations, preserve supplied patient values, avoid unsupported risk labels, avoid choosing a corridor from synthetic metrics, and state when evidence is insufficient. The configured model is llama3.2:3b served by the local Ollama endpoint.",
            styles["body"],
        ),
        p(
            "_parse_answer() handles Ollama’s response string, nested answer objects, invalid JSON, missing fields, and unexpected structures. HTTP, JSON, type, key, and value errors are returned as visible warnings rather than being converted into fabricated answers.",
            styles["body"],
        ),
        p("The response schema expects structured fields:", styles["body"]),
        table(
            [
                ["Answer section", "Purpose"],
                ["summary", "Short overall synthesis"],
                ["patient_findings", "Values explicitly supplied by the case"],
                ["risk_interpretation", "Cautious interpretation without undocumented thresholds"],
                ["corridor_comparison", "Structured supplied corridor values, not optimization"],
                ["evidence_supported_considerations", "Literature-grounded considerations"],
                ["uncertainties / missing_information", "Uncertainty and unavailable modalities"],
                ["limitations", "Scope and validation boundaries"],
                ["citations", "Normalized current-evidence references"],
            ],
            [65 * mm, 89 * mm],
            small=True,
        ),
        PageBreak(),
        p("7. Citations, verification, and numerical consistency", styles["h1"]),
        p(
            "Citation handling is implemented in normalize_citations() and validate_citations(). A citation is verified only if its identifier exactly matches a chunk in the current retrieved evidence. Supported response shapes include citation lists, single objects, nested citations/sources/references/evidence dictionaries, and ID-keyed dictionaries.",
            styles["body"],
        ),
        p(
            "Numeric bibliography references such as 9, [9, 10, 11], or 9–14 are not mapped to chunk IDs. Unresolved citations remain visible as unverified with the raw value and a reason. Verified citations are enriched from the retrieved evidence with the actual paper ID and title.",
            styles["body"],
        ),
        p(
            "verify_claims() associates structured claims with explicit cited chunk IDs where available and calculates token overlap against the cited passage. The configured threshold is 0.35. This is an automated lexical heuristic, not semantic entailment, fact verification, or clinical validation. Claims containing calculation or formula language are deliberately marked unverified when no project methodology defines the calculation.",
            styles["body"],
        ),
        p(
            "check_numerical_consistency() compares explicitly named generated patient values against the patient context. Omitted values are not discrepancies. Corridor values are checked against the matching corridor identifier. Tolerance allows ordinary integer or decimal rounding. The checker confirms consistency with supplied values; it does not validate medical formulas or risk calibration.",
            styles["body"],
        ),
        table(
            [
                ["Safeguard", "What it checks", "What it cannot establish"],
                ["Evidence validation", "Known chunk ID, passage, title", "Clinical relevance or authority"],
                ["Citation validation", "Current retrieved chunk membership", "Truth of cited claim"],
                ["Claim support", "Token overlap with cited passage", "Semantic entailment or causality"],
                ["Numerical checks", "Named values vs patient context", "Correctness of source methodology"],
                ["Interpretation warnings", "Unsupported labels/formulas/safety claims", "Clinical risk assessment"],
            ],
            [35 * mm, 65 * mm, 54 * mm],
            small=True,
        ),
        PageBreak(),
        p("8. Backend API and frontend integration", styles["h1"]),
        p(
            "The FastAPI application in api/main.py exposes a small local API. QueryRequest requires a question of at least three characters and defaults case_id to DEMO_001. QueryResponse contains the answer, citations, evidence, patient context, warnings, verification report, numerical checks, and retrieval diagnostics.",
            styles["body"],
        ),
        table(
            [
                ["Method", "Endpoint", "Purpose"],
                ["GET", "/", "Serve frontend index.html"],
                ["GET", "/api/health", "Health and knowledge-base existence"],
                ["GET", "/api/cases/{case_id}", "Load and validate a patient case"],
                ["POST", "/api/rag/query", "Run end-to-end RAG query"],
                ["POST", "/api/knowledge-base/rebuild", "Re-ingest local PDFs and rebuild pipeline"],
            ],
            [22 * mm, 67 * mm, 65 * mm],
            small=True,
        ),
        p(
            "frontend/app.js loads the case, posts questions, renders structured answer fields, displays warnings, and expands retrieved evidence. Evidence is rendered from the API’s passage field and includes chunk ID, title, paper ID, year, DOI, PMCID, and journal where available. API and rendering errors are visible rather than producing a blank answer section.",
            styles["body"],
        ),
        p(
            "The browser interface is intentionally lightweight: plain HTML, CSS, and JavaScript with no frontend framework. It contains a synthetic-data notice and provides no image viewer, segmentation overlay, trajectory optimizer, authentication, or clinical reporting workflow.",
            styles["body"],
        ),
        PageBreak(),
        p("9. Synthetic demonstration case", styles["h1"]),
        p(
            "data/demo_patient.json defines DEMO_001 and labels it SYNTHETIC_DEMO. patient_context.py validates the record through the extensible PatientContext schema and generates warnings for synthetic data and missing modalities.",
            styles["body"],
        ),
        table(
            [
                ["Category", "Supplied demonstration values"],
                ["Tumor", "Glioma; 42.3 ml total volume; 18.7 ml enhancing volume"],
                ["Classification", "demo_glioma; probability 0.87"],
                ["Risk", "risk_score 0.31; risk_uncertainty 0.12"],
                ["Missing modalities", "DTI, fMRI, vascular imaging"],
                ["Critical structure", "demo_structure_1"],
                ["Corridor A", "risk 0.18; coverage 0.91; length 44 mm"],
                ["Corridor B", "risk 0.24; coverage 0.96; length 37 mm"],
                ["Corridor C", "risk 0.31; coverage 0.98; length 29 mm"],
            ],
            [43 * mm, 111 * mm],
            small=True,
        ),
        p(
            "These values are synthetic test fixtures, not real patient measurements. The chatbot preserves and reports them but does not calculate them. In particular, the schema does not define a risk formula, calibration procedure, anatomical model, or optimization objective that could justify selecting a safest surgical corridor.",
            styles["body"],
        ),
        p(
            "PatientContext uses extra=\"allow\" and includes provenance, model information, quality indicators, modality fields, classification provenance, risk provenance, critical structures, and candidate corridors. This provides an adapter boundary for future modules without making those modules part of the current chatbot implementation.",
            styles["body"],
        ),
        PageBreak(),
        p("10. Testing and evaluation status", styles["h1"]),
        p(
            "The existing test suite was executed from the project root using the project virtual environment. The exact command was:",
            styles["body"],
        ),
        p(r".\\.venv\\Scripts\\python.exe -m pytest -q", styles["code"]),
        p("Observed result:", styles["body"]),
        p("23 passed, 1 warning in 49.57s", styles["result"]),
        p(
            "The warning was an existing StarletteDeprecationWarning concerning the use of HTTPX with starlette.testclient. No test failure occurred.",
            styles["body"],
        ),
        table(
            [
                ["Test group", "Coverage"],
                ["Core tests", "Chunking, metadata, BM25, reranking, evidence validation, patient warnings"],
                ["Citation tests", "Valid IDs, aliases, dictionary collections, missing and numeric references"],
                ["Numerical tests", "Matching values, omissions, rounding, mismatches, corridors A/B/C"],
                ["Safety checks", "Unsupported labels and undocumented formula claims"],
                ["API tests", "Health, case loading, unknown case, response shape, evidence passage field"],
            ],
            [43 * mm, 111 * mm],
            small=True,
        ),
        p(
            "data/evaluation_questions.json contains 25 realistic questions covering classification, resection, DTI/tractography, functional mapping, vascular planning, corridors, AI uncertainty, and patient context. All entries are pending_annotation with empty expected chunk IDs and candidate source papers. Consequently, Recall@K, Precision@K, citation correctness, and unsupported-claim-rate metrics are not reported.",
            styles["body"],
        ),
        p(
            "The passing automated tests establish behavior for the tested inputs. They do not establish retrieval quality, clinical accuracy, model calibration, latency, or patient safety.",
            styles["callout"],
        ),
        PageBreak(),
        p("11. Reproducibility", styles["h1"]),
        p("Environment setup:", styles["h2"]),
        p(r"python -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\nCopy-Item .env.example .env", styles["code"]),
        p("Literature and index build:", styles["h2"]),
        p(r".\\.venv\\Scripts\\python.exe scripts\\download_literature.py\n.\\.venv\\Scripts\\python.exe scripts\\build_knowledge_base.py\n.\\.venv\\Scripts\\python.exe scripts\\audit_corpus.py", styles["code"]),
        p("Ollama and API startup:", styles["h2"]),
        p(r"ollama serve\nollama pull llama3.2:3b\n.\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload", styles["code"]),
        p(
            "The expected local application URL is http://127.0.0.1:8000. The health endpoint is /api/health. The application requires the local chunks artifact and, for generated answers, a reachable Ollama service with the configured model. The vector index can be reused when its manifest remains compatible with the current corpus and embedding model.",
            styles["body"],
        ),
        p("Stored artifacts", styles["h2"]),
        table(
            [
                ["Artifact", "Role"],
                ["knowledge_base/chunks.json", "Chunk records and source metadata"],
                ["knowledge_base/vector_db/embeddings.npy", "Persisted dense vectors"],
                ["knowledge_base/vector_db/manifest.json", "Index compatibility metadata"],
                ["data/literature_metadata.csv", "Paper metadata and download state"],
                ["data/corpus_audit.json", "Corpus consistency summary"],
                ["data/demo_patient.json", "Synthetic demonstration case"],
            ],
            [75 * mm, 79 * mm],
            small=True,
        ),
        PageBreak(),
        p("12. Implementation status, limitations, and future improvements", styles["h1"]),
        table(
            [
                ["Area", "Status"],
                ["FastAPI, schemas, and frontend", "Implemented and covered by automated tests"],
                ["PDF ingestion and chunking", "Implemented; current artifacts contain 40 PDFs"],
                ["BM25, vectors, RRF, reranking", "Implemented; persisted vector artifact present"],
                ["Relevance threshold and source diversity", "Implemented with diagnostics and tests"],
                ["Ollama structured generation", "Implemented; fresh live behavior is not claimed by this report"],
                ["Citation normalization", "Implemented and unit-tested; current-evidence membership enforced"],
                ["Claim verification", "Partially implemented; lexical heuristic only"],
                ["Numerical consistency", "Implemented as deterministic comparison, not medical validation"],
                ["Retrieval evaluation", "Pending; requires human annotation"],
                ["Segmentation, DTI/fMRI, vascular modeling", "Not implemented as functional modules here"],
                ["Risk computation and corridor optimization", "Not implemented"],
                ["Clinical validation", "Not performed"],
            ],
            [70 * mm, 84 * mm],
            small=True,
        ),
        p("Recommended future work", styles["h2"]),
        p(
            "Future work should first review and annotate the literature corpus, verify evidence-type labels and licenses, assign human-reviewed relevance labels to the 25 evaluation questions, and calculate retrieval and citation metrics. Later extensions may integrate real segmentation, imaging, classification, spatial-risk, and trajectory modules behind the existing patient-context boundary. Any clinical use would require independent validation, governance, and safety review beyond this project.",
            styles["body"],
        ),
        p("Conclusion", styles["h2"]),
        p(
            "The workspace implements a coherent local RAG pipeline with auditable metadata, persisted retrieval artifacts, structured responses, visible warnings, and a synthetic demonstration case. Its strongest evidence is the source code, stored corpus/index artifacts, and passing automated tests. Its principal boundaries are the unreviewed corpus, pending retrieval annotations, heuristic claim checks, and absence of clinical validation.",
            styles["body"],
        ),
        PageBreak(),
        p("13. File-by-file implementation map", styles["h1"]),
        table(
            [
                ["File / directory", "Implementation role"],
                ["api/main.py", "FastAPI app, routes, pipeline lifecycle"],
                ["src/config.py", "Paths, model IDs, retrieval configuration"],
                ["src/rag_pipeline.py", "End-to-end query orchestration"],
                ["src/schemas.py", "Request, response, evidence, citation, patient models"],
                ["src/patient_context.py", "Patient loading and synthetic/missing-data warnings"],
                ["src/ingestion.py", "PDF extraction, cleaning, metadata filtering"],
                ["src/chunking.py", "Sentence-aware chunks and metadata propagation"],
                ["src/bm25_retriever.py", "Custom lexical scorer"],
                ["src/vector_retriever.py", "Embeddings, persistence, compatibility checks"],
                ["src/hybrid_retriever.py", "BM25/vector RRF fusion"],
                ["src/reranker.py", "CrossEncoder and lexical fallback"],
                ["src/relevance.py", "Query expansion, relevance gate, diversity"],
                ["src/llm.py", "Ollama prompt, parsing, citation normalization"],
                ["src/verification.py", "Evidence, claim, citation, numeric safeguards"],
                ["scripts/download_literature.py", "PMC/NCBI acquisition and status recording"],
                ["scripts/build_knowledge_base.py", "Chunk/index build and counts"],
                ["scripts/audit_corpus.py", "Corpus consistency audit"],
                ["scripts/evaluate.py", "Reviewed/pending evaluation reporting"],
                ["tests/test_core.py", "Retrieval, verification, numeric regression tests"],
                ["tests/test_api.py", "FastAPI endpoint and response-shape tests"],
                ["frontend/index.html", "UI structure and synthetic-data notice"],
                ["frontend/app.js", "API calls and answer/evidence rendering"],
                ["README.md", "Setup, architecture, limitations, evaluation notes"],
            ],
            [62 * mm, 92 * mm],
            small=True,
        ),
        Spacer(1, 8 * mm),
        p(
            "Evidence basis for this report: inspected source files, README, requirements manifest, stored corpus audit, patient JSON, evaluation dataset, vector manifest, test files, and the executed pytest result. No secrets, credentials, or private patient information are included.",
            styles["small"],
        ),
    ]
    return story


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#d7e0e2"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#607d86"))
        canvas.drawString(18 * mm, height - 11 * mm, "Brain-Tumor Surgical-Planning Knowledge-Base Chatbot")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.drawString(18 * mm, 10 * mm, "Capstone implementation report")
    canvas.restoreState()


def main() -> None:
    data = load_data()
    stylesheet = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=stylesheet["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=colors.HexColor("#173042"), alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", parent=stylesheet["Normal"], fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor("#607d86")),
        "eyebrow": ParagraphStyle("eyebrow", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=colors.HexColor("#e76f51"), tracking=1),
        "h1": ParagraphStyle("h1", parent=stylesheet["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=23, textColor=colors.HexColor("#264653"), spaceBefore=2, spaceAfter=10, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=stylesheet["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=colors.HexColor("#2a9d8f"), spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=colors.HexColor("#263238"), spaceAfter=7, alignment=TA_LEFT),
        "toc": ParagraphStyle("toc", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=11, leading=20, textColor=colors.HexColor("#344e58"), leftIndent=4),
        "caption": ParagraphStyle("caption", parent=stylesheet["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.HexColor("#607d86"), alignment=TA_CENTER, spaceAfter=8),
        "callout_title": ParagraphStyle("callout_title", parent=stylesheet["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=colors.HexColor("#e76f51"), spaceBefore=5, spaceAfter=3),
        "callout": ParagraphStyle("callout", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#59453d"), backColor=colors.HexColor("#fff4e6"), borderColor=colors.HexColor("#f4a261"), borderWidth=0.5, borderPadding=7, spaceAfter=8),
        "code": ParagraphStyle("code", parent=stylesheet["Code"], fontName="Courier", fontSize=8.2, leading=11, textColor=colors.HexColor("#173042"), backColor=colors.HexColor("#eef3f4"), borderColor=colors.HexColor("#c4d3d6"), borderWidth=0.5, borderPadding=6, spaceAfter=8),
        "result": ParagraphStyle("result", parent=stylesheet["BodyText"], fontName="Courier-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#1b6b59"), backColor=colors.HexColor("#e6f4ef"), borderColor=colors.HexColor("#2a9d8f"), borderWidth=0.5, borderPadding=7, spaceAfter=8),
        "small": ParagraphStyle("small", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=7.5, leading=10, textColor=colors.HexColor("#607d86")),
    }
    frame = Frame(18 * mm, 17 * mm, A4[0] - 36 * mm, A4[1] - 34 * mm, id="normal", leftPadding=0, rightPadding=0, topPadding=4 * mm, bottomPadding=3 * mm)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=17 * mm, title="Brain-Tumor Surgical-Planning Knowledge-Base Chatbot — Implementation Report", author="Capstone Project")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=on_page)])
    doc.build(build_story(data, styles))
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
