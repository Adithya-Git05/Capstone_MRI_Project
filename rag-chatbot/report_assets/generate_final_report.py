from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

from generate_report import IngestionDiagram, WorkflowDiagram, load_data, table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Capstone_RAG_Chatbot_Implementation_Report_Final.pdf"


class CompactWorkflowDiagram(Flowable):
    def __init__(self, title: str, steps: list[str], width: float = 170 * mm, height: float = 78 * mm):
        super().__init__()
        self.title = title
        self.steps = steps
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.HexColor("#264653"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(0, self.height - 11, self.title)
        box_w, box_h = 37 * mm, 13 * mm
        x_gap, y_gap = 5 * mm, 5 * mm
        columns = 4
        rows = [self.steps[i:i + columns] for i in range(0, len(self.steps), columns)]
        for row_index, row in enumerate(rows):
            y = self.height - 25 - row_index * (box_h + y_gap)
            for col_index, label in enumerate(row):
                x = col_index * (box_w + x_gap)
                c.setFillColor(colors.HexColor("#e8f1f2"))
                c.setStrokeColor(colors.HexColor("#2a9d8f"))
                c.roundRect(x, y, box_w, box_h, 4, fill=1, stroke=1)
                c.setFillColor(colors.HexColor("#173042"))
                c.setFont("Helvetica", 7)
                words = label.split()
                lines, current = [], ""
                for word in words:
                    candidate = f"{current} {word}".strip()
                    if len(candidate) > 19 and current:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                start = y + box_h / 2 + (len(lines) - 1) * 3.5
                for line in lines:
                    c.drawCentredString(x + box_w / 2, start, line)
                    start -= 7
                if col_index < len(row) - 1:
                    ax = x + box_w
                    ay = y + box_h / 2
                    c.setStrokeColor(colors.HexColor("#e76f51"))
                    c.line(ax, ay, ax + x_gap - 1, ay)
                    c.line(ax + x_gap - 1, ay, ax + x_gap - 4, ay + 2.5)
                    c.line(ax + x_gap - 1, ay, ax + x_gap - 4, ay - 2.5)
        c.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ftitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=colors.HexColor("#173042"), spaceAfter=8),
        "subtitle": ParagraphStyle("fsubtitle", parent=base["Normal"], fontSize=13, leading=18, textColor=colors.HexColor("#607d86")),
        "eyebrow": ParagraphStyle("feyebrow", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#e76f51"), tracking=1),
        "h1": ParagraphStyle("fh1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#264653"), spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("fh2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2a9d8f"), spaceBefore=7, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("fbody", parent=base["BodyText"], fontSize=8.9, leading=12.2, textColor=colors.HexColor("#263238"), spaceAfter=6),
        "caption": ParagraphStyle("fcaption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.5, leading=9, textColor=colors.HexColor("#607d86"), alignment=TA_CENTER, spaceAfter=6),
        "callout": ParagraphStyle("fcallout", parent=base["BodyText"], fontSize=8.7, leading=12, textColor=colors.HexColor("#59453d"), backColor=colors.HexColor("#fff4e6"), borderColor=colors.HexColor("#f4a261"), borderWidth=0.5, borderPadding=6, spaceAfter=7),
        "code": ParagraphStyle("fcode", parent=base["Code"], fontName="Courier", fontSize=7.5, leading=10, textColor=colors.HexColor("#173042"), backColor=colors.HexColor("#eef3f4"), borderColor=colors.HexColor("#c4d3d6"), borderWidth=0.5, borderPadding=5, spaceAfter=6),
        "small": ParagraphStyle("fsmall", parent=base["BodyText"], fontSize=7.1, leading=9, textColor=colors.HexColor("#607d86")),
    }


def P(text, st):
    return Paragraph(text, st)


def footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        w, h = A4
        canvas.setStrokeColor(colors.HexColor("#d7e0e2"))
        canvas.line(18 * mm, h - 14 * mm, w - 18 * mm, h - 14 * mm)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(colors.HexColor("#607d86"))
        canvas.drawString(18 * mm, h - 11 * mm, "Brain-Tumor Surgical-Planning Knowledge-Base Chatbot")
        canvas.drawString(18 * mm, 10 * mm, "Capstone implementation report")
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_story(d, s):
    audit, patient, questions = d["audit"], d["patient"], d["questions"]
    story = [
        Spacer(1, 20 * mm),
        P("CAPSTONE IMPLEMENTATION REPORT", s["eyebrow"]),
        Spacer(1, 4 * mm),
        P("Brain-Tumor Surgical-Planning<br/>Knowledge-Base Chatbot", s["title"]),
        P("Presentation-ready technical report of the implemented chatbot subsystem", s["subtitle"]),
        Spacer(1, 16 * mm),
        Table(
            [["Project", "Undergraduate Capstone research prototype"],
             ["Runtime", "Python, FastAPI, Ollama, SentenceTransformers"],
             ["Demonstration case", "DEMO_001 — synthetic patient context"],
             ["Corpus snapshot", "40 PDFs, 40 metadata records, 1,454 indexed chunks"],
             ["Report date", "10 October 2026"]],
            colWidths=[42 * mm, 112 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#264653")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f3ede3")),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#b7c4c8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        Spacer(1, 13 * mm),
        P("Scope", s["h2"]),
        P("This report covers the implemented knowledge-base chatbot. It consumes supplied patient-context data and retrieved literature; imaging analysis and surgical optimization are separate platform modules or future integration points.", s["callout"]),
        PageBreak(),
        P("1. Executive summary", s["h1"]),
        P("The project implements a local retrieval-augmented generation application for evidence-oriented brain-tumor surgical-planning questions. A browser client sends a question and case identifier to FastAPI. The backend loads the selected patient context, retrieves literature through complementary lexical and vector methods, fuses and reranks candidate passages, filters for relevance and source diversity, and supplies the resulting evidence to a local Ollama model.", s["body"]),
        P("The response is returned as structured JSON rather than unconstrained prose. It contains answer sections, citations, retrieved passages, patient context, warnings, verification details, numerical checks, and retrieval diagnostics. This structure allows the frontend to show both the generated synthesis and the evidence supporting it.", s["body"]),
        P("The implementation emphasizes traceability: chunk identifiers are preserved from indexing through retrieval and citation normalization; patient-specific numbers are compared with the supplied context; and unsupported formula, risk-label, or corridor-safety claims are surfaced as warnings.", s["body"]),
        table([
            ["Implementation area", "Current implementation"],
            ["Knowledge base", "PMC/NCBI-acquired PDFs, PyMuPDF extraction, metadata-aware chunks"],
            ["Retrieval", "Custom BM25 + SentenceTransformer vectors + RRF + CrossEncoder"],
            ["Generation", "Ollama llama3.2:3b with JSON output instructions"],
            ["Verification", "Evidence membership, citation normalization, lexical claim support, numeric consistency"],
            ["Validation evidence", "23 automated tests passed; evaluation questions await annotation"],
        ], [48 * mm, 106 * mm], small=True),
        PageBreak(),
        P("2. System architecture", s["h1"]),
        P("The implemented query path is orchestrated by RAGPipeline.query() in src/rag_pipeline.py. It keeps patient facts, retrieved evidence, generated answer fields, and verification results separate so that the interface can distinguish supplied data from literature synthesis.", s["body"]),
        CompactWorkflowDiagram("Implemented query workflow", [
            "Browser UI", "FastAPI", "Patient context", "BM25 + vector", "RRF fusion",
            "CrossEncoder", "Relevance filter", "Evidence validation", "Ollama JSON",
            "Citation + numeric checks", "Structured response", "Frontend rendering",
        ]),
        P("Figure 1. Runtime query workflow implemented in the chatbot.", s["caption"]),
        P("When a question is broad, src/relevance.py expands retrieval using known case topics without inventing missing anatomy or imaging findings. If no passage meets the configured relevance gate, the pipeline returns an insufficient-evidence response instead of calling Ollama with unsupported context.", s["body"]),
        PageBreak(),
        P("3. Technology stack", s["h1"]),
        P("The application uses a deliberately local architecture. FastAPI and Uvicorn provide the HTTP service; Pydantic defines request, response, evidence, citation, and patient models; the frontend is plain HTML, CSS, and JavaScript; Ollama provides local generation; and SentenceTransformers provides embedding and reranking models.", s["body"]),
        table([
            ["Component", "Configured implementation"],
            ["Language / API", "Python; FastAPI; Uvicorn"],
            ["Patient and API schemas", "Pydantic models in src/schemas.py"],
            ["LLM", "Ollama: llama3.2:3b"],
            ["Embeddings", "sentence-transformers/all-MiniLM-L6-v2"],
            ["Reranking", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
            ["Fusion", "Reciprocal Rank Fusion; RRF_K = 60"],
            ["Final result size", "TOP_K = 5"],
            ["Evidence diversity", "MAX_CHUNKS_PER_PAPER = 2"],
            ["Relevance gate", "RELEVANCE_THRESHOLD = 0.35"],
            ["Vector artifact", "NumPy embeddings.npy with manifest.json"],
        ], [48 * mm, 106 * mm], small=True),
        P("The runtime source uses a custom BM25-style scorer and NumPy vector search. The declared FAISS and rank-bm25 dependencies are not required by the inspected retrieval implementation.", s["body"]),
        PageBreak(),
        P("4. Knowledge-base construction and indexing", s["h1"]),
        P("scripts/download_literature.py searches PMC through NCBI E-utilities, restricts discovery to open-access records, deduplicates PMCID values, preserves bibliographic metadata, and verifies downloaded PDF content. Download status, errors, local filenames, source URLs, and manual-review flags are recorded in data/literature_metadata.csv.", s["body"]),
        IngestionDiagram(height=53 * mm),
        P("Figure 2. Literature acquisition and indexing workflow.", s["caption"]),
        P("src/ingestion.py extracts text with PyMuPDF, removes hyphenated line breaks and repeated whitespace, and skips documents with fewer than 50 extracted words. src/chunking.py performs sentence-aware chunking with a target of 400 words and 60 words of overlap. Each chunk retains its paper metadata and receives an identifier such as 13320052_C003.", s["body"]),
        table([
            ["Stored evidence", "Current value"],
            ["PDF files / metadata rows", "40 / 40"],
            ["Indexed papers / chunks", "40 / 1,454"],
            ["Duplicate PMCIDs / filenames", "0 / 0"],
            ["Metadata/PDF mismatches", "0"],
            ["License fields unavailable", "25; left blank rather than inferred"],
        ], [75 * mm, 79 * mm], small=True),
        P("The corpus snapshot is structurally consistent according to data/corpus_audit.json. Its topical relevance remains subject to further review; counts alone do not establish that every source is equally suitable for every surgical-planning question.", s["callout"]),
        PageBreak(),
        P("5. Hybrid retrieval and reranking", s["h1"]),
        P("BM25Retriever tokenizes chunk text and calculates a custom lexical score using term frequency, document frequency, and length normalization. VectorRetriever uses normalized SentenceTransformer embeddings when available and compares query and document vectors. HybridRetriever merges both candidate lists by chunk_id and assigns a reciprocal-rank fusion score.", s["body"]),
        P("Reranker applies the configured CrossEncoder to query/passage pairs. If model loading is unavailable, the implementation falls back to token overlap so that the application can still operate in a reduced local mode.", s["body"]),
        table([
            ["Stage", "Purpose"],
            ["BM25 retrieval", "Capture exact terminology and lexical matches"],
            ["Vector retrieval", "Capture semantically related passages"],
            ["RRF fusion", "Combine complementary ranked lists using constant 60"],
            ["CrossEncoder", "Reorder candidates using query-passage relevance"],
            ["Relevance filtering", "Apply query, domain, and surgical-planning signals"],
            ["Evidence diversity", "Limit final selection to two chunks per paper"],
        ], [48 * mm, 106 * mm], small=True),
        P("The vector index is persisted under knowledge_base/vector_db/. Its manifest records the embedding model, corpus hash, ordered chunk IDs, and vector format. Loading is allowed only when these values remain compatible with the current chunks; otherwise the builder regenerates the artifact.", s["body"]),
        PageBreak(),
        P("6. LLM response generation", s["h1"]),
        P("src/llm.py prepares a bounded evidence context containing the user question, patient context, available chunk IDs, paper identifiers, titles, and passage excerpts. The system prompt requires valid JSON and instructs the model to cite exact retrieved chunk IDs, separate patient facts from literature findings, avoid unsupported risk labels, and state when evidence is insufficient.", s["body"]),
        P("_parse_answer() handles the response field returned by Ollama, nested answer objects, invalid JSON, missing answer fields, and unexpected response shapes. Generation failures become visible warnings rather than silent fallback answers.", s["body"]),
        table([
            ["Response field", "Role in the UI/API"],
            ["summary", "Overall evidence-oriented synthesis"],
            ["patient_findings", "Values explicitly supplied by the case"],
            ["risk_interpretation", "Cautious interpretation without undocumented thresholds"],
            ["corridor_comparison", "Supplied corridor values and comparison context"],
            ["evidence_supported_considerations", "Literature-based considerations"],
            ["uncertainties / missing_information", "Uncertainty and unavailable data"],
            ["limitations", "Scope and validation boundary"],
            ["citations", "Normalized references to current retrieved evidence"],
        ], [65 * mm, 89 * mm], small=True),
        P("Ollama is local and optional for unit tests, but it is required for generated answers during normal interactive use.", s["callout"]),
        PageBreak(),
        P("7. Citation and response verification", s["h1"]),
        P("Citation normalization in src/llm.py accepts lists, individual objects, nested citation/source/reference collections, and explicit identifier aliases. A citation becomes verified only when its identifier exactly matches a chunk in the current retrieved evidence. Numeric bibliography references are never assumed to represent local chunk IDs.", s["body"]),
        P("Verified citations are enriched with the actual paper ID and title. Unresolved or malformed entries remain visible as unverified with their raw value and reason. This preserves uncertainty instead of treating an identifier’s presence as proof that a claim is correct.", s["body"]),
        P("verify_claims() associates structured claims with cited passages and calculates token overlap. The method is explicitly a lexical-support heuristic with a 0.35 threshold; it does not establish semantic entailment, causality, clinical correctness, or study quality. Claims describing undocumented calculations or formulas are marked unverified.", s["body"]),
        P("check_numerical_consistency() compares explicitly named generated values against the supplied patient context. Omitted values are not discrepancies, ordinary rounding is tolerated, and corridor values are matched by corridor identifier. The check detects consistency with supplied data; it does not validate medical formulas or risk calibration.", s["body"]),
        table([
            ["Check", "Implemented behavior"],
            ["Evidence", "Requires known chunk ID, passage text, and title"],
            ["Citation", "Requires membership in current retrieved evidence"],
            ["Claim support", "Token-overlap support heuristic"],
            ["Numbers", "Field-specific comparison with supplied context"],
            ["Warnings", "Unsupported labels, formulas, and safest-corridor claims are surfaced"],
        ], [42 * mm, 112 * mm], small=True),
        PageBreak(),
        P("8. Backend and frontend integration", s["h1"]),
        P("api/main.py defines the local FastAPI application and keeps the pipeline lazy-loaded. The response model is QueryResponse from src/schemas.py, which exposes answer fields together with citations, retrieved_evidence, patient_context, warnings, verification, numerical_checks, and retrieval_diagnostics.", s["body"]),
        table([
            ["Method", "Endpoint", "Purpose"],
            ["GET", "/", "Serve the browser application"],
            ["GET", "/api/health", "Return service and knowledge-base status"],
            ["GET", "/api/cases/{case_id}", "Load a validated case"],
            ["POST", "/api/rag/query", "Run the complete RAG workflow"],
            ["POST", "/api/knowledge-base/rebuild", "Re-ingest local PDFs and rebuild the pipeline"],
        ], [22 * mm, 67 * mm, 65 * mm], small=True),
        P("frontend/app.js loads DEMO_001, sends JSON requests, displays loading and error states, renders structured answer sections, and expands retrieved evidence using the API’s passage property. It also displays chunk IDs, titles, and available citation metadata.", s["body"]),
        P("The interface is intentionally lightweight and local: HTML, CSS, and JavaScript with no additional frontend framework.", s["body"]),
        P("Demonstration context", s["h2"]),
        P("data/demo_patient.json defines DEMO_001 as SYNTHETIC_DEMO. It supplies illustrative tumor measurements, classification and risk values, missing modalities, a critical structure, and candidate corridor metrics. These values are preserved for testing and response consistency; they are not real clinical measurements and do not define a surgical recommendation.", s["callout"]),
        P("9. Testing and results", s["h1"]),
        P("The latest available automated test result is:", s["body"]),
        P("23 passed, 1 warning", s["code"]),
        P("tests/test_core.py covers chunking, metadata propagation, BM25 retrieval, reranking, evidence validation, patient warnings, relevance filtering, citation normalization, malformed and numeric citations, numerical omission and mismatch handling, corridor values, and unsupported formula claims. tests/test_api.py covers health, case loading, unknown cases, query response structure, verification fields, numerical checks, retrieval diagnostics, and evidence passage rendering.", s["body"]),
        table([
            ["Validation area", "Result / status"],
            ["Automated regression suite", "23 passed; 1 existing dependency warning"],
            ["Corpus snapshot", "40 PDFs, 40 metadata rows, 40 papers, 1,454 chunks"],
            ["Persisted vector artifact", "Present with compatibility manifest"],
            ["Evaluation questions", "25; pending human annotation"],
            ["Retrieval metrics", "Not reported without reviewed ground truth"],
            ["Clinical performance", "Not measured or claimed"],
        ], [60 * mm, 94 * mm], small=True),
        P("The 25-question evaluation set covers glioma classification, resection, DTI/tractography, functional mapping, vascular planning, corridor questions, AI uncertainty, and patient context. Because expected source IDs have not been manually reviewed, Recall@K, Precision@K, citation correctness, and unsupported-claim rates are not presented.", s["body"]),
        P("The automated tests demonstrate behavior for defined inputs and response shapes. They do not establish retrieval quality, clinical accuracy, model calibration, latency, or patient safety.", s["callout"]),
        PageBreak(),
        P("10. Reproducibility and conclusion", s["h1"]),
        P("Create the environment and install the declared dependencies:", s["h2"]),
        P(r"python -m venv .venv<br/>.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt<br/>Copy-Item .env.example .env", s["code"]),
        P("Acquire and index literature:", s["h2"]),
        P(r".\\.venv\\Scripts\\python.exe scripts\\download_literature.py<br/>.\\.venv\\Scripts\\python.exe scripts\\build_knowledge_base.py<br/>.\\.venv\\Scripts\\python.exe scripts\\audit_corpus.py", s["code"]),
        P("Start Ollama and the API:", s["h2"]),
        P(r"ollama serve<br/>ollama pull llama3.2:3b<br/>.\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload", s["code"]),
        P("Open http://127.0.0.1:8000. The health endpoint is /api/health and the query endpoint is /api/rag/query.", s["body"]),
        P("Conclusion", s["h2"]),
        P("The implemented chatbot provides a clear local evidence-synthesis workflow: literature is acquired and indexed with traceable metadata; BM25 and vector retrieval are fused and reranked; evidence diversity and relevance are controlled; Ollama produces structured JSON; and citations, claims, and patient-specific numbers receive explicit automated checks. The architecture is compact, reproducible, and suitable as a Capstone research prototype.", s["body"]),
        P("The principal validation boundary is intentional: the demonstration patient is synthetic, claim support is heuristic, evaluation ground truth is pending, and no clinical validation is claimed. These boundaries keep the chatbot focused on evidence retrieval and response traceability while allowing future integration with other platform modules.", s["callout"]),
        P("Useful implementation references: src/rag_pipeline.py, src/llm.py, src/verification.py, src/vector_retriever.py, api/main.py, frontend/app.js, scripts/build_knowledge_base.py, tests/test_core.py, and tests/test_api.py.", s["small"]),
    ]
    return story


def main():
    d = load_data()
    s = styles()
    frame = Frame(18 * mm, 17 * mm, A4[0] - 36 * mm, A4[1] - 34 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=4 * mm, bottomPadding=3 * mm)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=17 * mm, title="Brain-Tumor Surgical-Planning Knowledge-Base Chatbot — Final Implementation Report", author="Capstone Project")
    doc.addPageTemplates([PageTemplate(id="final", frames=[frame], onPage=footer)])
    doc.build(build_story(d, s))
    print(OUTPUT)


if __name__ == "__main__":
    main()
