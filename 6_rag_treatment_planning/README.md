# Brain Tumor Surgical Planning RAG Prototype

A local undergraduate research prototype that retrieves supplied literature, verifies source metadata, loads an explicitly synthetic patient case, and sends grounded context to Gemini. It is not clinically validated and does not make autonomous medical decisions.

## Architecture

PDFs are extracted with PyMuPDF and split into sentence-aware overlapping chunks. Hybrid retrieval combines BM25 and vector retrieval with reciprocal rank fusion. `sentence-transformers` and its cross-encoder are used when their models are available; deterministic lexical fallbacks keep retrieval runnable offline. The builder writes a persisted embedding artifact under `knowledge_base/vector_db/`, including a corpus hash, ordered chunk IDs, and embedding model name. The application reuses it only when those values are compatible; otherwise it regenerates the artifact. Evidence is source-checked before Gemini receives it.

After reranking, evidence passes a configurable relevance gate (`RELEVANCE_THRESHOLD`, default `0.35`). The gate combines query-term coverage, reranker score, glioma/neurosurgical domain signal, and generic population/topic mismatch penalties. Broad case questions are expanded with only the known case topics; they do not infer missing age, anatomy, or imaging. The reranker considers a bounded candidate pool, and evidence is then limited by `MAX_CHUNKS_PER_PAPER`, so fewer than `TOP_K` passages are returned when the remaining candidates are irrelevant. The API exposes `retrieval_diagnostics` with candidate counts, scores, and filtering reasons. Gemini can still summarize supplied MRI results when no literature passage passes; in that case, it is instructed to make no literature claims.

## Setup (PowerShell)

Run these commands from `6_rag_treatment_planning/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `6_rag_treatment_planning/.env`. The provided `.env.example` selects `gemini-3.5-flash`; update `GEMINI_MODEL` there to use another supported Gemini model. The API key is read from the environment and is never stored in source code.

## Build and run

The literature chunks, downloaded PDFs, and vector index are generated local
artifacts and are not stored in Git. A fresh checkout must acquire the open
access papers and build its own index before literature retrieval is available.

```powershell
.\.venv\Scripts\python.exe scripts\build_knowledge_base.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000`. The UI labels `Patient 001` as synthetic data. API endpoints: `GET /api/health`, `GET /api/cases/{case_id}`, `POST /api/rag/query`, and local-development `POST /api/knowledge-base/rebuild`.

## Literature

```powershell
.\.venv\Scripts\python.exe scripts\download_literature.py
.\.venv\Scripts\python.exe scripts\build_knowledge_base.py
```

The downloader uses NCBI PMC E-utilities and open-access records only. It records metadata and download failures; it does not invent papers, identifiers, or licenses. Review `data/literature_metadata.csv` before use. Add a row there and a matching PDF in `data/raw_pdfs/` to add literature.

## Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate.py
```

`data/evaluation_questions.json` contains 25 realistic questions, including the five exact retrieval-regression questions. They are explicitly marked `pending_annotation` with empty expected IDs until a human reviewer assigns relevant paper/chunk IDs. `scripts/evaluate.py` reports pending status and calculates Recall@K/Precision@K only for entries marked `reviewed`; citation correctness and unsupported-claim rate require manual claim review. No scores are fabricated.

```powershell
.\.venv\Scripts\python.exe scripts\audit_corpus.py
```

The audit is stored in `data/corpus_audit.json`. Evidence-type labels are query-derived categories, not independently verified study-design labels, and blank licensing fields remain unavailable.

## Patient interface

`data/demo_patient.json` supplies only Patient 001's clearly labeled fictional history; it does not contain example tumor measurements, classification, risk, or corridor results. In the integrated workflow, those findings are supplied only from the active MRI inference run and omitted when a model stage has not produced them. Raw images and voxel arrays are not sent to Gemini. `src/patient_context.py` validates the case ID through the extensible `PatientContext` schema, which accepts provenance, model/version, quality, modality, segmentation, classification, risk, anatomy, and corridor fields. The RAG layer must preserve supplied model values and must not recalculate volume, risk, anatomy, or corridor safety. The chat presents one text summary with verified knowledge-base citations instead of separate source-passage panels; no unsupported literature statements are added when retrieval finds no relevant passage. Each API response includes automated claim-support and patient-number consistency reports; these are not clinical validation.

## Limitations

The history is synthetic, and MRI model outputs are experimental and not clinically validated. Retrieval quality depends on downloaded documents and model availability. Gemini is required for generated answers and receives the question, verified passages, and supplied structured patient context; configure an authorized Gemini API key and do not send identifiable clinical information. Raw MRI images are not sent by the RAG integration. Automated claim checks use lexical overlap and cannot establish clinical entailment. Tractography, functional mapping, vascular imaging, risk estimation, and corridor optimization are not implemented. This system is for research and education, not patient care.
