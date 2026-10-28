# 10 — RAG methodology and results

## Corpus and ingestion

The local audit records 40 papers/PDFs and 1,454 chunks. `ingestion.py` uses PyMuPDF, joins extracted page text, removes hyphenation across line breaks and collapses whitespace. Catalog rows must exist and have a downloaded status; documents under 50 words are skipped. `chunking.py` splits at sentence boundaries, targets 400 words with 60-word overlap and attaches paper ID/title/authors/year/journal/DOI/PMID/PMCID/type/section metadata. Audit says 25 license values are unavailable and evidence-type tags are query-derived and require manual review. The corpus is not a verified literature survey.

## Retrieval and filtering

Default embedding model is `sentence-transformers/all-MiniLM-L6-v2`; normalized sentence embeddings are searched by dot product. If optional model loading fails, code falls back to token-set vectors and a cosine-like intersection score. Persisted embeddings are reused only when corpus hash, chunk order and model name match. Lexical retrieval is a custom BM25-like implementation. Hybrid search requests up to max(3*top_k,10) results from each retriever and uses reciprocal-rank fusion with k=60. Cross-encoder default is `cross-encoder/ms-marco-MiniLM-L-6-v2`; a token-overlap reranker fallback is provided.

Defaults are top_k=5, maximum two chunks per paper, relevance threshold .35. Relevance combines meaningful query overlap (0.55), sigmoid-normalized reranker score (0.25), domain terms (0.20), then applies penalties for irrelevant/non-planning/population mismatch. Diagnostics track candidate/kept counts and filter reasons. If no passage passes the gate, generation is skipped.

## Generation, schema and verification

Default generator is local Ollama `llama3.2:3b`; environment variables may override defaults, and actual deployed values are not recorded. JSON-mode prompt demands exact retrieved chunk-ID citations and no invented facts, diagnoses, risk/safety claims or recommendations. Passage text is capped at 1,600 characters. FastAPI request takes a question (min length 3), case ID and optional extensible typed `PatientContext`; response contains answer, citations, evidence, warnings, retrieval diagnostics, verification and numerical checks.

Citation normalization only marks an ID verified when it maps to a retrieved chunk; citation source validation checks IDs/text/title. Automated claim support is token overlap >=.35 (methodology/calculation claims forced unverified), not semantic entailment or clinical validation. Numerical consistency checks named structured values against generated answer values using an implementation-defined tolerance; this checks consistency, not truth. Citation syntax/ID resolution is not proof that a cited passage supports the claim.

Integrated application passes only structured run context after explicit user authorization; raw MRI is not transmitted. RAG API is a separate service and generation availability is external to the Streamlit process.

## Results and open evaluation

All 25 evaluation questions are `pending_annotation`. `scripts/evaluate.py` returns pending/reviewed counts and, only when human-reviewed expected chunk IDs exist, computes mean retrieval Recall@K and Precision@K. Citation correctness and unsupported-claim rates require manual review. No retrieval score, citation metric, grounding score, numerical-consistency accuracy, or report-quality score is currently available. Corpus size is not a quality metric. No RAG evaluation plot or sample answer report was found.
