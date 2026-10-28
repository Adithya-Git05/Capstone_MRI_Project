import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EMBEDDING_MODEL, METADATA_PATH, VECTOR_DIR
from src.ingestion import ingest_directory
from src.vector_retriever import VectorRetriever

chunks = ingest_directory()
VectorRetriever(chunks, EMBEDDING_MODEL).save(VECTOR_DIR)
papers = len({chunk["paper_id"] for chunk in chunks})
with METADATA_PATH.open(encoding="utf-8", newline="") as handle:
    metadata = list(csv.DictReader(handle))
downloaded = [
    row for row in metadata
    if row.get("download_status") in {"downloaded", "downloaded_existing"}
]
failed = [row for row in metadata if row.get("download_status") == "failed"]
manual_review = [row for row in metadata if row.get("manual_review") == "yes"]
print(f"Discovered: {len(metadata)} unique papers")
print(f"Verified PDFs: {len(downloaded)}")
print(f"Parsed successfully: {papers}")
print(f"Indexed: {papers} papers, {len(chunks)} chunks")
print(f"Failures: {len(failed)}")
if failed:
    print("Failed PMCID(s): " + ", ".join(row["pmcid"] for row in failed))
print(f"Manual review: {len(manual_review)}")
if manual_review:
    print("Manual-review PMCID(s): " + ", ".join(row["pmcid"] for row in manual_review))
