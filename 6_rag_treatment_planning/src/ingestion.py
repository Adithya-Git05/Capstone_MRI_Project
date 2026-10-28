import json
import re
from pathlib import Path
from typing import Any

from .chunking import make_chunks
from .config import CHUNKS_PATH, METADATA_PATH, RAW_PDF_DIR


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    import fitz
    document = fitz.open(path)
    return clean_text("\n".join(page.get_text() for page in document))


def load_metadata() -> list[dict[str, Any]]:
    import csv
    with METADATA_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ingest_directory(pdf_dir: Path = RAW_PDF_DIR) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    metadata = {row["local_filename"]: row for row in load_metadata()}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        paper = metadata.get(pdf_path.name)
        if not paper or paper.get("download_status") not in {"downloaded", "downloaded_existing"}:
            continue
        paper = dict(paper)
        paper["authors"] = [value.strip() for value in paper.get("authors", "").split(";") if value.strip()]
        text = extract_pdf(pdf_path)
        if len(text.split()) < 50:
            continue
        all_chunks.extend(make_chunks(text, paper))
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    return all_chunks
