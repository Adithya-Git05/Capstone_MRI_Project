import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
metadata_path = ROOT / "data" / "literature_metadata.csv"
pdf_dir = ROOT / "data" / "raw_pdfs"
chunks_path = ROOT / "knowledge_base" / "chunks.json"
rows = list(csv.DictReader(metadata_path.open(encoding="utf-8", newline="")))
chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
pdfs = {path.name for path in pdf_dir.glob("*.pdf")}
metadata_files = {row["local_filename"] for row in rows}
report = {
    "metadata_rows": len(rows),
    "pdf_files": len(pdfs),
    "indexed_chunks": len(chunks),
    "indexed_papers": len({item["paper_id"] for item in chunks}),
    "duplicate_pmcids": [key for key, group in Counter(row["pmcid"] for row in rows).items() if group > 1],
    "duplicate_local_filenames": [key for key, group in Counter(row["local_filename"] for row in rows).items() if group > 1],
    "metadata_without_pdf": sorted(metadata_files - pdfs),
    "pdf_without_metadata": sorted(pdfs - metadata_files),
    "missing_fields": {
        field: sum(not row.get(field, "").strip() for row in rows)
        for field in ("title", "authors", "year", "doi", "pmcid", "source_url", "license")
    },
    "evidence_type_counts": dict(Counter(row["evidence_type"] for row in rows)),
    "evidence_type_note": "Labels are query-derived categories and require independent manual review.",
    "license_note": "Blank license values were retained as unavailable; no license was inferred.",
}
(ROOT / "data" / "corpus_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
