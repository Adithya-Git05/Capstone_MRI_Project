"""Discover and download relevant open-access PMC articles with auditable status."""
import csv
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from src.config import METADATA_PATH, RAW_PDF_DIR

QUERIES = [
    ("glioma AND (classification OR molecular OR IDH OR 1p19q)", "classification"),
    ("glioma AND (guideline OR consensus OR recommendations)", "guideline"),
    ("glioma AND (surgery OR resection OR intraoperative)", "primary_study"),
    ("glioma AND (DTI OR tractography OR diffusion tensor)", "systematic_review"),
    ("glioma AND (functional mapping OR awake surgery OR language)", "consensus_statement"),
    ("glioma AND (vascular OR angiography OR vessel)", "primary_study"),
    ("glioma AND (surgical planning OR operative planning OR extent of resection)", "systematic_review"),
]
TARGET_PAPERS = 40
MAX_RESULTS_PER_QUERY = 10
CSV_FIELDS = [
    "paper_id", "title", "authors", "year", "journal", "pmid", "pmcid",
    "doi", "topic", "evidence_type", "license", "source_url",
    "local_filename", "download_status", "download_error", "manual_review",
]
HEADERS = {"User-Agent": "brain-tumor-rag-research-prototype/1.0"}


def _get(client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
    response = client.get(url, headers=HEADERS, **kwargs)
    response.raise_for_status()
    return response


def _summary_metadata(summary: dict[str, Any], pmcid: str, topic: str) -> dict[str, str]:
    article_ids = summary.get("articleids", [])
    if isinstance(article_ids, list):
        article_ids = {
            item.get("idtype", ""): item.get("value", "")
            for item in article_ids
            if isinstance(item, dict)
        }
    doi = article_ids.get("doi", "")
    pmid = article_ids.get("pmid", "")
    authors = "; ".join(
        author.get("name", "").strip()
        for author in summary.get("authors", [])
        if author.get("name", "").strip()
    )
    return {
        "paper_id": pmcid.removeprefix("PMC"),
        "title": summary.get("title", "").strip(),
        "authors": authors,
        "year": summary.get("pubdate", "")[:4],
        "journal": summary.get("fulljournalname", "").strip(),
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "topic": topic,
        "evidence_type": topic,
        "license": "",
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "local_filename": f"{pmcid}.pdf",
        "download_status": "discovered",
        "download_error": "",
        "manual_review": "no",
    }


def _download_pdf(client: httpx.Client, pmcid: str, destination: Path) -> tuple[bool, str, str]:
    listing_url = "https://pmc-oa-opendata.s3.amazonaws.com/"
    listing_xml = _get(
        client,
        listing_url,
        params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
        timeout=30,
    ).text
    root = ET.fromstring(listing_xml)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    versions = [
        item.text.rstrip("/")
        for item in root.findall("s3:CommonPrefixes/s3:Prefix", namespace)
        if item.text
    ]
    if not versions:
        return False, "PMC Cloud Service has no downloadable article version.", ""
    version = sorted(versions)[-1]
    metadata_url = f"https://pmc-oa-opendata.s3.amazonaws.com/{version}/{version}.json"
    metadata = _get(client, metadata_url, timeout=30).json()
    pdf_url = metadata.get("pdf_url")
    if not pdf_url:
        return False, "PMC Cloud Service article version has no permitted PDF.", metadata.get("license_code", "")
    if pdf_url.startswith("s3://pmc-oa-opendata/"):
        pdf_url = pdf_url.replace(
            "s3://pmc-oa-opendata/",
            "https://pmc-oa-opendata.s3.amazonaws.com/",
            1,
        )
    pdf_response = _get(client, pdf_url, timeout=90)
    try:
        content = pdf_response.content
    except OSError as exc:
        return False, f"Could not read PMC PDF response: {exc}", metadata.get("license_code", "")
    if len(content) < 1024 or not content.startswith(b"%PDF-"):
        return False, "Downloaded content is not a valid PDF.", metadata.get("license_code", "")
    temporary = destination.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return True, "", metadata.get("license_code", "")


def main():
    rows_by_pmcid: dict[str, dict[str, str]] = {}
    with httpx.Client(follow_redirects=True) as client:
        for query, evidence_type in QUERIES:
            params = {
                "db": "pmc",
                "term": (
                    f"({query})[Title/Abstract] AND open access[filter] "
                    'AND ("2010"[PDAT] : "2025"[PDAT])'
                ),
                "retmode": "json",
                "retmax": MAX_RESULTS_PER_QUERY,
            }
            search = _get(
                client,
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                timeout=30,
            ).json()
            for raw_id in search["esearchresult"]["idlist"]:
                pmcid = f"PMC{raw_id}"
                if pmcid in rows_by_pmcid or len(rows_by_pmcid) >= TARGET_PAPERS:
                    continue
                try:
                    summary = _get(
                        client,
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={"db": "pmc", "id": raw_id, "retmode": "json"},
                        timeout=30,
                    ).json()["result"].get(raw_id, {})
                    row = _summary_metadata(summary, pmcid, evidence_type)
                    if not row["title"]:
                        row["download_status"] = "failed"
                        row["download_error"] = "PMC summary did not contain a title."
                        row["manual_review"] = "yes"
                    else:
                        destination = RAW_PDF_DIR / row["local_filename"]
                        if destination.exists() and destination.stat().st_size >= 1024:
                            row["download_status"] = "downloaded_existing"
                        else:
                            ok, error, license_code = _download_pdf(client, pmcid, destination)
                            row["license"] = license_code
                            row["download_status"] = "downloaded" if ok else "failed"
                            row["download_error"] = error
                            row["manual_review"] = "yes" if not ok else "no"
                    rows_by_pmcid[pmcid] = row
                except (httpx.HTTPError, KeyError, ValueError, OSError) as exc:
                    rows_by_pmcid[pmcid] = {
                        "paper_id": raw_id, "title": "", "authors": "", "year": "",
                        "journal": "", "pmid": "", "pmcid": pmcid, "doi": "",
                        "topic": evidence_type, "evidence_type": evidence_type,
                        "license": "", "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                        "local_filename": f"{pmcid}.pdf", "download_status": "failed",
                        "download_error": str(exc), "manual_review": "yes",
                    }
                time.sleep(0.34)
            if len(rows_by_pmcid) >= TARGET_PAPERS:
                break
    with METADATA_PATH.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_by_pmcid.values())
    counts = {}
    for row in rows_by_pmcid.values():
        counts[row["download_status"]] = counts.get(row["download_status"], 0) + 1
    print(f"Discovered {len(rows_by_pmcid)} unique PMC records.")
    print(f"Download status: {counts}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"PDFs: {RAW_PDF_DIR}")

if __name__ == '__main__': main()
