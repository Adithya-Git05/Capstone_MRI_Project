import re
from typing import Any


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def chunk_text(text: str, target_words: int = 400, overlap_words: int = 60) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > target_words:
            chunks.append(" ".join(current))
            overlap = " ".join(current).split()[-overlap_words:]
            current = [" ".join(overlap)] if overlap else []
            count = len(overlap)
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunks(text: str, paper: dict[str, Any], section: str = "Body") -> list[dict[str, Any]]:
    chunks = chunk_text(text)
    return [
        {
            "chunk_id": f"{paper['paper_id']}_C{index:03d}",
            "paper_id": paper["paper_id"],
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "journal": paper.get("journal", ""),
            "doi": paper.get("doi", ""),
            "pmid": paper.get("pmid", ""),
            "pmcid": paper.get("pmcid", ""),
            "evidence_type": paper.get("evidence_type", "unknown"),
            "section": section,
            "text": value,
        }
        for index, value in enumerate(chunks, 1)
    ]
