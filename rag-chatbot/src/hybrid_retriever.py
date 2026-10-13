from typing import Any
from .bm25_retriever import BM25Retriever
from .config import RRF_K, VECTOR_DIR
from .vector_retriever import VectorRetriever

class HybridRetriever:
    def __init__(self, chunks: list[dict[str, Any]], embedding_model: str = ""):
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks, embedding_model, VECTOR_DIR)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        limit = max(top_k * 3, 10)
        lexical = self.bm25.search(query, limit)
        semantic = self.vector.search(query, limit)
        merged: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(lexical, 1):
            merged.setdefault(item["chunk_id"], dict(item))["bm25_rank"] = rank
        for rank, item in enumerate(semantic, 1):
            merged.setdefault(item["chunk_id"], dict(item))["vector_rank"] = rank
        for item in merged.values():
            item["score"] = (1 / (RRF_K + item.get("bm25_rank", 999))) + (1 / (RRF_K + item.get("vector_rank", 999)))
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]
