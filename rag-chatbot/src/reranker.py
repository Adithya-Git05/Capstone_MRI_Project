from typing import Any
from .bm25_retriever import tokenize

class Reranker:
    def __init__(self, model_name: str = ""):
        self.model = None
        try:
            if model_name:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.model:
            scores = self.model.predict([(query, item["text"]) for item in candidates])
        else:
            query_tokens = set(tokenize(query))
            scores = [len(query_tokens & set(tokenize(item["text"]))) for item in candidates]
        ranked = []
        for item, score in zip(candidates, scores):
            copied = dict(item)
            copied["rerank_score"] = float(score)
            ranked.append(copied)
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
