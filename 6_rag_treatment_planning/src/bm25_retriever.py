import re
from collections import Counter
from typing import Any


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.document_frequency = Counter(token for document in self.tokens for token in set(document))
        self.average_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        total = len(self.tokens)
        scores = []
        for index, document in enumerate(self.tokens):
            counts = Counter(document)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                idf = max(0.0, (total - self.document_frequency[token] + 0.5) / (self.document_frequency[token] + 0.5))
                length_factor = 1.5 * (1 - 0.75 + 0.75 * len(document) / max(self.average_length, 1))
                score += idf * counts[token] * 2.5 / (counts[token] + length_factor)
            scores.append(score)
        results = []
        for index in sorted(range(total), key=lambda item: scores[item], reverse=True)[:top_k]:
            result = dict(self.chunks[index])
            result["score"] = float(scores[index])
            results.append(result)
        return results


def bm25_search(chunks: list[dict[str, Any]], query: str, top_k: int = 10) -> list[dict[str, Any]]:
    return BM25Retriever(chunks).search(query, top_k)
