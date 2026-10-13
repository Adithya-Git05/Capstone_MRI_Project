import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _corpus_hash(chunks: list[dict[str, Any]]) -> str:
    chunk_ids = "\n".join(chunk["chunk_id"] for chunk in chunks)
    return hashlib.sha256(chunk_ids.encode("utf-8")).hexdigest()


class VectorRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = "",
        index_directory: str | Path | None = None,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.model = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        self.vectors = self._load_compatible(index_directory) if index_directory else None
        if self.vectors is None:
            self.vectors = [self._encode(chunk["text"]) for chunk in chunks]

    def _load_compatible(self, directory: str | Path) -> list[Any] | None:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("corpus_hash") != _corpus_hash(self.chunks)
                or manifest.get("chunk_ids") != [chunk["chunk_id"] for chunk in self.chunks]
                or manifest.get("embedding_model") != self.model_name
            ):
                return None
            if manifest.get("format") == "numpy":
                import numpy as np
                values = np.load(directory / "embeddings.npy")
                if len(values) != len(self.chunks):
                    return None
                return list(values)
            if manifest.get("format") == "tokens":
                return [set(value) for value in manifest.get("vectors", [])]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return None

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": "numpy" if self.model else "tokens",
            "embedding_model": self.model_name,
            "corpus_hash": _corpus_hash(self.chunks),
            "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
        }
        if self.model:
            import numpy as np
            np.save(directory / "embeddings.npy", np.asarray(self.vectors))
        else:
            manifest["vectors"] = [sorted(vector) for vector in self.vectors]
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _encode(self, text: str) -> Any:
        if self.model:
            return self.model.encode(text, normalize_embeddings=True)
        return _tokens(text)

    def _similarity(self, left: Any, right: Any) -> float:
        if self.model:
            return float(left @ right)
        if not left and not right:
            return 0.0
        return len(left & right) / math.sqrt(max(len(left) * len(right), 1))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        encoded = self._encode(query)
        scores = [self._similarity(encoded, vector) for vector in self.vectors]
        results = []
        for index in sorted(range(len(self.chunks)), key=lambda item: scores[item], reverse=True)[:top_k]:
            result = dict(self.chunks[index])
            result["score"] = float(scores[index])
            results.append(result)
        return results


def vector_search(chunks: list[dict[str, Any]], query: str, top_k: int = 10, model_name: str = "") -> list[dict[str, Any]]:
    return VectorRetriever(chunks, model_name).search(query, top_k)
