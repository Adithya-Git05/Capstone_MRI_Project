import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_pipeline import RAGPipeline

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_questions.json"


def main() -> None:
    dataset: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    pending = [item for item in dataset if item.get("status") != "reviewed"]
    reviewed = [item for item in dataset if item.get("status") == "reviewed"]
    print(json.dumps({"questions": len(dataset), "pending_annotation": len(pending), "reviewed": len(reviewed)}, indent=2))
    if not reviewed:
        print("No human-reviewed ground truth is present; Recall@K, Precision@K, citation correctness, and unsupported-claim rate are not reported.")
        return
    totals = {"recall": 0.0, "precision": 0.0}
    for item in reviewed:
        result = pipeline.query(item["question"], "DEMO_001")
        retrieved = {e["chunk_id"] for e in result["retrieved_evidence"]}
        expected = set(item["expected_chunk_ids"])
        totals["recall"] += len(retrieved & expected) / max(len(expected), 1)
        totals["precision"] += len(retrieved & expected) / max(len(retrieved), 1)
    count = len(reviewed)
    print(json.dumps({"reviewed_recall_at_k": totals["recall"] / count, "reviewed_precision_at_k": totals["precision"] / count}, indent=2))
    print("Citation correctness and unsupported-claim rate require manual review of generated claims and are not inferred from retrieval overlap.")


if __name__ == "__main__":
    main()
