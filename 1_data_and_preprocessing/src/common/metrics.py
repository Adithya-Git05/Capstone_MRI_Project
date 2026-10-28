from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    targets_tensor = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    if logits_tensor.ndim != 2 or logits_tensor.shape[0] != targets_tensor.numel():
        raise ValueError("Logits must have shape [samples, classes] matching targets")
    if targets_tensor.numel() == 0 or not torch.isfinite(logits_tensor).all():
        raise ValueError("Calibration requires nonempty finite logits")
    if torch.any(targets_tensor < 0) or torch.any(targets_tensor >= logits_tensor.shape[1]):
        raise ValueError("Calibration targets contain an unknown class index")

    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        bounded_log_temperature = log_temperature.clamp(
            min=float(np.log(0.05)), max=float(np.log(20.0))
        )
        loss = F.cross_entropy(
            logits_tensor / bounded_log_temperature.exp(), targets_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(
        min=float(np.log(0.05)), max=float(np.log(20.0))
    ).exp())


def probability_quality_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("Probabilities must have shape [samples, classes] matching targets")
    if targets.size == 0 or bins < 1:
        raise ValueError("Probability metrics require samples and at least one bin")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(targets < 0)
        or np.any(targets >= probabilities.shape[1])
    ):
        raise ValueError("Probabilities or targets contain invalid values")
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have positive sum")
    probabilities = probabilities / row_sums[:, None]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    bin_indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    reliability_bins = []
    expected_calibration_error = 0.0
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        if count:
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            expected_calibration_error += count / targets.size * abs(
                accuracy - mean_confidence
            )
        else:
            mean_confidence = None
            accuracy = None
        reliability_bins.append(
            {
                "lower_bound": bin_index / bins,
                "upper_bound": (bin_index + 1) / bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    return {
        "negative_log_likelihood": float(-np.log(clipped[np.arange(targets.size), targets]).mean()),
        "multiclass_brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error": float(expected_calibration_error),
        "reliability_bins": reliability_bins,
    }


def _binary_curve(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.uint8).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return {"auroc": None, "auprc": None, "fpr": [], "tpr": [], "precision": [], "recall": []}

    order = np.argsort(-score, kind="mergesort")
    sorted_scores = score[order]
    sorted_target = target[order]
    end_points = np.r_[np.flatnonzero(np.diff(sorted_scores)), target.size - 1]
    true_positive = np.cumsum(sorted_target)[end_points].astype(np.float64)
    false_positive = (1 + end_points - true_positive).astype(np.float64)
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, false_positive / negatives]
    recall_curve = np.r_[0.0, recall]
    precision_curve = np.r_[1.0, precision]
    return {
        "auroc": float(np.trapz(tpr, fpr)),
        "auprc": float(np.sum(np.diff(recall_curve) * precision)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision_curve.tolist(),
        "recall": recall_curve.tolist(),
    }


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (targets.size, len(class_names)):
        raise ValueError("Probability shape must be [samples, number of classes]")
    if targets.size == 0:
        raise ValueError("Cannot evaluate an empty classification split")
    if np.any(targets < 0) or np.any(targets >= len(class_names)):
        raise ValueError("Targets contain an unknown class index")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and non-negative")

    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    support = confusion.sum(axis=1)
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = support - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    present = support > 0
    total = int(targets.size)
    normalized_confusion = confusion / np.maximum(support[:, None], 1)

    per_class = []
    curves = {}
    for index, name in enumerate(class_names):
        curve = _binary_curve(targets == index, probabilities[:, index])
        curves[name] = curve
        per_class.append(
            {
                "class": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "auroc_ovr": curve["auroc"],
                "auprc_ovr": curve["auprc"],
            }
        )

    return {
        "sample_count": total,
        "accuracy": float(tp.sum() / total),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision.mean()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "macro_recall": float(recall.mean()),
        "weighted_recall": float(np.average(recall, weights=support)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
        "row_normalized_confusion_matrix": normalized_confusion.tolist(),
        "per_class": per_class,
        "curves_ovr": curves,
    }


def binary_segmentation_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Segmentation prediction and target shapes differ")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    if denominator == 0:
        return {"dice": 1.0, "iou": 1.0}
    return {"dice": 2 * tp / denominator, "iou": tp / union}


def multiclass_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    label_ids: list[int],
) -> dict[str, Any]:
    results = {}
    for label_id in label_ids:
        results[str(label_id)] = binary_segmentation_metrics(
            prediction == label_id, target == label_id
        )
    return results
