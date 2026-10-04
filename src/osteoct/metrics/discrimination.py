"""Discrimination metrics for the classification and clinical endpoints."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from osteoct.utils.numeric import ranked


def auc(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float:
    positives = labels == 1
    negatives = labels == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = ranked(np.asarray(scores, dtype=np.float64)) + 1.0
    positive_rank_sum = float(ranks[positives].sum())
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return statistic / (n_positive * n_negative)


def roc_curve(
    scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    sorted_labels = np.asarray(labels, dtype=np.int64)[order]
    positives = int(sorted_labels.sum())
    negatives = int(sorted_labels.size - positives)
    if positives == 0 or negatives == 0:
        return (np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64))
    true_positive = np.cumsum(sorted_labels, dtype=np.float64)
    false_positive = np.cumsum(1 - sorted_labels, dtype=np.float64)
    sensitivity = np.concatenate([[0.0], true_positive / positives])
    fall_out = np.concatenate([[0.0], false_positive / negatives])
    return (fall_out, sensitivity)


def threshold_at_specificity(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    target_specificity: float,
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    candidates = sorted({float(value) for value in values.tolist()}, reverse=True)
    if not candidates:
        return float("nan")
    for threshold in candidates:
        if specificity_at_threshold(values, labels, threshold) >= target_specificity - 1e-12:
            return float(threshold)
    return float(candidates[-1])


def sensitivity_at_specificity(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    target_specificity: float = 0.90,
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    if not bool(positives.any()):
        return float("nan")
    best = float("nan")
    for threshold in sorted({float(value) for value in values.tolist()}, reverse=True):
        if specificity_at_threshold(values, labels, threshold) < target_specificity - 1e-12:
            continue
        sensitivity = float((values >= threshold)[positives].sum() / max(1, int(positives.sum())))
        best = sensitivity if not np.isfinite(best) else max(best, sensitivity)
    return best


def specificity_at_threshold(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], threshold: float) -> float:
    predictions = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    negatives = labels == 0
    if not bool(negatives.any()):
        return float("nan")
    return float(1.0 - predictions[negatives].sum() / max(1, int(negatives.sum())))


def accuracy(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], threshold: float = 0.5) -> float:
    predictions = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    return float(np.mean(predictions == labels))


def balanced_accuracy(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], threshold: float = 0.5) -> float:
    sensitivity = _sensitivity(scores, labels, threshold)
    specificity = specificity_at_threshold(scores, labels, threshold)
    return float(0.5 * (sensitivity + specificity))


def _sensitivity(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], threshold: float) -> float:
    predictions = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    positives = labels == 1
    if not bool(positives.any()):
        return float("nan")
    return float(predictions[positives].sum() / max(1, int(positives.sum())))


def band_is_osteoporotic(index: npt.NDArray[np.int64], osteoporosis_min: int = 2) -> npt.NDArray[np.int64]:
    return (np.asarray(index, dtype=np.int64) >= osteoporosis_min).astype(np.int64)


def positive_rate(labels: npt.NDArray[np.int64]) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels == 1))


def rank_scores(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return ranked(np.asarray(scores, dtype=np.float64))
