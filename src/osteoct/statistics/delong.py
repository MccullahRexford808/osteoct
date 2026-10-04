"""DeLong variance and covariance for the area under the receiver operating curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class DelongResult:
    auc: float
    variance: float
    standard_error: float

    def interval(self, alpha: float = 0.05) -> tuple[float, float]:
        from scipy import stats

        z_value = float(stats.norm.ppf(1.0 - alpha / 2.0))
        half = z_value * self.standard_error
        return (float(self.auc - half), float(self.auc + half))


def midrank(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < sorted_values.size:
        stop = index + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[index]:
            stop += 1
        average = 0.5 * (index + stop - 1) + 1.0
        ranks[order[index:stop]] = average
        index = stop
    return ranks


def placement_values(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    positives = labels == 1
    negatives = labels == 0
    count_positive = int(positives.sum())
    count_negative = int(negatives.sum())
    if count_positive == 0 or count_negative == 0:
        raise ValueError("the DeLong placement values need one positive and one negative at least")
    overall = midrank(np.asarray(scores, dtype=np.float64))
    positive_ranks = overall[positives]
    negative_ranks = overall[negatives]
    within_positive = midrank(np.asarray(scores, dtype=np.float64)[positives])
    within_negative = midrank(np.asarray(scores, dtype=np.float64)[negatives])
    v01 = (positive_ranks - within_positive) / float(count_negative)
    v10 = 1.0 - (negative_ranks - within_negative) / float(count_positive)
    return (v01, v10)


def auc_variance(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
) -> DelongResult:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    v01, v10 = placement_values(scores, labels)
    value = float(np.mean(v01))
    variance = float(np.var(v01, ddof=1)) / float(positives) + float(np.var(v10, ddof=1)) / float(negatives)
    return DelongResult(auc=value, variance=variance, standard_error=float(np.sqrt(max(0.0, variance))))


def paired_delong(
    first_scores: npt.NDArray[np.float64],
    second_scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
) -> tuple[float, float, float]:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    first_v01, first_v10 = placement_values(first_scores, labels)
    second_v01, second_v10 = placement_values(second_scores, labels)
    first_auc = float(np.mean(first_v01))
    second_auc = float(np.mean(second_v01))
    variance_first = float(np.var(first_v01, ddof=1)) / positives + float(np.var(first_v10, ddof=1)) / negatives
    variance_second = float(np.var(second_v01, ddof=1)) / positives + float(np.var(second_v10, ddof=1)) / negatives
    covariance = float(np.cov(first_v01, second_v01, ddof=1)[0, 1]) / positives
    covariance += float(np.cov(first_v10, second_v10, ddof=1)[0, 1]) / negatives
    difference_variance = variance_first + variance_second - 2.0 * covariance
    return (first_auc, second_auc, float(np.sqrt(max(0.0, difference_variance))))


def paired_z(first_scores: npt.NDArray[np.float64], second_scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float:
    first_auc, second_auc, spread = paired_delong(first_scores, second_scores, labels)
    if spread <= 0.0:
        return 0.0
    return float((first_auc - second_auc) / spread)


def paired_p_value(
    first_scores: npt.NDArray[np.float64],
    second_scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
) -> float:
    from scipy import stats

    z_value = paired_z(first_scores, second_scores, labels)
    return float(2.0 * (1.0 - float(stats.norm.cdf(abs(z_value)))))
