"""Test-retest stability filter: descriptors are kept only when they reproduce."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class StabilityVerdict:
    retained: tuple[str, ...]
    dropped: tuple[str, ...]
    agreement: tuple[float, ...]

    def retained_index(self, names: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(position for position, name in enumerate(names) if name in self.retained)

    def drop_share(self) -> float:
        total = len(self.retained) + len(self.dropped)
        if total == 0:
            return 0.0
        return len(self.dropped) / total


def icc_2_1(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    if first.shape != second.shape:
        raise ValueError("the two repeats must have the same shape")
    stacked = np.stack([first, second], axis=0)
    subjects = stacked.shape[1]
    if subjects < 2:
        return 0.0
    grand = float(np.mean(stacked))
    subject_means = np.mean(stacked, axis=0)
    rater_means = np.mean(stacked, axis=1)
    ms_subject = (2.0 / (subjects - 1)) * float(np.sum((subject_means - grand) ** 2))
    residual = stacked - subject_means[None, :] - rater_means[:, None] + grand
    ms_error = float(np.sum(residual**2)) / max(1.0, float(subjects - 1))
    denominator = ms_subject + ms_error
    if denominator <= 0.0:
        return 1.0
    return float((ms_subject - ms_error) / denominator)


def reproducibility_coefficient(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    difference = first - second
    return float(1.96 * np.std(difference, ddof=1) / max(1e-12, abs(float(np.mean(np.concatenate([first, second]))))))


def stability_filter(
    names: tuple[str, ...],
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
    threshold: float,
) -> StabilityVerdict:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("the repeats must be matrices of subject by descriptor")
    if first.shape != second.shape:
        raise ValueError("the two repeats must have the same shape")
    if first.shape[1] != len(names):
        raise ValueError("one name per descriptor column is required")
    agreement: list[float] = []
    retained: list[str] = []
    dropped: list[str] = []
    for column, name in enumerate(names):
        value = icc_2_1(first[:, column], second[:, column])
        agreement.append(value)
        if value >= threshold:
            retained.append(name)
        else:
            dropped.append(name)
    return StabilityVerdict(tuple(retained), tuple(dropped), tuple(agreement))


def apply_filter(
    vector: npt.NDArray[np.float64], names: tuple[str, ...], verdict: StabilityVerdict
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    selected = tuple(position for position, name in enumerate(names) if name in verdict.retained)
    values = tuple(float(vector[position]) for position in selected)
    return values, tuple(names[position] for position in selected)
