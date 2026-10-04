"""Permutation testing and the Holm step-down correction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def label_permutation(
    statistic: Callable[[npt.NDArray[np.int64]], float],
    size: int,
    replicates: int,
    seed: int,
    observed: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    values = np.zeros(replicates, dtype=np.float64)
    base = np.arange(size, dtype=np.int64)
    for index in range(replicates):
        values[index] = statistic(rng.permutation(base))
    exceed = float(np.sum(np.abs(values) >= abs(observed))) if np.isfinite(observed) else float(replicates)
    p_value = (exceed + 1.0) / (replicates + 1.0)
    return (p_value, values)


def paired_permutation(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    observed = float(np.mean(difference))
    rng = np.random.default_rng(seed)
    values = np.zeros(replicates, dtype=np.float64)
    for index in range(replicates):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=difference.size)
        values[index] = float(np.mean(difference * signs))
    exceed = float(np.sum(np.abs(values) >= abs(observed)))
    return ((exceed + 1.0) / (replicates + 1.0), observed)


def holm_step_down(p_values: npt.NDArray[np.float64], alpha: float = 0.05) -> npt.NDArray[np.float64]:
    count = int(p_values.size)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    scaled = np.zeros(count, dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order.tolist()):
        candidate = float(p_values[position]) * float(count - rank)
        running = max(running, candidate)
        scaled[position] = min(1.0, running)
    del alpha
    return scaled


def benjamini_hochberg(p_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    count = int(p_values.size)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(-p_values, kind="stable")
    adjusted = np.zeros(count, dtype=np.float64)
    running = 1.0
    for rank, position in enumerate(order.tolist()):
        candidate = float(p_values[position]) * float(count) / float(count - rank)
        running = min(running, candidate)
        adjusted[position] = min(1.0, running)
    return adjusted


def significant_after(adjusted: npt.NDArray[np.float64], alpha: float = 0.05) -> npt.NDArray[np.bool_]:
    return adjusted < alpha


def min_detectable_difference(
    spread: float,
    size: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    from scipy import stats

    if size < 2 or spread <= 0.0:
        return float("nan")
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * spread / np.sqrt(size))
