"""Patient-level bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

DEFAULT_REPLICATES = 2000


def percentile_interval(values: npt.NDArray[np.float64], alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return (float(np.percentile(values, lower)), float(np.percentile(values, upper)))


def bootstrap_indices(size: int, replicates: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    return rng.integers(0, size, size=(replicates, size)).astype(np.int64)


def patient_clusters(patient_ids: Sequence[str]) -> tuple[npt.NDArray[np.int64], ...]:
    table: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        table.setdefault(patient, []).append(index)
    return tuple(np.asarray(rows, dtype=np.int64) for rows in table.values())


def cluster_bootstrap(
    patient_ids: Sequence[str],
    statistic: Callable[[npt.NDArray[np.int64]], float],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    clusters = patient_clusters(patient_ids)
    total = int(sum(cluster.size for cluster in clusters))
    rng = np.random.default_rng(seed)
    values = np.zeros(replicates, dtype=np.float64)
    for index in range(replicates):
        drawn = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(position)] for position in drawn.tolist()])
        if rows.size == 0 or rows.size > 4 * total:
            values[index] = float("nan")
            continue
        values[index] = statistic(rows)
    return values


def bootstrap_interval(
    patient_ids: Sequence[str],
    statistic: Callable[[npt.NDArray[np.int64]], float],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = cluster_bootstrap(patient_ids, statistic, replicates, seed)
    finite = values[np.isfinite(values)]
    return percentile_interval(finite, alpha)


def bootstrap_mean_interval(
    values: npt.NDArray[np.float64],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    size = int(values.size)
    if size == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, size, size=(replicates, size))
    means = np.asarray(values, dtype=np.float64)[draws].mean(axis=1)
    return percentile_interval(means, alpha)


def standard_error(values: npt.NDArray[np.float64]) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))
