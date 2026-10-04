"""Run-length descriptors along the three axes of the trabecular region of interest."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

RUNLENGTH_FEATURES: tuple[str, ...] = (
    "rlm_sre",
    "rlm_lre",
    "rlm_gln",
    "rlm_rln",
    "rlm_rp",
)


def run_length_matrix(quantised: npt.NDArray[np.int32], axis: int, levels: int) -> npt.NDArray[np.float64]:
    moved = np.moveaxis(quantised, axis, 0)
    length = moved.shape[0]
    matrix = np.zeros((levels, length), dtype=np.float64)
    flat = moved.reshape(length, -1)
    for column in range(flat.shape[1]):
        column_values = flat[:, column]
        start = 0
        for index in range(1, length + 1):
            if index == length or column_values[index] != column_values[start]:
                grey = int(column_values[start])
                run = index - start
                matrix[grey, run - 1] += 1.0
                start = index
    return matrix


def runlength_features(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    total = float(matrix.sum())
    if total <= 0.0:
        return np.zeros(len(RUNLENGTH_FEATURES), dtype=np.float64)
    normalised = matrix / total
    levels, length = matrix.shape
    run = np.arange(1, length + 1, dtype=np.float64)[None, :]
    sre = float(np.sum(normalised / (run**2)))
    lre = float(np.sum(normalised * run**2))
    gln = float(np.sum((normalised.sum(axis=1)) ** 2))
    rln = float(np.sum((normalised.sum(axis=0)) ** 2))
    run_counts = normalised.sum(axis=0)
    rp = float(np.sum(run_counts**2) / max(1.0, float(np.count_nonzero(run_counts))))
    del levels
    return np.asarray([sre, lre, gln, rln, rp], dtype=np.float64)


def runlength_descriptors(quantised: npt.NDArray[np.int32], levels: int) -> npt.NDArray[np.float64]:
    if quantised.ndim != 3:
        raise ValueError("run-length descriptors need a three-dimensional region")
    accumulated = np.zeros(len(RUNLENGTH_FEATURES), dtype=np.float64)
    for axis in range(3):
        matrix = run_length_matrix(quantised, axis, levels)
        accumulated += runlength_features(matrix)
    return accumulated / 3.0
