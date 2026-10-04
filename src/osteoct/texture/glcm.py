"""Grey-level co-occurrence descriptors over the trabecular region of interest."""

from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt

GLCM_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (first, second, third)
    for first, second, third in product((-1, 0, 1), repeat=3)
    if (first, second, third) != (0, 0, 0) and (first, second, third) > (-first, -second, -third)
)

GLCM_FEATURES: tuple[str, ...] = (
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "glcm_homogeneity",
    "glcm_dissimilarity",
)


def quantise(
    block: npt.NDArray[np.float64],
    low: float,
    high: float,
    levels: int,
) -> npt.NDArray[np.int32]:
    if levels < 2:
        raise ValueError("a co-occurrence matrix needs at least two grey levels")
    if high <= low:
        raise ValueError("the grey-level window must have positive width")
    scaled = np.clip((block - low) / (high - low), 0.0, 1.0 - 1e-9)
    return np.floor(scaled * levels).astype(np.int32)


def _pair_mask(shape: tuple[int, int, int], offset: tuple[int, int, int]) -> npt.NDArray[np.bool_]:
    mask = np.ones(shape, dtype=bool)
    for axis, step in enumerate(offset):
        if step == 0:
            continue
        selector: list[slice] = [slice(None)] * 3
        selector[axis] = slice(None, -step) if step > 0 else slice(-step, None)
        keep = np.zeros(shape, dtype=bool)
        keep[tuple(selector)] = True
        mask &= keep
    return mask


def cooccurrence(
    quantised: npt.NDArray[np.int32],
    offset: tuple[int, int, int],
    levels: int,
    symmetric: bool = True,
) -> npt.NDArray[np.float64]:
    shifted = quantised
    for axis, step in enumerate(offset):
        if step != 0:
            shifted = np.roll(shifted, -step, axis=axis)
    valid = _pair_mask(quantised.shape, offset)
    source = quantised[valid]
    target = shifted[valid]
    counts = np.bincount(source * levels + target, minlength=levels * levels).astype(np.float64)
    matrix = counts.reshape(levels, levels)
    if symmetric:
        matrix = matrix + matrix.T
    total = float(matrix.sum())
    if total <= 0.0:
        return matrix
    return matrix / total


def harness_features(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    levels = matrix.shape[0]
    indices = np.arange(levels, dtype=np.float64)
    row = indices[:, None]
    column = indices[None, :]
    difference = row - column
    contrast = float(np.sum(matrix * difference**2))
    dissimilarity = float(np.sum(matrix * np.abs(difference)))
    homogeneity = float(np.sum(matrix / (1.0 + difference**2)))
    energy = float(np.sqrt(np.sum(matrix**2)))
    mean_row = float(np.sum(matrix * row))
    mean_column = float(np.sum(matrix * column))
    sd_row = float(np.sqrt(max(0.0, np.sum(matrix * (row - mean_row) ** 2))))
    sd_column = float(np.sqrt(max(0.0, np.sum(matrix * (column - mean_column) ** 2))))
    if sd_row * sd_column <= 0.0:
        correlation = 0.0
    else:
        numerator = float(np.sum(matrix * (row - mean_row) * (column - mean_column)))
        correlation = numerator / (sd_row * sd_column)
    return np.asarray([contrast, correlation, energy, homogeneity, dissimilarity], dtype=np.float64)


def glcm_descriptors(
    quantised: npt.NDArray[np.int32],
    levels: int,
    offsets: tuple[tuple[int, int, int], ...] = GLCM_OFFSETS,
) -> npt.NDArray[np.float64]:
    if quantised.ndim != 3:
        raise ValueError("co-occurrence descriptors need a three-dimensional region")
    accumulated = np.zeros(len(GLCM_FEATURES), dtype=np.float64)
    used = 0
    for offset in offsets:
        matrix = cooccurrence(quantised, offset, levels)
        if float(matrix.sum()) <= 0.0:
            continue
        accumulated += harness_features(matrix)
        used += 1
    if used == 0:
        raise ValueError("no co-occurrence offset produced a non-empty matrix")
    return accumulated / float(used)
