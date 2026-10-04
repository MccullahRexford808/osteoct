"""Wavelet energy descriptors from a separable Haar decomposition of the region."""

from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt

SUB_BANDS: tuple[str, ...] = tuple("".join(band) for band in product(("l", "h"), repeat=3))
WAVELET_FEATURES: tuple[str, ...] = tuple(f"wavelet_{band}_energy" for band in SUB_BANDS) + (
    "wavelet_total_energy",
    "wavelet_entropy",
)


def haar_step(values: npt.NDArray[np.float64], axis: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    moved = np.moveaxis(values, axis, 0)
    length = moved.shape[0]
    if length % 2 == 1:
        moved = np.concatenate([moved, moved[-1:]], axis=0)
        length += 1
    pairs = moved.reshape(length // 2, 2, *moved.shape[1:])
    scale = 1.0 / np.sqrt(2.0)
    low = (pairs[:, 0] + pairs[:, 1]) * scale
    high = (pairs[:, 0] - pairs[:, 1]) * scale
    return (np.moveaxis(low, 0, axis), np.moveaxis(high, 0, axis))


def haar_decompose(
    block: npt.NDArray[np.float64],
    levels: int,
) -> dict[str, npt.NDArray[np.float64]]:
    if levels < 1:
        raise ValueError("a wavelet decomposition needs at least one level")
    bands: dict[str, npt.NDArray[np.float64]] = {}
    approximation = block
    for level in range(1, levels + 1):
        if min(approximation.shape) < 2:
            break
        low_0, high_0 = haar_step(approximation, 0)
        low_1, high_1 = haar_step(low_0, 1)
        high_low_1, high_high_1 = haar_step(high_0, 1)
        sub_bands = {
            "ll": low_1,
            "hl": high_low_1,
            "lh": high_1,
            "hh": high_high_1,
        }
        for name, sub in sub_bands.items():
            bands[f"{name}_{level}"] = sub
        approximation = low_1
    return bands


def wavelet_descriptors(block: npt.NDArray[np.float64], levels: int = 2) -> npt.NDArray[np.float64]:
    decomposition = haar_decompose(block, levels)
    grouped: dict[str, list[float]] = {band: [] for band in SUB_BANDS}
    for name, coefficients in decomposition.items():
        head = name.split("_")[0]
        band = {"ll": "lll", "hl": "lhl", "lh": "llh", "hh": "lhh"}[head]
        grouped[band].append(float(np.mean(coefficients**2)))
    energies = np.asarray([float(np.mean(grouped[band])) if grouped[band] else 0.0 for band in SUB_BANDS], dtype=np.float64)
    total = float(np.sum(energies))
    if total <= 0.0:
        entropy = 0.0
    else:
        shares = energies / total
        nonzero = shares[shares > 0.0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
    return np.concatenate([energies, np.asarray([total, entropy], dtype=np.float64)])
