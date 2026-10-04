"""Local binary pattern descriptors on the trabecular region of interest."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NEIGHBOUR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (x_step, y_step, z_step)
    for x_step in (-1, 0, 1)
    for y_step in (-1, 0, 1)
    for z_step in (-1, 0, 1)
    if (x_step, y_step, z_step) != (0, 0, 0)
)

LBP_FEATURES: tuple[str, ...] = (
    "lbp_mean_popcount",
    "lbp_sd_popcount",
    "lbp_transition_mean",
    "lbp_uniform_share",
    "lbp_histogram_entropy",
)


def _shift(block: npt.NDArray[np.float64], offset: tuple[int, int, int]) -> npt.NDArray[np.float64]:
    shifted = block
    for axis, step in enumerate(offset):
        if step != 0:
            shifted = np.roll(shifted, -step, axis=axis)
    return shifted


def _interior(shape: tuple[int, int, int]) -> npt.NDArray[np.bool_]:
    mask = np.ones(shape, dtype=bool)
    if min(shape) <= 2:
        return mask
    mask[0, :, :] = False
    mask[-1, :, :] = False
    mask[:, 0, :] = False
    mask[:, -1, :] = False
    mask[:, :, 0] = False
    mask[:, :, -1] = False
    return mask


def pattern_codes(block: npt.NDArray[np.float64]) -> npt.NDArray[np.uint32]:
    offsets = NEIGHBOUR_OFFSETS
    code = np.zeros(block.shape, dtype=np.uint32)
    for position, offset in enumerate(offsets):
        neighbour = _shift(block, offset)
        code |= (neighbour >= block).astype(np.uint32) << np.uint32(position)
    return code


def lbp_descriptors(block: npt.NDArray[np.float64], uniform_threshold: int = 2) -> npt.NDArray[np.float64]:
    if block.ndim != 3:
        raise ValueError("local binary patterns need a three-dimensional region")
    interior = _interior(block.shape)
    code = pattern_codes(block)[interior]
    if code.size == 0:
        return np.zeros(len(LBP_FEATURES), dtype=np.float64)
    popcount = np.asarray([int(bin(int(value)).count("1")) for value in code], dtype=np.float64)
    transition_counts = np.asarray(
        [_ring_transitions(int(value), len(NEIGHBOUR_OFFSETS)) for value in code],
        dtype=np.float64,
    )
    uniform_share = float(np.mean(transition_counts <= float(uniform_threshold)))
    histogram = np.bincount(popcount.astype(np.int64).tolist(), minlength=len(NEIGHBOUR_OFFSETS) + 1).astype(np.float64)
    shares = histogram / max(1.0, float(histogram.sum()))
    nonzero = shares[shares > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    return np.asarray(
        [
            float(np.mean(popcount)),
            float(np.std(popcount)),
            float(np.mean(transition_counts)),
            uniform_share,
            entropy,
        ],
        dtype=np.float64,
    )


def _ring_transitions(value: int, neighbours: int) -> int:
    bits = [(value >> position) & 1 for position in range(neighbours)]
    bits.append(bits[0])
    return sum(1 for index in range(neighbours) if bits[index] != bits[index + 1])
