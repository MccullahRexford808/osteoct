"""Numerical helpers shared by the encoders, losses and statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch.nn.functional as functional
from torch import Tensor


def window_partition_3d(values: Tensor, window: int) -> tuple[Tensor, tuple[int, int, int, int, int]]:
    if window < 1:
        raise ValueError("the window size must be positive")
    batch, channels, depth, height, width = values.shape
    pad_width = (window - width % window) % window
    pad_height = (window - height % window) % window
    pad_depth = (window - depth % window) % window
    padded = functional.pad(values, (0, pad_width, 0, pad_height, 0, pad_depth))
    padded_depth, padded_height, padded_width = padded.shape[2:]
    reshaped = padded.view(
        batch,
        channels,
        padded_depth // window,
        window,
        padded_height // window,
        window,
        padded_width // window,
        window,
    )
    permuted = reshaped.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = permuted.view(-1, window * window * window, channels)
    return (windows, (batch, channels, depth, height, width))


def window_reverse_3d(windows: Tensor, shape: tuple[int, int, int, int, int], window: int) -> Tensor:
    batch, channels, depth, height, width = shape
    pad_width = (window - width % window) % window
    pad_height = (window - height % window) % window
    pad_depth = (window - depth % window) % window
    padded_depth = depth + pad_depth
    padded_height = height + pad_height
    padded_width = width + pad_width
    reshaped = windows.view(
        batch,
        padded_depth // window,
        padded_height // window,
        padded_width // window,
        window,
        window,
        window,
        channels,
    )
    permuted = reshaped.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    merged = permuted.view(batch, channels, padded_depth, padded_height, padded_width)
    return merged[:, :, :depth, :height, :width]


def safe_divide(
    numerator: npt.NDArray[np.float64], denominator: npt.NDArray[np.float64], fill: float = 0.0
) -> npt.NDArray[np.float64]:
    output: npt.NDArray[np.float64] = np.full_like(numerator, fill, dtype=np.float64)
    divided: npt.NDArray[np.float64] = np.asarray(
        np.divide(numerator, denominator, out=output, where=denominator != 0.0), dtype=np.float64
    )
    return divided


def safe_scalar(numerator: float, denominator: float, fill: float = 0.0) -> float:
    if denominator == 0.0:
        return fill
    return numerator / denominator


def entropy(values: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    clipped = np.clip(values, 1e-12, None)
    negative: npt.NDArray[np.float64] = np.asarray(-np.sum(clipped * np.log(clipped), axis=axis), dtype=np.float64)
    return negative


def normalise_rows(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    totals = values.sum(axis=1, keepdims=True)
    return safe_divide(values, totals, fill=1.0 / max(1, values.shape[1]))


def moving_average(values: Sequence[float], window: int) -> tuple[float, ...]:
    if window < 1:
        raise ValueError("the averaging window must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return ()
    cumulative = np.cumsum(np.concatenate([[0.0], array]))
    output: list[float] = []
    for index in range(array.size):
        start = max(0, index - window + 1)
        span = index - start + 1
        output.append(float((cumulative[index + 1] - cumulative[start]) / span))
    return tuple(output)


def standardise(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mean = float(np.mean(values))
    spread = float(np.std(values))
    if spread <= 0.0:
        return np.zeros_like(values)
    return (values - mean) / spread


def ranked(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    index = 0
    while index < sorted_values.size:
        stop = index + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[index]:
            stop += 1
        ranks[order[index:stop]] = 0.5 * (index + stop - 1)
        index = stop
    return ranks


def softmax_numpy(logits: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return safe_divide(exponentiated, exponentiated.sum(axis=axis, keepdims=True), fill=0.0)


def logit(probability: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def one_hot(indices: npt.NDArray[np.int64], classes: int) -> npt.NDArray[np.float64]:
    output = np.zeros((indices.size, classes), dtype=np.float64)
    for row, column in enumerate(indices.tolist()):
        output[row, column] = 1.0
    return output


def percentile_interval(values: npt.NDArray[np.float64], low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, low)), float(np.percentile(values, high)))


def clip_probabilities(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.clip(values, 1e-6, 1.0 - 1e-6)
