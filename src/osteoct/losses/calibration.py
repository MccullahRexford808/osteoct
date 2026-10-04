"""Calibration-aware penalty terms."""

from __future__ import annotations

from torch import Tensor


def brier_score(probabilities: Tensor, targets: Tensor) -> Tensor:
    return ((probabilities - targets) ** 2).mean()


def expected_calibration_error(probabilities: Tensor, targets: Tensor, bins: int = 10) -> Tensor:
    if bins < 2:
        raise ValueError("the calibration error needs at least two bins")
    edges = [index / bins for index in range(bins + 1)]
    total = probabilities.new_zeros(())
    size = int(probabilities.shape[0])
    if size == 0:
        return total
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        confidence = probabilities[mask].mean()
        accuracy = targets[mask].mean()
        total = total + mask.to(probabilities.dtype).mean() * (confidence - accuracy).abs()
    return total
