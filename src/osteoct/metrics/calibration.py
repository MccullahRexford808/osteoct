"""Calibration of the predicted risk against the observed outcome."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.utils.numeric import logit

MAX_NEWTON_STEPS = 60
NEWTON_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    slope: float
    intercept: float
    iterations: int
    converged: bool

    def as_mapping(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}


def calibration_slope_intercept(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> CalibrationFit:
    scores = logit(np.asarray(probabilities, dtype=np.float64)).reshape(-1, 1)
    target = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return CalibrationFit(float("nan"), float("nan"), 0, False)
    design = np.concatenate([np.ones((scores.shape[0], 1), dtype=np.float64), scores], axis=1)
    beta = np.zeros(2, dtype=np.float64)
    converged = False
    iteration = 0
    for _iteration in range(1, MAX_NEWTON_STEPS + 1):
        linear = design @ beta
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        gradient = design.T @ (target - fitted)
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if float(np.max(np.abs(step))) < NEWTON_TOLERANCE:
            converged = True
            break
    return CalibrationFit(slope=float(beta[1]), intercept=float(beta[0]), iterations=iteration, converged=converged)


def brier(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    return float(np.mean((np.asarray(probabilities, dtype=np.float64) - np.asarray(outcomes, dtype=np.float64)) ** 2))


def calibration_in_the_large(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    return float(np.mean(np.asarray(outcomes, dtype=np.float64)) - np.mean(np.asarray(probabilities, dtype=np.float64)))


def reliability_bins(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    bins: int = 10,
) -> tuple[tuple[float, float, int], ...]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[tuple[float, float, int]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((float(np.mean(probabilities[mask])), float(np.mean(outcomes[mask])), count))
    return tuple(rows)


def observed_expected_ratio(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    expected = float(np.sum(probabilities))
    if expected <= 0.0:
        return float("nan")
    return float(np.sum(outcomes) / expected)


def over_dispersion(fit: CalibrationFit) -> bool:
    return bool(np.isfinite(fit.slope) and fit.slope > 1.0 + 1e-6)
