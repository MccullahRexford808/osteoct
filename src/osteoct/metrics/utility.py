"""Decision-curve net benefit and its components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class DecisionCurvePoint:
    threshold: float
    net_benefit: float
    treat_all: float
    treat_none: float
    flagged_share: float


def net_benefit(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    threshold: float,
) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("the decision threshold must lie strictly inside (0, 1)")
    predictions = np.asarray(probabilities, dtype=np.float64) >= threshold
    target = np.asarray(outcomes, dtype=np.float64)
    size = max(1, target.size)
    true_positive = float(np.sum(predictions & (target > 0.5)))
    false_positive = float(np.sum(predictions & (target <= 0.5)))
    odds = threshold / (1.0 - threshold)
    return float(true_positive / size - false_positive / size * odds)


def treat_all_net_benefit(outcomes: npt.NDArray[np.float64], threshold: float) -> float:
    target = np.asarray(outcomes, dtype=np.float64)
    prevalence = float(np.mean(target)) if target.size else 0.0
    odds = threshold / (1.0 - threshold)
    return float(prevalence - (1.0 - prevalence) * odds)


def decision_curve(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
) -> tuple[DecisionCurvePoint, ...]:
    points: list[DecisionCurvePoint] = []
    for threshold in thresholds:
        points.append(
            DecisionCurvePoint(
                threshold=threshold,
                net_benefit=net_benefit(probabilities, outcomes, threshold),
                treat_all=treat_all_net_benefit(outcomes, threshold),
                treat_none=0.0,
                flagged_share=float(np.mean(np.asarray(probabilities, dtype=np.float64) >= threshold)),
            )
        )
    return tuple(points)


def net_benefit_gain(
    probabilities: npt.NDArray[np.float64],
    comparator_probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    threshold: float,
) -> float:
    return net_benefit(probabilities, outcomes, threshold) - net_benefit(comparator_probabilities, outcomes, threshold)


def standardised_net_benefit(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    threshold: float,
) -> float:
    prevalence = float(np.mean(np.asarray(outcomes, dtype=np.float64)))
    if prevalence <= 0.0 or prevalence >= 1.0:
        return float("nan")
    value = net_benefit(probabilities, outcomes, threshold)
    return float(value / prevalence)


def best_threshold(
    probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
) -> float:
    values = [net_benefit(probabilities, outcomes, threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(np.asarray(values, dtype=np.float64)))])
