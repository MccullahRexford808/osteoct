"""Composite objective for the scorer and for the hardened-regulariser arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from torch import Tensor

from osteoct.losses.calibration import brier_score, expected_calibration_error
from osteoct.losses.heads import HeadLossWeights, band_and_tier_loss
from osteoct.losses.regulariser import RegulariserWeights, axiom_penalty


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    heads: HeadLossWeights = field(default_factory=HeadLossWeights)
    calibration: float = 0.2
    regulariser: RegulariserWeights = field(default_factory=RegulariserWeights)
    regulariser_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    supervised: float
    calibration: float
    regulariser: float
    total: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "supervised": self.supervised,
            "calibration": self.calibration,
            "regulariser": self.regulariser,
            "total": self.total,
        }


def compose(
    band_logits: Tensor,
    tier_logits: Tensor,
    severity_logits: Tensor,
    risk_logit: Tensor,
    band_target: Tensor,
    tier_target: Tensor,
    severity_target: Tensor,
    risk_target: Tensor,
    weights: ObjectiveWeights,
    derived_band: npt.NDArray[np.int64] | None = None,
    derived_tier: npt.NDArray[np.int64] | None = None,
    graded_severity: npt.NDArray[np.int64] | None = None,
    evidence_complete: npt.NDArray[np.float64] | None = None,
) -> tuple[Tensor, ObjectiveBreakdown]:
    supervised = band_and_tier_loss(
        band_logits,
        tier_logits,
        severity_logits,
        risk_logit,
        band_target,
        tier_target,
        severity_target,
        risk_target,
        weights.heads,
    )
    probabilities = risk_logit.sigmoid()
    calibration = brier_score(probabilities, risk_target) + expected_calibration_error(probabilities, risk_target)
    regulariser_term = risk_logit.new_zeros(())
    if weights.regulariser_scale > 0.0 and derived_band is not None and derived_tier is not None and graded_severity is not None:
        completeness = evidence_complete if evidence_complete is not None else np.ones(int(band_logits.shape[0]), dtype=np.float64)
        regulariser_term = axiom_penalty(
            band_logits,
            tier_logits,
            severity_logits,
            derived_band,
            derived_tier,
            graded_severity,
            completeness,
            weights.regulariser,
        )
    total = supervised + weights.calibration * calibration + weights.regulariser_scale * regulariser_term
    breakdown = ObjectiveBreakdown(
        supervised=float(supervised.detach().item()),
        calibration=float(calibration.detach().item()),
        regulariser=float(regulariser_term.detach().item()),
        total=float(total.detach().item()),
    )
    return (total, breakdown)
