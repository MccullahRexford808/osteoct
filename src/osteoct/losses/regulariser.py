"""The additive axiom penalty used by the hardened-regulariser arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RegulariserWeights:
    band: float = 0.5
    tier: float = 0.32
    severity: float = 0.3
    evidence: float = 0.22


def band_penalty(band_logits: Tensor, derived_band: npt.NDArray[np.int64]) -> Tensor:
    target = torch.as_tensor(derived_band, dtype=torch.long, device=band_logits.device)
    log_probabilities = functional.log_softmax(band_logits, dim=-1)
    gathered = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    penalty: Tensor = -gathered.mean()
    return penalty


def tier_penalty(tier_logits: Tensor, derived_tier: npt.NDArray[np.int64]) -> Tensor:
    target = torch.as_tensor(derived_tier, dtype=torch.long, device=tier_logits.device)
    log_probabilities = functional.log_softmax(tier_logits, dim=-1)
    gathered = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    penalty: Tensor = -gathered.mean()
    return penalty


def severity_penalty(severity_logits: Tensor, graded_severity: npt.NDArray[np.int64]) -> Tensor:
    probabilities = functional.softmax(severity_logits, dim=-1)
    ranks = torch.arange(int(severity_logits.shape[-1]), device=severity_logits.device, dtype=severity_logits.dtype)[None, :]
    expectation = (probabilities * ranks).sum(dim=-1)
    target = torch.as_tensor(graded_severity, dtype=severity_logits.dtype, device=severity_logits.device)
    penalty: Tensor = (expectation - target).abs().mean()
    return penalty


def evidence_penalty(evidence_complete: npt.NDArray[np.float64]) -> Tensor:
    completeness = torch.as_tensor(evidence_complete, dtype=torch.float32)
    penalty: Tensor = (1.0 - completeness).mean()
    return penalty


def axiom_penalty(
    band_logits: Tensor,
    tier_logits: Tensor,
    severity_logits: Tensor,
    derived_band: npt.NDArray[np.int64],
    derived_tier: npt.NDArray[np.int64],
    graded_severity: npt.NDArray[np.int64],
    evidence_complete: npt.NDArray[np.float64],
    weights: RegulariserWeights,
) -> Tensor:
    return (
        weights.band * band_penalty(band_logits, derived_band)
        + weights.tier * tier_penalty(tier_logits, derived_tier)
        + weights.severity * severity_penalty(severity_logits, graded_severity)
        + weights.evidence * evidence_penalty(evidence_complete)
    )
