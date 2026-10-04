"""Head losses for the band, tier and severity outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class HeadLossWeights:
    band: float = 1.0
    tier: float = 0.6
    severity: float = 0.6
    risk: float = 0.8
    ordinal: float = 0.4


def categorical_loss(logits: Tensor, targets: Tensor, class_weights: Tensor | None = None) -> Tensor:
    loss: Tensor = functional.cross_entropy(logits, targets, weight=class_weights)
    return loss


def focal_loss(logits: Tensor, targets: Tensor, gamma: float = 1.5) -> Tensor:
    log_probabilities = functional.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probabilities)
    gathered = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    modulator = (1.0 - probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)).pow(gamma)
    loss: Tensor = -(modulator * gathered).mean()
    return loss


def ordinal_distance_loss(logits: Tensor, targets: Tensor) -> Tensor:
    classes = int(logits.shape[-1])
    if classes < 2:
        raise ValueError("the ordinal distance loss needs at least two classes")
    probabilities = functional.softmax(logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)[:, :-1]
    thresholds = torch.arange(classes - 1, device=logits.device, dtype=logits.dtype)[None, :]
    indicator = (targets[:, None] > thresholds).to(logits.dtype)
    loss: Tensor = ((cumulative - indicator) ** 2).mean()
    return loss


def expected_rank(logits: Tensor) -> Tensor:
    probabilities = functional.softmax(logits, dim=-1)
    ranks = torch.arange(int(logits.shape[-1]), device=logits.device, dtype=logits.dtype)[None, :]
    expectation: Tensor = (probabilities * ranks).sum(dim=-1)
    return expectation


def binary_loss(logits: Tensor, targets: Tensor, positive_weight: float = 1.0) -> Tensor:
    weights = torch.where(targets > 0.5, torch.full_like(targets, positive_weight), torch.ones_like(targets))
    loss: Tensor = functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)
    return loss


def band_and_tier_loss(
    band_logits: Tensor,
    tier_logits: Tensor,
    severity_logits: Tensor,
    risk_logit: Tensor,
    band_target: Tensor,
    tier_target: Tensor,
    severity_target: Tensor,
    risk_target: Tensor,
    weights: HeadLossWeights,
) -> Tensor:
    return (
        weights.band * categorical_loss(band_logits, band_target)
        + weights.tier * categorical_loss(tier_logits, tier_target)
        + weights.severity * categorical_loss(severity_logits, severity_target)
        + weights.risk * binary_loss(risk_logit, risk_target)
        + weights.ordinal * ordinal_distance_loss(tier_logits, tier_target)
        + weights.ordinal * ordinal_distance_loss(severity_logits, severity_target)
    )
