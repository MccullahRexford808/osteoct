"""The learned scorer that produces the candidate decision c*."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from osteoct.schema.vocabulary import BAND_ORDER, TIER_ORDER, DeformityGrade, DiagnosticBand, RiskTier

BAND_CLASSES = len(BAND_ORDER)
TIER_CLASSES = len(TIER_ORDER)
SEVERITY_CLASSES = len(DeformityGrade)


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    descriptor_dim: int
    texture_dim: int
    concept_dim: int
    embedding_dim: int
    hidden: int = 64
    dropout: float = 0.1
    use_descriptors: bool = True
    use_texture: bool = True
    use_concepts: bool = True
    use_embedding: bool = True
    normalise: bool = True

    def input_dim(self) -> int:
        total = 2
        if self.use_descriptors:
            total += self.descriptor_dim
        if self.use_texture:
            total += self.texture_dim
        if self.use_concepts:
            total += self.concept_dim
        if self.use_embedding:
            total += self.embedding_dim
        return total


@dataclass(frozen=True, slots=True)
class ScorerFeatures:
    descriptors: Tensor
    texture: Tensor
    concepts: Tensor
    context: Tensor
    embedding: Tensor

    def width(self) -> int:
        return int(self.descriptors.shape[-1] + self.texture.shape[-1] + self.concepts.shape[-1] + self.embedding.shape[-1])


@dataclass(frozen=True, slots=True)
class ScorerOutput:
    band_logits: Tensor
    tier_logits: Tensor
    severity_logits: Tensor
    risk_logit: Tensor

    def band(self, row: int = 0) -> DiagnosticBand:
        return BAND_ORDER[int(torch.argmax(self.band_logits[row]).item())]

    def tier(self, row: int = 0) -> RiskTier:
        return TIER_ORDER[int(torch.argmax(self.tier_logits[row]).item())]

    def severity(self, row: int = 0) -> DeformityGrade:
        index = int(torch.argmax(self.severity_logits[row]).item())
        return DeformityGrade(index)

    def probability(self, row: int = 0) -> float:
        return float(torch.sigmoid(self.risk_logit[row]).item())


class OsteoScorer(nn.Module):
    descriptor_mean: Tensor
    descriptor_scale: Tensor
    texture_mean: Tensor
    texture_scale: Tensor
    embedding_mean: Tensor
    embedding_scale: Tensor

    def __init__(self, spec: ScorerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer("descriptor_mean", torch.zeros(spec.descriptor_dim))
        self.register_buffer("descriptor_scale", torch.ones(spec.descriptor_dim))
        self.register_buffer("texture_mean", torch.zeros(spec.texture_dim))
        self.register_buffer("texture_scale", torch.ones(spec.texture_dim))
        self.register_buffer("embedding_mean", torch.zeros(spec.embedding_dim))
        self.register_buffer("embedding_scale", torch.ones(spec.embedding_dim))
        self.trunk = nn.Sequential(
            nn.Linear(spec.input_dim(), spec.hidden),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, spec.hidden),
            nn.GELU(),
        )
        self.band_head = nn.Linear(spec.hidden, BAND_CLASSES)
        self.tier_head = nn.Linear(spec.hidden, TIER_CLASSES)
        self.severity_head = nn.Linear(spec.hidden, SEVERITY_CLASSES)
        self.risk_head = nn.Linear(spec.hidden, 1)

    def standardise(self, values: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        if not self.spec.normalise or int(values.shape[-1]) == 0:
            return values
        standardised: Tensor = (values - mean) / scale
        return standardised

    @torch.no_grad()
    def fit_normalisation(self, features: ScorerFeatures) -> None:
        if not self.spec.normalise:
            return
        for values, mean_name, scale_name in (
            (features.descriptors, "descriptor_mean", "descriptor_scale"),
            (features.texture, "texture_mean", "texture_scale"),
            (features.embedding, "embedding_mean", "embedding_scale"),
        ):
            if int(values.shape[-1]) == 0 or int(values.shape[0]) < 2:
                continue
            centred = values.float() - values.float().mean(dim=0, keepdim=True)
            spread = centred.std(dim=0, unbiased=False).clamp_min(1e-3)
            getattr(self, mean_name).copy_(values.float().mean(dim=0))
            getattr(self, scale_name).copy_(spread)

    def assemble(self, features: ScorerFeatures) -> Tensor:
        parts: list[Tensor] = [features.context]
        if self.spec.use_descriptors:
            parts.append(self.standardise(features.descriptors, self.descriptor_mean, self.descriptor_scale))
        if self.spec.use_texture:
            parts.append(self.standardise(features.texture, self.texture_mean, self.texture_scale))
        if self.spec.use_concepts:
            parts.append(features.concepts)
        if self.spec.use_embedding:
            parts.append(self.standardise(features.embedding, self.embedding_mean, self.embedding_scale))
        return torch.cat(parts, dim=-1)

    def forward(self, features: ScorerFeatures) -> ScorerOutput:
        hidden: Tensor = self.trunk(self.assemble(features))
        return ScorerOutput(
            band_logits=self.band_head(hidden),
            tier_logits=self.tier_head(hidden),
            severity_logits=self.severity_head(hidden),
            risk_logit=self.risk_head(hidden).squeeze(-1),
        )


@dataclass(frozen=True, slots=True)
class ScorerBatch:
    descriptors: npt.NDArray[np.float64]
    texture: npt.NDArray[np.float64]
    concepts: npt.NDArray[np.float64]
    context: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    band_target: npt.NDArray[np.int64]
    tier_target: npt.NDArray[np.int64]
    severity_target: npt.NDArray[np.int64]
    risk_target: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64] = field(default_factory=lambda: np.ones(0, dtype=np.float64))

    def width(self) -> int:
        return int(self.descriptors.shape[1] + self.texture.shape[1] + self.concepts.shape[1] + self.embedding.shape[1])

    def size(self) -> int:
        return int(self.descriptors.shape[0])

    def slice(self, index: npt.NDArray[np.int64]) -> ScorerBatch:
        return ScorerBatch(
            descriptors=self.descriptors[index],
            texture=self.texture[index],
            concepts=self.concepts[index],
            context=self.context[index],
            embedding=self.embedding[index],
            band_target=self.band_target[index],
            tier_target=self.tier_target[index],
            severity_target=self.severity_target[index],
            risk_target=self.risk_target[index],
            weights=self.weights[index] if self.weights.size else self.weights,
        )


def features_to_tensors(batch: ScorerBatch, device: torch.device) -> ScorerFeatures:
    return ScorerFeatures(
        descriptors=torch.as_tensor(batch.descriptors, dtype=torch.float32, device=device),
        texture=torch.as_tensor(batch.texture, dtype=torch.float32, device=device),
        concepts=torch.as_tensor(batch.concepts, dtype=torch.float32, device=device),
        context=torch.as_tensor(batch.context, dtype=torch.float32, device=device),
        embedding=torch.as_tensor(batch.embedding, dtype=torch.float32, device=device),
    )


def zero_features(size: int, texture_dim: int, concept_dim: int, embedding_dim: int, descriptor_dim: int) -> ScorerBatch:
    return ScorerBatch(
        descriptors=np.zeros((size, descriptor_dim), dtype=np.float64),
        texture=np.zeros((size, texture_dim), dtype=np.float64),
        concepts=np.zeros((size, concept_dim), dtype=np.float64),
        context=np.zeros((size, 2), dtype=np.float64),
        embedding=np.zeros((size, embedding_dim), dtype=np.float64),
        band_target=np.zeros(size, dtype=np.int64),
        tier_target=np.zeros(size, dtype=np.int64),
        severity_target=np.zeros(size, dtype=np.int64),
        risk_target=np.zeros(size, dtype=np.float64),
    )
