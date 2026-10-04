"""The planner: the learned scorer plus the optional perception encoder it reads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from osteoct.models.arms import ArmBundle, build_arm
from osteoct.models.scorer import ScorerBatch, ScorerFeatures, ScorerOutput, features_to_tensors


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    arm: str = "constrained"
    descriptor_dim: int = 3
    texture_dim: int = 25
    concept_dim: int = 30
    embed_dim: int = 16
    hidden: int = 64
    dropout: float = 0.1
    device: str = "cpu"

    def as_mapping(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "descriptor_dim": self.descriptor_dim,
            "texture_dim": self.texture_dim,
            "concept_dim": self.concept_dim,
            "embed_dim": self.embed_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "device": self.device,
        }


class Planner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.bundle: ArmBundle = build_arm(
            config.arm,
            descriptor_dim=config.descriptor_dim,
            texture_dim=config.texture_dim,
            concept_dim=config.concept_dim,
            embed_dim=config.embed_dim,
            hidden=config.hidden,
            dropout=config.dropout,
        )
        self.device = torch.device(config.device)

    def parameters(self) -> list[nn.Parameter]:
        params = list(self.bundle.scorer.parameters())
        if self.bundle.encoder is not None:
            params.extend(self.bundle.encoder.parameters())
        return params

    def modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.bundle.scorer]
        if self.bundle.encoder is not None:
            modules.append(self.bundle.encoder)
        return tuple(modules)

    def to(self, device: str) -> Planner:
        self.device = torch.device(device)
        for module in self.modules():
            module.to(self.device)
        return self

    def train(self, flag: bool = True) -> None:
        for module in self.modules():
            module.train(flag)

    def eval(self) -> None:
        self.train(False)

    def encode(self, volumes: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        if self.bundle.encoder is None or volumes.size == 0:
            return np.zeros((int(volumes.shape[0]), 0), dtype=np.float64)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(volumes, dtype=np.float32), device=self.device)
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(1)
            encoded: Tensor = self.bundle.encoder(tensor)
        return np.asarray(encoded.detach().cpu().numpy(), dtype=np.float64)

    def encode_one(self, volume: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        batch = np.asarray(volume, dtype=np.float32)[None, ...]
        return self.encode(batch)

    def features(self, batch: ScorerBatch) -> ScorerFeatures:
        return features_to_tensors(batch, self.device)

    def forward(self, batch: ScorerBatch) -> ScorerOutput:
        output: ScorerOutput = self.bundle.scorer(self.features(batch))
        return output

    def spec(self) -> dict[str, object]:
        return {
            "arm": self.bundle.name,
            "encoder": self.bundle.spec.encoder,
            "planner": self.bundle.spec.planner,
            "use_descriptors": self.bundle.spec.use_descriptors,
            "use_texture": self.bundle.spec.use_texture,
            "use_concepts": self.bundle.spec.use_concepts,
            "use_embedding": self.bundle.spec.use_embedding,
            "embedding_dim": self.bundle.embedding_dim,
        }

    def state(self) -> dict[str, dict[str, Tensor]]:
        payload = {"scorer": {key: value.detach().clone() for key, value in self.bundle.scorer.state_dict().items()}}
        if self.bundle.encoder is not None:
            payload["encoder"] = {key: value.detach().clone() for key, value in self.bundle.encoder.state_dict().items()}
        return payload

    def load_state(self, payload: dict[str, dict[str, Tensor]]) -> None:
        self.bundle.scorer.load_state_dict(payload["scorer"])
        if self.bundle.encoder is not None and "encoder" in payload:
            self.bundle.encoder.load_state_dict(payload["encoder"])


def build_planner(config: PlannerConfig | None = None) -> Planner:
    return Planner(PlannerConfig() if config is None else config)


def planner_report(planner: Planner) -> dict[str, object]:
    return {"configuration": planner.config.as_mapping(), "structure": planner.spec()}
