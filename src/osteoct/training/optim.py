"""Optimiser construction and gradient handling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OptimConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    gradient_clip: float = 1.0

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "encoder_learning_rate": self.encoder_learning_rate,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "gradient_clip": self.gradient_clip,
        }


def build_optimizer(
    scorer_parameters: Iterable[nn.Parameter],
    encoder_parameters: Iterable[nn.Parameter],
    config: OptimConfig,
) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = [
        {"params": [parameter for parameter in scorer_parameters if parameter.requires_grad], "lr": config.learning_rate},
    ]
    encoder = [parameter for parameter in encoder_parameters if parameter.requires_grad]
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_learning_rate})
    if config.name == "adamw":
        return torch.optim.AdamW(
            groups, lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
        )
    if config.name == "adam":
        return torch.optim.Adam(
            groups, lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
        )
    if config.name == "sgd":
        return torch.optim.SGD(groups, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    raise KeyError(f"unknown optimiser {config.name!r}")


def clip_gradients(parameters: Iterable[nn.Parameter], maximum_norm: float) -> float:
    listed = [parameter for parameter in parameters if parameter.grad is not None]
    if not listed or maximum_norm <= 0.0:
        return 0.0
    norm: Tensor = torch.nn.utils.clip_grad_norm_(listed, maximum_norm)
    return float(norm.detach().item())


def parameter_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().pow(2).sum().item())
    return float(total**0.5)


def backpropagate(loss: Tensor) -> None:
    torch.autograd.backward(loss)


@dataclass(slots=True)
class ParameterGroups:
    groups: dict[str, list[nn.Parameter]] = field(default_factory=dict)

    def add(self, name: str, parameters: Iterable[nn.Parameter]) -> None:
        self.groups[name] = list(parameters)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.groups))

    def count(self) -> int:
        return sum(len(parameters) for parameters in self.groups.values())
