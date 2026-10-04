"""Exponential moving average of the trained parameters."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


class EmaTracker:
    def __init__(self, modules: Iterable[nn.Module], decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("the EMA decay must lie strictly inside (0, 1)")
        self.decay = decay
        self.modules = tuple(modules)
        self.shadow: dict[str, Tensor] = {}
        self.updates = 0
        for name, tensor in self._named():
            self.shadow[name] = tensor.detach().clone()

    def _named(self) -> tuple[tuple[str, Tensor], ...]:
        entries: list[tuple[str, Tensor]] = []
        for index, module in enumerate(self.modules):
            for key, value in module.state_dict().items():
                entries.append((f"{index}:{key}", value))
        return tuple(entries)

    @torch.no_grad()
    def update(self) -> None:
        self.updates += 1
        for name, tensor in self._named():
            current = tensor.detach()
            if current.dtype.is_floating_point:
                shadow = self.shadow[name]
                self.shadow[name] = shadow * self.decay + current * (1.0 - self.decay)
            else:
                self.shadow[name] = current.clone()

    @torch.no_grad()
    def copy_to(self, modules: Iterable[nn.Module] | None = None) -> None:
        targets = self.modules if modules is None else tuple(modules)
        for index, module in enumerate(targets):
            payload: dict[str, Tensor] = {}
            for key, value in module.state_dict().items():
                name = f"{index}:{key}"
                payload[key] = self.shadow[name].clone() if name in self.shadow else value
            module.load_state_dict(payload)

    def bias_corrected(self) -> dict[str, Tensor]:
        if self.updates == 0:
            return {name: value.clone() for name, value in self.shadow.items()}
        correction = 1.0 - self.decay**self.updates
        return {name: value / correction for name, value in self.shadow.items()}

    def state(self) -> dict[str, Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def divergence(self) -> float:
        total = 0.0
        for name, tensor in self._named():
            total += float((tensor.detach() - self.shadow[name]).pow(2).sum().item())
        return float(total**0.5)
