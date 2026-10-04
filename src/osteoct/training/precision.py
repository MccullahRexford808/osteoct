"""Mixed-precision configuration."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

SUPPORTED_PRECISIONS: tuple[str, ...] = ("fp32", "bf16", "fp16", "tf32")


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    name: str = "bf16"

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_PRECISIONS:
            raise KeyError(f"unknown precision mode {self.name!r}")

    def autocast(self, device_type: str) -> AbstractContextManager[Any]:
        if self.name == "fp32":
            return nullcontext()
        if self.name == "tf32":
            return nullcontext()
        dtype = torch.bfloat16 if self.name == "bf16" else torch.float16
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)

    def uses_scaler(self) -> bool:
        return self.name == "fp16"

    def build_scaler(self) -> Any:
        if not self.uses_scaler():
            return None
        return torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]

    def as_mapping(self) -> dict[str, object]:
        return {"name": self.name, "uses_scaler": self.uses_scaler()}


def device_type_of(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def resolve_precision(name: str) -> PrecisionConfig:
    return PrecisionConfig(name=name)
