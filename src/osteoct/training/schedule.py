"""Learning-rate schedules: linear warmup followed by cosine or step decay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    name: str = "cosine"
    warmup_steps: int = 500
    total_steps: int = 14000
    minimum_ratio: float = 0.02
    step_size: int = 1500
    decay_factor: float = 0.5

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "minimum_ratio": self.minimum_ratio,
            "step_size": self.step_size,
            "decay_factor": self.decay_factor,
        }


class WarmupSchedule:
    def __init__(self, config: ScheduleConfig, base_learning_rate: float) -> None:
        if config.warmup_steps < 0:
            raise ValueError("the warmup length cannot be negative")
        if config.total_steps <= 0:
            raise ValueError("the schedule needs a positive horizon")
        self.config = config
        self.base_learning_rate = base_learning_rate

    def factor(self, step: int) -> float:
        if step < self.config.warmup_steps:
            if self.config.warmup_steps == 0:
                return 1.0
            return float((step + 1) / self.config.warmup_steps)
        decayed = self._decay(step - self.config.warmup_steps)
        return float(max(self.config.minimum_ratio, decayed))

    def learning_rate(self, step: int) -> float:
        return float(self.base_learning_rate * self.factor(step))

    def _decay(self, step: int) -> float:
        if self.config.name == "cosine":
            span = max(1, self.config.total_steps - self.config.warmup_steps)
            progress = min(1.0, step / float(span))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(self.config.minimum_ratio + (1.0 - self.config.minimum_ratio) * cosine)
        if self.config.name == "step":
            exponent = step // max(1, self.config.step_size)
            return float(self.config.decay_factor**exponent)
        if self.config.name == "linear":
            span = max(1, self.config.total_steps - self.config.warmup_steps)
            progress = min(1.0, step / float(span))
            return float(self.config.minimum_ratio + (1.0 - self.config.minimum_ratio) * (1.0 - progress))
        if self.config.name == "constant":
            return 1.0
        raise KeyError(f"unknown schedule {self.config.name!r}")

    def curve(self, steps: int) -> tuple[float, ...]:
        return tuple(self.learning_rate(step) for step in range(steps))


def apply_learning_rate(optimizer: object, learning_rate: float) -> None:
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        raise TypeError("the optimiser exposes no param_groups")
    for group in groups:
        group["lr"] = learning_rate


def schedule_curve(schedule: WarmupSchedule, steps: int) -> np.ndarray:
    return np.asarray(schedule.curve(steps), dtype=np.float64)
