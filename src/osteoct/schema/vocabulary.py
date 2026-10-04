"""Clinical vocabulary the decision layer emits and constrains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticBand(str, Enum):
    NORMAL = "normal"
    OSTEOPENIA = "osteopenia"
    OSTEOPOROSIS = "osteoporosis"
    SEVERE_OSTEOPOROSIS = "severe_osteoporosis"


class RiskTier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DeformityGrade(int, Enum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class VertebralLevel(str, Enum):
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    T11 = "T11"
    T12 = "T12"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


BAND_ORDER: tuple[DiagnosticBand, ...] = (
    DiagnosticBand.NORMAL,
    DiagnosticBand.OSTEOPENIA,
    DiagnosticBand.OSTEOPOROSIS,
    DiagnosticBand.SEVERE_OSTEOPOROSIS,
)

TIER_ORDER: tuple[RiskTier, ...] = (
    RiskTier.LOW,
    RiskTier.MODERATE,
    RiskTier.HIGH,
    RiskTier.VERY_HIGH,
)

THORACOLUMBAR_LEVELS: tuple[VertebralLevel, ...] = (
    VertebralLevel.T4,
    VertebralLevel.T5,
    VertebralLevel.T6,
    VertebralLevel.T7,
    VertebralLevel.T8,
    VertebralLevel.T9,
    VertebralLevel.T10,
    VertebralLevel.T11,
    VertebralLevel.T12,
    VertebralLevel.L1,
    VertebralLevel.L2,
    VertebralLevel.L3,
    VertebralLevel.L4,
)


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    prior_fragility_fracture: bool = False
    treatment_exposure: bool = False
    reference_t_score: float | None = None

    def reference_density_available(self) -> bool:
        return self.reference_t_score is not None

    def as_flags(self) -> tuple[bool, bool]:
        return (self.prior_fragility_fracture, self.treatment_exposure)
