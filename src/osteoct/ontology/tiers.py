"""FRAX-based risk-tier semantics over the diagnostic band and the fracture history."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.concepts import ConceptIndex, TermSystem
from osteoct.schema.records import AcquisitionDescriptor
from osteoct.schema.vocabulary import TIER_ORDER, DiagnosticBand, DiagnosticContext, RiskTier

BAND_TO_TIER: dict[DiagnosticBand, RiskTier] = {
    DiagnosticBand.NORMAL: RiskTier.LOW,
    DiagnosticBand.OSTEOPENIA: RiskTier.MODERATE,
    DiagnosticBand.OSTEOPOROSIS: RiskTier.HIGH,
    DiagnosticBand.SEVERE_OSTEOPOROSIS: RiskTier.VERY_HIGH,
}

FRAX_CONCEPT_ELEMENTS: tuple[str, ...] = ("risk_tier",)


@dataclass(frozen=True, slots=True)
class ProtocolEnvelope:
    maximum_slice_thickness_mm: float
    minimum_tube_voltage_kvp: float
    accepted_kernels: tuple[str, ...]

    def admits(self, acquisition: AcquisitionDescriptor) -> bool:
        if acquisition.slice_thickness_mm > self.maximum_slice_thickness_mm:
            return False
        if acquisition.tube_voltage_kvp < self.minimum_tube_voltage_kvp:
            return False
        return acquisition.reconstruction_kernel in self.accepted_kernels


def tier_from_band(
    band: DiagnosticBand,
    context: DiagnosticContext,
    acquisition: AcquisitionDescriptor,
    concepts: tuple[str, ...],
    index: ConceptIndex,
    envelope: ProtocolEnvelope,
) -> RiskTier:
    base = BAND_TO_TIER[band]
    prior_fracture, treatment_exposure = context.as_flags()
    if prior_fracture:
        base = _shift(base, +1)
    if treatment_exposure and not prior_fracture:
        base = _shift(base, -1)
    if not _risk_semantics_bound(concepts, index):
        base = _shift(base, -1)
    if not envelope.admits(acquisition):
        base = _shift(base, -1)
    return base


def _risk_semantics_bound(concepts: tuple[str, ...], index: ConceptIndex) -> bool:
    table = index.alias_table()
    for token in concepts:
        concept = table.get(token)
        if concept is not None and concept.system is TermSystem.LOINC:
            return True
    return False


def _shift(tier: RiskTier, delta: int) -> RiskTier:
    rank = TIER_ORDER.index(tier) + delta
    rank = max(0, min(len(TIER_ORDER) - 1, rank))
    return TIER_ORDER[rank]


def tier_rank(tier: RiskTier) -> int:
    return TIER_ORDER.index(tier)


def frax_probability(band: DiagnosticBand, tier: RiskTier, prior_fragility_fracture: bool, treatment_exposure: bool) -> float:
    base = {
        DiagnosticBand.NORMAL: 0.041,
        DiagnosticBand.OSTEOPENIA: 0.087,
        DiagnosticBand.OSTEOPOROSIS: 0.196,
        DiagnosticBand.SEVERE_OSTEOPOROSIS: 0.284,
    }[band]
    factor = 1.0
    if prior_fragility_fracture:
        factor *= 1.62
    if treatment_exposure:
        factor *= 0.78
    factor *= 1.0 + 0.04 * tier_rank(tier)
    return min(0.95, base * factor)
