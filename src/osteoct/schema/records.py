"""Datatypes exchanged between the measurement stages, the scorer and the operator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from osteoct.schema.vocabulary import (
    DeformityGrade,
    DiagnosticBand,
    DiagnosticContext,
    RiskTier,
    VertebralLevel,
)


@dataclass(frozen=True, slots=True)
class AcquisitionDescriptor:
    tube_voltage_kvp: float
    scanner_family: str
    reconstruction_kernel: str
    slice_thickness_mm: float

    def signature(self) -> str:
        return f"{self.tube_voltage_kvp:.0f}kVp|{self.scanner_family}|{self.reconstruction_kernel}|{self.slice_thickness_mm:.2f}mm"

    def as_vector(self) -> tuple[float, float, float]:
        kernel_code = float(sum(ord(char) for char in self.reconstruction_kernel) % 97)
        return (self.tube_voltage_kvp, float(self.scanner_family.__hash__() % 31), kernel_code)


@dataclass(frozen=True, slots=True)
class VertebraRegion:
    level: VertebralLevel
    centre_index: tuple[int, int, int]
    extent_index: tuple[int, int, int]
    voxel_count: int
    trabecular_mask_fraction: float


@dataclass(frozen=True, slots=True)
class MorphologyFeatures:
    anterior_height_mm: float
    middle_height_mm: float
    posterior_height_mm: float

    def wedge_ratio(self) -> float:
        return _safe_ratio(self.anterior_height_mm, self.posterior_height_mm)

    def biconcavity_ratio(self) -> float:
        return _safe_ratio(self.middle_height_mm, self.posterior_height_mm)

    def crush_ratio(self) -> float:
        reference = max(self.anterior_height_mm, self.middle_height_mm, self.posterior_height_mm)
        return _safe_ratio(min(self.anterior_height_mm, self.middle_height_mm), reference)

    def height_loss(self) -> float:
        reference = self.posterior_height_mm
        reduced = min(self.anterior_height_mm, self.middle_height_mm)
        return 1.0 - _safe_ratio(reduced, reference)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class VertebraMeasurement:
    region: VertebraRegion
    morphology: MorphologyFeatures
    mean_attenuation_hu: float
    attenuation_sd_hu: float


@dataclass(frozen=True, slots=True)
class MeasurementBundle:
    standard_attenuation_hu: float
    uncertainty_hu: float
    texture_vector: tuple[float, ...]
    density_indicator: float
    acquisition: AcquisitionDescriptor
    vertebrae: tuple[VertebraMeasurement, ...]
    retained_texture_names: tuple[str, ...] = ()

    def interval(self) -> tuple[float, float]:
        return (self.standard_attenuation_hu - self.uncertainty_hu, self.standard_attenuation_hu + self.uncertainty_hu)

    def height_losses(self) -> tuple[float, ...]:
        return tuple(measurement.morphology.height_loss() for measurement in self.vertebrae)

    def mean_height_loss(self) -> float:
        losses = self.height_losses()
        if not losses:
            return 0.0
        return float(np.mean(losses))

    def bundle_signature(self) -> str:
        texture = ",".join(f"{value:.4f}" for value in self.texture_vector)
        return f"{self.standard_attenuation_hu:.3f}|{self.uncertainty_hu:.3f}|{self.density_indicator:.4f}|{texture}"


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    band: DiagnosticBand
    tier: RiskTier
    severity: DeformityGrade
    concepts: tuple[str, ...]
    evidence: tuple[str, ...]
    probability: float
    context: DiagnosticContext

    def with_band(self, band: DiagnosticBand) -> CandidateDecision:
        return self._replace(band=band)

    def with_tier(self, tier: RiskTier) -> CandidateDecision:
        return self._replace(tier=tier)

    def with_severity(self, severity: DeformityGrade) -> CandidateDecision:
        return self._replace(severity=severity)

    def with_evidence(self, evidence: tuple[str, ...]) -> CandidateDecision:
        return self._replace(evidence=evidence)

    def _replace(
        self,
        *,
        band: DiagnosticBand | None = None,
        tier: RiskTier | None = None,
        severity: DeformityGrade | None = None,
        evidence: tuple[str, ...] | None = None,
    ) -> CandidateDecision:
        return CandidateDecision(
            band=self.band if band is None else band,
            tier=self.tier if tier is None else tier,
            severity=self.severity if severity is None else severity,
            concepts=self.concepts,
            evidence=self.evidence if evidence is None else evidence,
            probability=self.probability,
            context=self.context,
        )


@dataclass(frozen=True, slots=True)
class AdmittedDecision:
    candidate: CandidateDecision
    conformance: float
    repairs: int
    violated_families: tuple[str, ...]

    def band(self) -> DiagnosticBand:
        return self.candidate.band


@dataclass(frozen=True, slots=True)
class Abstention:
    reason: str
    stage: str
    repairs: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Examination:
    record_id: str
    patient_id: str
    site: str
    region: str
    volume: np.ndarray
    acquisition: AcquisitionDescriptor
    context: DiagnosticContext
    incident_fracture_time_days: float | None = None
    observation_time_days: float | None = None
    censored: bool = True

    def has_reference_density(self) -> bool:
        return self.context.reference_density_available()

    def has_followup(self) -> bool:
        return self.incident_fracture_time_days is not None or self.observation_time_days is not None


@dataclass(frozen=True, slots=True)
class ConceptBinding:
    element: str
    concept_key: str
    measurement: str


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    record_id: str
    band: DiagnosticBand
    tier: RiskTier
    severity: DeformityGrade
    conformance: float
    bindings: tuple[ConceptBinding, ...]
    evidence: tuple[str, ...]
    repairs: int

    def as_mapping(self) -> Mapping[str, str]:
        return {
            "record_id": self.record_id,
            "band": self.band.value,
            "tier": self.tier.value,
            "severity": str(int(self.severity.value)),
            "conformance": f"{self.conformance:.6f}",
        }


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    patient_id: str
    site: str
    event_time_days: float
    observed: bool
    treatment_exposed: bool
    predicted_risk: float
    band: DiagnosticBand
    severity: DeformityGrade
    densitometric_band: DiagnosticBand | None = None


@dataclass(slots=True)
class RunLog:
    seeds: tuple[int, ...]
    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values[name] = value

    def mean(self, prefix: str) -> float:
        selected = [value for key, value in self.values.items() if key.startswith(prefix)]
        if not selected:
            return float("nan")
        return float(np.mean(selected))
