"""The standard-conformance endpoint and its components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osteoct.ontology.axioms import AxiomSet, conformance_score, supported_code
from osteoct.schema.records import CandidateDecision, DecisionRecord, MeasurementBundle


@dataclass(frozen=True, slots=True)
class ConformanceSummary:
    emitted: int
    abstained: int
    processed: int
    mean_conformance: float
    unsupported_rate: float
    abstention_rate: float
    family_violation_rates: dict[str, float]

    def as_mapping(self) -> dict[str, float]:
        payload = {
            "emitted": float(self.emitted),
            "abstained": float(self.abstained),
            "processed": float(self.processed),
            "mean_conformance": self.mean_conformance,
            "unsupported_rate": self.unsupported_rate,
            "abstention_rate": self.abstention_rate,
        }
        for family, rate in self.family_violation_rates.items():
            payload[f"violation_rate_{family}"] = rate
        return payload


def record_conformance(axiom_set: AxiomSet, candidate: CandidateDecision, bundle: MeasurementBundle) -> float:
    return conformance_score(axiom_set, candidate, bundle)


def record_supported(axiom_set: AxiomSet, candidate: CandidateDecision, bundle: MeasurementBundle) -> bool:
    return supported_code(candidate, axiom_set, bundle)


def summarise(
    axiom_set: AxiomSet,
    records: tuple[DecisionRecord, ...],
    candidates: tuple[CandidateDecision, ...],
    bundles: tuple[MeasurementBundle, ...],
    processed: int,
) -> ConformanceSummary:
    if len(records) != len(candidates):
        raise ValueError("one candidate per admitted record is required")
    if len(records) != len(bundles):
        raise ValueError("one bundle per admitted record is required")
    scores = [conformance_score(axiom_set, candidate, bundle) for candidate, bundle in zip(candidates, bundles, strict=True)]
    unsupported = [
        1.0 for candidate, bundle in zip(candidates, bundles, strict=True) if not supported_code(candidate, axiom_set, bundle)
    ]
    family_counts = {family.value: 0 for family in axiom_set.all_families()}
    for candidate, bundle in zip(candidates, bundles, strict=True):
        seen = {violation.family.value for violation in axiom_set.violations(candidate, bundle)}
        for family in seen:
            if family in family_counts:
                family_counts[family] += 1
    denominator = max(1, len(records))
    family_rates = {family: count / denominator for family, count in family_counts.items()}
    emitted = len(records)
    abstained = max(0, processed - emitted)
    return ConformanceSummary(
        emitted=emitted,
        abstained=abstained,
        processed=processed,
        mean_conformance=float(np.mean(scores)) if scores else 0.0,
        unsupported_rate=float(len(unsupported) / denominator),
        abstention_rate=float(abstained / processed) if processed > 0 else 0.0,
        family_violation_rates=family_rates,
    )


def interval_conformance(
    axiom_set: AxiomSet,
    band_is_consistent: tuple[bool, ...],
    evidence_complete: tuple[bool, ...],
) -> float:
    if len(band_is_consistent) != len(evidence_complete):
        raise ValueError("both families must be scored on the same records")
    if not band_is_consistent:
        return 0.0
    band_share = float(np.mean(np.asarray(band_is_consistent, dtype=np.float64)))
    evidence_share = float(np.mean(np.asarray(evidence_complete, dtype=np.float64)))
    return 0.5 * (band_share + evidence_share)


def unsupported_codes(axiom_set: AxiomSet, candidate: CandidateDecision) -> tuple[str, ...]:
    return axiom_set.unresolved(candidate)
