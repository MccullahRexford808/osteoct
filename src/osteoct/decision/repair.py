"""Identifier resolution and the deterministic repair loop of the constraint operator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from osteoct.decision.evidence import attach
from osteoct.ontology.axioms import AxiomSet, FamilyName, Violation, band_after_interval
from osteoct.ontology.severity import grade_from_vertebrae
from osteoct.ontology.tiers import tier_from_band
from osteoct.schema.concepts import ConceptIndex
from osteoct.schema.records import CandidateDecision, MeasurementBundle


@dataclass(frozen=True, slots=True)
class RepairStep:
    family: FamilyName
    index: int
    detail: str


@dataclass(frozen=True, slots=True)
class RepairTrace:
    candidate: CandidateDecision
    steps: tuple[RepairStep, ...]
    terminal_violation: Violation | None
    resolved_identifiers: bool
    budget_exhausted: bool

    def families(self) -> tuple[FamilyName, ...]:
        return tuple(step.family for step in self.steps)

    def repairs(self) -> int:
        return len(self.steps)


def resolve_identifiers(candidate: CandidateDecision, index: ConceptIndex) -> CandidateDecision:
    table = index.alias_table()
    resolved: list[str] = []
    changed = False
    for token in candidate.concepts:
        if token in table:
            resolved.append(token)
            continue
        replacement = _resolve_token(token, table)
        if replacement is None:
            resolved.append(token)
            continue
        resolved.append(replacement)
        changed = True
    if not changed and tuple(resolved) == candidate.concepts:
        return candidate
    return CandidateDecision(
        band=candidate.band,
        tier=candidate.tier,
        severity=candidate.severity,
        concepts=tuple(dict.fromkeys(resolved)),
        evidence=candidate.evidence,
        probability=candidate.probability,
        context=candidate.context,
    )


def _resolve_token(token: str, table: Mapping[str, object]) -> str | None:
    lowered = token.strip().lower()
    if lowered in table:
        return lowered
    for key in table:
        if key.lower() == lowered:
            return key
    suffix = token.split(":")[-1].strip().lower()
    for key in table:
        if key.split(":")[-1].lower() == suffix:
            return key
    for key in table:
        if suffix and suffix in key.lower():
            return key
    return None


def is_repairable(axiom_set: AxiomSet, candidate: CandidateDecision, bundle: MeasurementBundle) -> bool:
    repairable = set(axiom_set.repairable_families())
    return any(violation.family in repairable for violation in axiom_set.violations(candidate, bundle))


def apply_repair(
    family: FamilyName,
    candidate: CandidateDecision,
    bundle: MeasurementBundle,
    axiom_set: AxiomSet,
) -> CandidateDecision:
    if family is FamilyName.BAND:
        derived = band_after_interval(axiom_set, bundle)
        return candidate.with_band(derived)
    if family is FamilyName.TIER:
        derived_tier = tier_from_band(
            candidate.band,
            candidate.context,
            bundle.acquisition,
            candidate.concepts,
            axiom_set.index,
            axiom_set.envelope,
        )
        return candidate.with_tier(derived_tier)
    if family is FamilyName.SEVERITY:
        derived_severity = grade_from_vertebrae(
            tuple(measurement.morphology for measurement in bundle.vertebrae), axiom_set.thresholds
        )
        return candidate.with_severity(derived_severity)
    required = axiom_set.required_evidence(candidate)
    return candidate.with_evidence(attach(candidate.evidence, required))


def repair_loop(
    axiom_set: AxiomSet,
    candidate: CandidateDecision,
    bundle: MeasurementBundle,
    budget: int,
) -> RepairTrace:
    if budget < 0:
        raise ValueError("the repair budget cannot be negative")
    working = candidate
    steps: list[RepairStep] = []
    counter = 0
    while is_repairable(axiom_set, working, bundle) and counter < budget:
        violation = axiom_set.first_violation(working, bundle)
        if violation is None:
            break
        if violation.family not in set(axiom_set.repairable_families()):
            break
        working = apply_repair(violation.family, working, bundle, axiom_set)
        steps.append(RepairStep(family=violation.family, index=counter, detail=violation.detail))
        counter += 1
    remaining = axiom_set.first_violation(working, bundle)
    exhausted = remaining is not None and counter >= budget
    return RepairTrace(
        candidate=working,
        steps=tuple(steps),
        terminal_violation=remaining,
        resolved_identifiers=True,
        budget_exhausted=exhausted,
    )


def repair_order(axiom_set: AxiomSet) -> tuple[FamilyName, ...]:
    return axiom_set.repairable_families()


def first_violation_family(axiom_set: AxiomSet, candidate: CandidateDecision, bundle: MeasurementBundle) -> FamilyName | None:
    violation = axiom_set.first_violation(candidate, bundle)
    return None if violation is None else violation.family
