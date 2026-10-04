"""The ontology-constrained decision operator, the operator of Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.decision.abstention import abstain
from osteoct.decision.conformance import record_conformance
from osteoct.decision.evidence import close_chain
from osteoct.decision.repair import RepairTrace, repair_loop, resolve_identifiers
from osteoct.ontology.axioms import AxiomSet, FamilyName
from osteoct.schema.records import Abstention, AdmittedDecision, CandidateDecision, MeasurementBundle


@dataclass(frozen=True, slots=True)
class OperatorOutcome:
    admitted: AdmittedDecision | None
    abstention: Abstention | None
    trace: RepairTrace

    def emitted(self) -> bool:
        return self.admitted is not None

    def conformance(self) -> float:
        return 0.0 if self.admitted is None else self.admitted.conformance

    def reason(self) -> str:
        return "" if self.abstention is None else self.abstention.reason


def constrain(
    candidate: CandidateDecision,
    bundle: MeasurementBundle,
    axiom_set: AxiomSet,
    budget: int,
) -> OperatorOutcome:
    working = candidate
    resolved = resolve_identifiers(working, axiom_set.index)
    working = resolved
    unresolved = axiom_set.unresolved(working)
    if unresolved:
        return OperatorOutcome(None, abstain("unresolved_identifier", "resolve", 0, ", ".join(unresolved)), _empty_trace(working))

    trace = repair_loop(axiom_set, working, bundle, budget)
    working = trace.candidate
    if axiom_set.hard_violations(working, bundle):
        violation = trace.terminal_violation
        reason = "budget_exhausted" if trace.budget_exhausted else (violation.family.value if violation is not None else "band")
        return OperatorOutcome(None, abstain(reason, "operate", trace.repairs()), trace)

    closed = working.with_evidence(close_chain(working.evidence, bundle))
    score = record_conformance(axiom_set, closed, bundle)
    violated = tuple(
        sorted(
            {violation.family.value for violation in axiom_set.violations(closed, bundle)}, key=lambda label: _family_rank(label)
        )
    )
    admitted = AdmittedDecision(candidate=closed, conformance=score, repairs=trace.repairs(), violated_families=violated)
    return OperatorOutcome(admitted, None, trace)


def _family_rank(label: str) -> int:
    for position, family in enumerate(FamilyName):
        if family.value == label:
            return position
    return len(FamilyName)


def _empty_trace(candidate: CandidateDecision) -> RepairTrace:
    return RepairTrace(candidate=candidate, steps=(), terminal_violation=None, resolved_identifiers=False, budget_exhausted=False)


def unconstrained(candidate: CandidateDecision, bundle: MeasurementBundle, axiom_set: AxiomSet) -> AdmittedDecision:
    score = record_conformance(axiom_set, candidate, bundle)
    violated = tuple(
        sorted(
            {violation.family.value for violation in axiom_set.violations(candidate, bundle)}, key=lambda label: _family_rank(label)
        )
    )
    return AdmittedDecision(candidate=candidate, conformance=score, repairs=0, violated_families=violated)


def enumerate_admissible(
    candidates: tuple[CandidateDecision, ...],
    bundle: MeasurementBundle,
    axiom_set: AxiomSet,
) -> tuple[CandidateDecision, ...]:
    return tuple(candidate for candidate in candidates if axiom_set.satisfies(candidate, bundle))


def repair_sequence(
    candidate: CandidateDecision,
    bundle: MeasurementBundle,
    axiom_set: AxiomSet,
    budget: int,
) -> tuple[str, ...]:
    trace = repair_loop(axiom_set, candidate, bundle, budget)
    return tuple(step.family.value for step in trace.steps)
