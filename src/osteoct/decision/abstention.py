"""Typed abstention records, one reason string per termination route."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.records import Abstention

TERMINATION_REASONS: tuple[str, ...] = (
    "no_assessable_vertebra",
    "unresolved_identifier",
    "band",
    "tier",
    "sev",
    "ev",
    "budget_exhausted",
)

REASON_DESCRIPTIONS: dict[str, str] = {
    "no_assessable_vertebra": "localisation produced no assessable vertebral body",
    "unresolved_identifier": "the concept set does not resolve against the released index after the repair step",
    "band": "the diagnostic band remains inconsistent with the harmonised interval after the repair budget",
    "tier": "the risk tier remains inconsistent with the band and the fracture history after the repair budget",
    "sev": "the reported deformity grade remains inconsistent with the measured morphology after the repair budget",
    "ev": "an asserted element remains without a measurement link after the repair budget",
    "budget_exhausted": "the repair budget was exhausted before the axiom set was satisfied",
}


def abstain(reason: str, stage: str, repairs: int = 0, detail: str = "") -> Abstention:
    if reason not in REASON_DESCRIPTIONS:
        raise KeyError(f"unknown termination reason {reason!r}")
    return Abstention(reason=reason, stage=stage, repairs=repairs, detail=detail if detail else REASON_DESCRIPTIONS[reason])


def describe(reason: str) -> str:
    return REASON_DESCRIPTIONS.get(reason, "unrecorded termination route")


@dataclass(frozen=True, slots=True)
class AbstentionTally:
    counts: dict[str, int]

    def add(self, reason: str) -> AbstentionTally:
        updated = dict(self.counts)
        updated[reason] = updated.get(reason, 0) + 1
        return AbstentionTally(updated)

    def total(self) -> int:
        return sum(self.counts.values())

    def share(self, reason: str, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return self.counts.get(reason, 0) / float(denominator)


def tally(reasons: tuple[str, ...]) -> AbstentionTally:
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return AbstentionTally(counts)
