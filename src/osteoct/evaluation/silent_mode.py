"""Silent-mode deployment audit: what the agent emits without a reader in the loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.agent.pipeline import Policy
from osteoct.decision.abstention import describe
from osteoct.evaluation.study import PARTITION_EXTERNAL, ArmResult, StudyContext, arm_result
from osteoct.ontology.axioms import FamilyName
from osteoct.training.trainer import TrainingResult


@dataclass(frozen=True, slots=True)
class SilentModeAudit:
    partition: str
    processed: int
    emitted: int
    abstained: int
    abstention_reasons: dict[str, int]
    mean_conformance: float
    unsupported_rate: float
    mean_risk: float
    escalated_share: float
    threshold: float
    family_violation_rates: dict[str, float]
    repaired_share: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "processed": self.processed,
            "emitted": self.emitted,
            "abstained": self.abstained,
            "abstention_reasons": self.abstention_reasons,
            "mean_conformance": round(self.mean_conformance, 4),
            "unsupported_rate": round(self.unsupported_rate, 4),
            "mean_risk": round(self.mean_risk, 4),
            "escalated_share": round(self.escalated_share, 4),
            "escalation_threshold": round(self.threshold, 4),
            "family_violation_rates": {key: round(value, 4) for key, value in self.family_violation_rates.items()},
            "repaired_share": round(self.repaired_share, 4),
        }


def family_violation_rates(result: ArmResult) -> dict[str, float]:
    emitted = [record for record in result.ensemble if record.emitted]
    counts = {family.value: 0 for family in FamilyName}
    for record in emitted:
        for family in record.violated_families:
            counts[family] = counts.get(family, 0) + 1
    denominator = max(1, len(emitted))
    return {family: count / denominator for family, count in counts.items()}


def run_silent_mode(
    context: StudyContext,
    arm: str,
    policy: Policy,
    training: dict[str, TrainingResult],
    threshold: float,
    partition: str = PARTITION_EXTERNAL,
) -> SilentModeAudit:
    rows = context.rows(partition)
    result = arm_result(context, arm, policy, rows, training)
    emitted = [record for record in result.ensemble if record.emitted]
    abstained = [record for record in result.ensemble if not record.emitted]
    reasons: dict[str, int] = {}
    for record in abstained:
        reasons[record.reason] = reasons.get(record.reason, 0) + 1
    escalated = [record for record in emitted if record.risk >= threshold]
    violations = family_violation_rates(result)
    repaired = [record for record in emitted if record.repairs > 0]
    return SilentModeAudit(
        partition=partition,
        processed=int(rows.size),
        emitted=len(emitted),
        abstained=len(abstained),
        abstention_reasons={key: value for key, value in sorted(reasons.items())},
        mean_conformance=float(np.mean([record.conformance for record in emitted])) if emitted else 0.0,
        unsupported_rate=float(np.mean([1.0 if record.unsupported else 0.0 for record in emitted])) if emitted else 0.0,
        mean_risk=float(np.mean([record.risk for record in emitted])) if emitted else 0.0,
        escalated_share=float(len(escalated) / max(1, len(emitted))),
        threshold=float(threshold),
        family_violation_rates=violations,
        repaired_share=float(len(repaired) / max(1, len(emitted))),
    )


def abstention_reason_text(reason: str) -> str:
    return describe(reason)


def external_only(context: StudyContext) -> npt.NDArray[np.int64]:
    return context.rows(PARTITION_EXTERNAL)


def recount_reasons(audit: SilentModeAudit) -> int:
    return int(sum(audit.abstention_reasons.values()))
