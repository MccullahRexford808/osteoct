"""The main comparison table over the internal multicentre test partition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.agent.pipeline import Policy
from osteoct.evaluation.study import PARTITION_INTERNAL, ArmResult, StudyContext, arm_result
from osteoct.metrics.discrimination import auc, band_is_osteoporotic, sensitivity_at_specificity
from osteoct.statistics.bootstrap import bootstrap_interval
from osteoct.training.trainer import TrainingResult


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    method: str
    policy: str
    auroc: float
    auroc_low: float
    auroc_high: float
    sensitivity_at_90: float
    conformance_mean: float
    conformance_sd: float
    unsupported_mean: float
    unsupported_sd: float
    abstention_mean: float
    abstention_sd: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "method": self.method,
            "policy": self.policy,
            "auroc": round(self.auroc, 4),
            "auroc_ci": [round(self.auroc_low, 4), round(self.auroc_high, 4)],
            "sensitivity_at_90_specificity": round(self.sensitivity_at_90, 4),
            "conformance": round(self.conformance_mean, 4),
            "conformance_sd": round(self.conformance_sd, 4),
            "unsupported": round(self.unsupported_mean, 4),
            "unsupported_sd": round(self.unsupported_sd, 4),
            "abstention": round(self.abstention_mean, 4),
            "abstention_sd": round(self.abstention_sd, 4),
        }


@dataclass(frozen=True, slots=True)
class ComparisonTable:
    partition: str
    rows: tuple[ComparisonRow, ...]
    labelled_examinations: int
    patients_with_followup: int
    seeds: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "labelled_examinations": self.labelled_examinations,
            "patients_with_followup": self.patients_with_followup,
            "seeds": self.seeds,
            "rows": [row.as_mapping() for row in self.rows],
        }

    def row(self, method: str) -> ComparisonRow:
        for entry in self.rows:
            if entry.method == method:
                return entry
        raise KeyError(f"unknown method {method!r}")


def reference_labels(context: StudyContext, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return band_is_osteoporotic(context.encoded.band_target[rows])


def bootstrap_auroc_interval(
    context: StudyContext,
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    patients = context.patient_ids(rows)

    def statistic(drawn: npt.NDArray[np.int64]) -> float:
        return auc(scores[drawn], labels[drawn])

    return bootstrap_interval(patients, statistic, replicates, seed)


def evaluate_arm(
    context: StudyContext,
    arm: str,
    method: str,
    policy: Policy,
    training: dict[str, TrainingResult],
    replicates: int,
    seed: int,
) -> tuple[ArmResult, ComparisonRow]:
    rows = context.rows(PARTITION_INTERNAL)
    result = arm_result(context, arm, policy, rows, training)
    labels = reference_labels(context, rows)
    ensemble_risk = np.asarray([record.risk for record in result.ensemble], dtype=np.float64)
    emitted_mask = np.asarray([record.emitted for record in result.ensemble], dtype=bool)
    emitted_risk = ensemble_risk[emitted_mask]
    emitted_labels = labels[emitted_mask]
    point = auc(emitted_risk, emitted_labels)
    low, high = bootstrap_auroc_interval(context, rows[emitted_mask], emitted_risk, emitted_labels, replicates, seed)
    sensitivity = sensitivity_at_specificity(emitted_risk, emitted_labels, 0.90)
    conformance = result.conformance_per_seed()
    unsupported = result.unsupported_per_seed()
    abstention = result.abstention_per_seed()
    row = ComparisonRow(
        method=method,
        policy=policy.value,
        auroc=float(point),
        auroc_low=float(low),
        auroc_high=float(high),
        sensitivity_at_90=float(sensitivity),
        conformance_mean=float(np.mean(conformance)) if conformance else 0.0,
        conformance_sd=float(np.std(conformance, ddof=1)) if len(conformance) > 1 else 0.0,
        unsupported_mean=float(np.mean(unsupported)) if unsupported else 0.0,
        unsupported_sd=float(np.std(unsupported, ddof=1)) if len(unsupported) > 1 else 0.0,
        abstention_mean=float(np.mean(abstention)) if abstention else 0.0,
        abstention_sd=float(np.std(abstention, ddof=1)) if len(abstention) > 1 else 0.0,
    )
    return (result, row)


def run_main_comparison(
    context: StudyContext,
    training: dict[str, TrainingResult],
    specification: tuple[tuple[str, str, Policy], ...],
    replicates: int,
    seed: int,
) -> tuple[ComparisonTable, dict[str, ArmResult]]:
    rows: list[ComparisonRow] = []
    results: dict[str, ArmResult] = {}
    for arm, method, policy in specification:
        result, row = evaluate_arm(context, arm, method, policy, training, replicates, seed)
        results[method] = result
        rows.append(row)
    test_rows = context.rows(PARTITION_INTERNAL)
    patients = {record.patient_id for record in context.encoded.records if record.has_followup()}
    seeds = max((len(runs.runs) for runs in training.values()), default=0)
    table = ComparisonTable(
        partition=PARTITION_INTERNAL,
        rows=tuple(rows),
        labelled_examinations=int(test_rows.size),
        patients_with_followup=len(patients),
        seeds=seeds,
    )
    return (table, results)


def delta_against_reference(table: ComparisonTable, reference: str = "OSTEO-CT (constrained)") -> dict[str, float]:
    base = table.row(reference).auroc
    return {row.method: float(row.auroc - base) for row in table.rows}


def conformance_delta(table: ComparisonTable, reference: str = "OSTEO-CT (constrained)") -> dict[str, float]:
    base = table.row(reference).conformance_mean
    return {row.method: float(row.conformance_mean - base) for row in table.rows}


def unconstrained_comparator(table: ComparisonTable) -> ComparisonRow | None:
    for row in table.rows:
        if row.policy == Policy.UNCONSTRAINED.value:
            return row
    return None


def ordering(table: ComparisonTable) -> tuple[str, ...]:
    return tuple(row.method for row in sorted(table.rows, key=lambda entry: -entry.auroc))
