"""Constraint-family ablation and component substitution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.agent.pipeline import Policy
from osteoct.cohort.cohort import CohortRecord
from osteoct.evaluation.study import PARTITION_INTERNAL, ArmResult, StudyContext, arm_result
from osteoct.metrics.discrimination import auc, band_is_osteoporotic
from osteoct.metrics.survival import CensoringRule, concordance_index
from osteoct.ontology.axioms import AxiomSet, FamilyName
from osteoct.training.trainer import TrainingResult

FAMILY_LABELS: dict[FamilyName, str] = {
    FamilyName.IDENTITY: "identity",
    FamilyName.BAND: "band",
    FamilyName.TIER: "tier",
    FamilyName.SEVERITY: "severity",
    FamilyName.EVIDENCE: "evidence",
}


@dataclass(frozen=True, slots=True)
class AblationRow:
    configuration: str
    family: str | None
    auroc: float
    auroc_delta: float
    c_index: float
    conformance: float
    conformance_delta: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "family": self.family,
            "auroc": round(self.auroc, 4),
            "auroc_delta": round(self.auroc_delta, 4),
            "c_index": round(self.c_index, 4),
            "conformance": round(self.conformance, 4),
            "conformance_delta": round(self.conformance_delta, 4),
        }


@dataclass(frozen=True, slots=True)
class AblationTable:
    rows: tuple[AblationRow, ...]
    conformance_monotone: bool
    removed_family_lowers_conformance: dict[str, bool]

    def as_mapping(self) -> dict[str, object]:
        return {
            "rows": [row.as_mapping() for row in self.rows],
            "conformance_monotone_under_removal": self.conformance_monotone,
            "per_family_decline": self.removed_family_lowers_conformance,
        }

    def row(self, configuration: str) -> AblationRow:
        for entry in self.rows:
            if entry.configuration == configuration:
                return entry
        raise KeyError(f"unknown configuration {configuration!r}")


@dataclass(frozen=True, slots=True)
class OutcomeArrays:
    times: npt.NDArray[np.float64]
    events: npt.NDArray[np.bool_]
    treatment_start_days: npt.NDArray[np.float64]


def treatment_start_of(record: CohortRecord) -> float:
    return float("inf") if record.treatment_start_days is None else float(record.treatment_start_days)


def outcome_arrays(context: StudyContext, rows: npt.NDArray[np.int64]) -> OutcomeArrays:
    records = context.encoded.records
    return OutcomeArrays(
        times=np.asarray([event_time_of(records[int(row)]) for row in rows.tolist()], dtype=np.float64),
        events=np.asarray([not records[int(row)].censored for row in rows.tolist()], dtype=bool),
        treatment_start_days=np.asarray(
            [treatment_start_of(records[int(row)]) for row in rows.tolist()],
            dtype=np.float64,
        ),
    )


def event_time_of(record: CohortRecord) -> float:
    if record.fracture_time_days is not None:
        return float(record.fracture_time_days)
    return float(record.observation_time_days)


def score_once(
    context: StudyContext,
    arm: str,
    policy: Policy,
    rows: npt.NDArray[np.int64],
    training: dict[str, TrainingResult],
    axioms: AxiomSet,
) -> ArmResult:
    return arm_result(context, arm, policy, rows, training, axioms)


def metrics_of(
    result: ArmResult,
    labels: npt.NDArray[np.int64],
    outcomes: OutcomeArrays,
) -> tuple[float, float, float]:
    risk = np.asarray([record.risk for record in result.ensemble], dtype=np.float64)
    mask = np.asarray([record.emitted for record in result.ensemble], dtype=bool)
    if not bool(mask.any()):
        return (float("nan"), float("nan"), 0.0)
    auroc = float(auc(risk[mask], labels[mask]))
    c_index = float(
        concordance_index(
            outcomes.times[mask],
            outcomes.events[mask],
            risk[mask],
            CensoringRule(),
            outcomes.treatment_start_days[mask],
        )
    )
    values = result.conformance_per_seed()
    conformance = float(np.mean(values)) if values else 0.0
    return (auroc, c_index, conformance)


def run_family_ablation(
    context: StudyContext,
    training: dict[str, TrainingResult],
    substitutions: tuple[tuple[str, str, Policy], ...],
    reference_arm: str,
    reference_policy: Policy,
) -> AblationTable:
    rows = context.rows(PARTITION_INTERNAL)
    labels = band_is_osteoporotic(context.encoded.band_target[rows])
    outcomes = outcome_arrays(context, rows)
    base_result = score_once(context, reference_arm, reference_policy, rows, training, context.attributes)
    base_auroc, base_c_index, base_conformance = metrics_of(base_result, labels, outcomes)
    entries: list[AblationRow] = [
        AblationRow("all_families", None, base_auroc, 0.0, base_c_index, base_conformance, 0.0),
    ]
    declines: dict[str, bool] = {}
    for family in FamilyName:
        if family not in context.attributes.active:
            continue
        reduced = context.attributes.without_family(family)
        result = score_once(context, reference_arm, reference_policy, rows, training, reduced)
        auroc, c_index, conformance = metrics_of(result, labels, outcomes)
        label = FAMILY_LABELS[family]
        entries.append(
            AblationRow(
                configuration=f"without_{label}",
                family=label,
                auroc=auroc,
                auroc_delta=float(base_auroc - auroc),
                c_index=c_index,
                conformance=conformance,
                conformance_delta=float(conformance - base_conformance),
            )
        )
        declines[label] = bool(conformance < base_conformance - 1e-12)
    for arm, method, policy in substitutions:
        if arm not in training:
            continue
        result = score_once(context, arm, policy, rows, training, context.attributes)
        auroc, c_index, conformance = metrics_of(result, labels, outcomes)
        entries.append(
            AblationRow(
                configuration=method,
                family=None,
                auroc=auroc,
                auroc_delta=float(base_auroc - auroc),
                c_index=c_index,
                conformance=conformance,
                conformance_delta=float(conformance - base_conformance),
            )
        )
    monotone = all(declines.values()) if declines else True
    return AblationTable(rows=tuple(entries), conformance_monotone=monotone, removed_family_lowers_conformance=declines)


def non_monotone_rows(table: AblationTable, reference_configuration: str = "all_families") -> tuple[AblationRow, ...]:
    base = table.row(reference_configuration)
    return tuple(row for row in table.rows if row.auroc > base.auroc + 1e-12)


def utility_gain_rows(table: AblationTable, reference_configuration: str = "all_families") -> tuple[AblationRow, ...]:
    base = table.row(reference_configuration)
    return tuple(row for row in table.rows if row.c_index > base.c_index + 1e-12)


def smallest_utility_change(table: AblationTable, reference_configuration: str = "all_families") -> float:
    base = table.row(reference_configuration)
    changes = [abs(row.auroc - base.auroc) for row in table.rows if row.configuration != reference_configuration]
    return float(min(changes)) if changes else 0.0


def conformance_ordering(table: AblationTable) -> tuple[str, ...]:
    removable = [entry for entry in table.rows if entry.family is not None]
    ordered = sorted(removable, key=lambda entry: entry.conformance_delta, reverse=True)
    labels: list[str] = []
    for entry in ordered:
        if entry.family is not None:
            labels.append(entry.family)
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class CumulativePoint:
    label: str
    active_families: tuple[str, ...]
    auroc: float
    c_index: float
    conformance: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "active_families": list(self.active_families),
            "auroc": round(self.auroc, 4),
            "c_index": round(self.c_index, 4),
            "conformance": round(self.conformance, 4),
        }


def cumulative_family_axes(
    context: StudyContext,
    training: dict[str, TrainingResult],
    reference_arm: str,
    reference_policy: Policy,
) -> tuple[CumulativePoint, ...]:
    rows = context.rows(PARTITION_INTERNAL)
    labels = band_is_osteoporotic(context.encoded.band_target[rows])
    outcomes = outcome_arrays(context, rows)
    points: list[CumulativePoint] = []
    for count in range(0, len(FAMILY_LABELS) + 1):
        active = tuple(family for family in FamilyName if family in set(list(FAMILY_LABELS)[:count]))
        restricted = context.attributes.with_only(active)
        result = score_once(context, reference_arm, reference_policy, rows, training, restricted)
        auroc, c_index, conformance = metrics_of(result, labels, outcomes)
        points.append(
            CumulativePoint(
                label="none" if count == 0 else FAMILY_LABELS[list(FAMILY_LABELS)[count - 1]],
                active_families=tuple(family.value for family in active),
                auroc=auroc,
                c_index=c_index,
                conformance=conformance,
            )
        )
    return tuple(points)


def conformance_is_monotone(points: tuple[CumulativePoint, ...]) -> bool:
    return all(points[index].conformance <= points[index + 1].conformance + 1e-9 for index in range(len(points) - 1))


def utility_is_monotone(points: tuple[CumulativePoint, ...]) -> bool:
    return all(points[index].c_index <= points[index + 1].c_index + 1e-9 for index in range(len(points) - 1))


def utility_dips(points: tuple[CumulativePoint, ...]) -> tuple[str, ...]:
    return tuple(
        points[index + 1].label for index in range(len(points) - 1) if points[index + 1].c_index < points[index].c_index - 1e-9
    )
