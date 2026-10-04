"""Clinical-outcome validation, decision utility and reclassification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.agent.pipeline import Policy
from osteoct.evaluation.family_ablation import OutcomeArrays, outcome_arrays
from osteoct.evaluation.study import PARTITION_INTERNAL, ArmResult, StudyContext, arm_result
from osteoct.metrics.calibration import CalibrationFit, calibration_slope_intercept
from osteoct.metrics.reclassification import Reclassification, reclassify
from osteoct.metrics.survival import CensoringRule, concordance_index
from osteoct.metrics.utility import net_benefit
from osteoct.statistics.bootstrap import bootstrap_interval
from osteoct.training.trainer import TrainingResult

NET_BENEFIT_THRESHOLDS: tuple[float, float] = (0.20, 0.30)
RECLASSIFICATION_CUT = 0.50


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    method: str
    c_index: float
    c_index_low: float
    c_index_high: float
    calibration: CalibrationFit
    net_benefit_20: float
    net_benefit_30: float
    events: int
    censored: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "method": self.method,
            "c_index": round(self.c_index, 4),
            "c_index_ci": [round(self.c_index_low, 4), round(self.c_index_high, 4)],
            "calibration_slope": round(self.calibration.slope, 4),
            "calibration_intercept": round(self.calibration.intercept, 4),
            "net_benefit_0.20": round(self.net_benefit_20, 4),
            "net_benefit_0.30": round(self.net_benefit_30, 4),
            "events": self.events,
            "censored": self.censored,
        }


@dataclass(frozen=True, slots=True)
class OutcomeTable:
    partition: str
    rows: tuple[OutcomeRow, ...]
    cut: float
    reclassification: Reclassification | None

    def as_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "partition": self.partition,
            "reclassification_cut": self.cut,
            "rows": [row.as_mapping() for row in self.rows],
        }
        if self.reclassification is not None:
            payload["reclassification"] = self.reclassification.as_mapping()
        return payload

    def row(self, method: str) -> OutcomeRow:
        for entry in self.rows:
            if entry.method == method:
                return entry
        raise KeyError(f"unknown method {method!r}")


def outcome_scores(
    context: StudyContext,
    result: ArmResult,
    outcomes: OutcomeArrays,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    risk = np.asarray([record.risk for record in result.ensemble], dtype=np.float64)
    mask = np.asarray([record.emitted for record in result.ensemble], dtype=bool)
    del context
    del outcomes
    return (risk, mask)


def c_index_of(risk: npt.NDArray[np.float64], outcomes: OutcomeArrays, mask: npt.NDArray[np.bool_]) -> float:
    return float(
        concordance_index(
            outcomes.times[mask],
            outcomes.events[mask],
            risk[mask],
            CensoringRule(),
            outcomes.treatment_start_days[mask],
        )
    )


def bootstrap_c_index_interval(
    context: StudyContext,
    rows: npt.NDArray[np.int64],
    masks: npt.NDArray[np.bool_],
    risk: npt.NDArray[np.float64],
    outcomes: OutcomeArrays,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    kept = rows[masks]
    patients = context.patient_ids(kept)
    positions = np.flatnonzero(masks).astype(np.int64)

    def statistic(drawn: npt.NDArray[np.int64]) -> float:
        selection = positions[drawn]
        return float(
            concordance_index(
                outcomes.times[selection],
                outcomes.events[selection],
                risk[selection],
                CensoringRule(),
                outcomes.treatment_start_days[selection],
            )
        )

    return bootstrap_interval(patients, statistic, replicates, seed)


def evaluate_outcome_arm(
    context: StudyContext,
    arm: str,
    method: str,
    policy: Policy,
    training: dict[str, TrainingResult],
    rows: npt.NDArray[np.int64],
    outcomes: OutcomeArrays,
    replicates: int,
    seed: int,
) -> tuple[ArmResult, OutcomeRow]:
    result = arm_result(context, arm, policy, rows, training)
    risk, mask = outcome_scores(context, result, outcomes)
    point = c_index_of(risk, outcomes, mask)
    low, high = bootstrap_c_index_interval(context, rows, mask, risk, outcomes, replicates, seed)
    outcomes_observed = outcomes.events[mask].astype(np.float64)
    fit = calibration_slope_intercept(risk[mask], outcomes_observed)
    return (
        result,
        OutcomeRow(
            method=method,
            c_index=point,
            c_index_low=float(low),
            c_index_high=float(high),
            calibration=fit,
            net_benefit_20=float(net_benefit(risk[mask], outcomes_observed, NET_BENEFIT_THRESHOLDS[0])),
            net_benefit_30=float(net_benefit(risk[mask], outcomes_observed, NET_BENEFIT_THRESHOLDS[1])),
            events=int(outcomes_observed.sum()),
            censored=int(outcomes_observed.size - outcomes_observed.sum()),
        ),
    )


def run_outcome_validation(
    context: StudyContext,
    training: dict[str, TrainingResult],
    specification: tuple[tuple[str, str, Policy], ...],
    comparator_method: str,
    replicates: int,
    seed: int,
) -> tuple[OutcomeTable, dict[str, ArmResult]]:
    rows = context.rows(PARTITION_INTERNAL)
    outcomes = outcome_arrays(context, rows)
    entries: list[OutcomeRow] = []
    results: dict[str, ArmResult] = {}
    comparator_risk: npt.NDArray[np.float64] | None = None
    comparator_mask: npt.NDArray[np.bool_] | None = None
    outcomes_observed: npt.NDArray[np.float64] | None = None
    for arm, method, policy in specification:
        result, row = evaluate_outcome_arm(context, arm, method, policy, training, rows, outcomes, replicates, seed)
        results[method] = result
        entries.append(row)
        if method == comparator_method:
            comparator_risk, comparator_mask = outcome_scores(context, result, outcomes)
            outcomes_observed = outcomes.events[comparator_mask].astype(np.float64)
    reclassification: Reclassification | None = None
    target_result = results.get("OSTEO-CT (constrained)")
    if comparator_risk is not None and comparator_mask is not None and outcomes_observed is not None and target_result is not None:
        target_risk, target_mask = outcome_scores(context, target_result, outcomes)
        shared = target_mask & comparator_mask
        reclassification = reclassify(
            target_risk[shared], comparator_risk[shared], outcomes.events[shared].astype(np.float64), RECLASSIFICATION_CUT
        )
    return (
        OutcomeTable(
            partition=PARTITION_INTERNAL, rows=tuple(entries), cut=RECLASSIFICATION_CUT, reclassification=reclassification
        ),
        results,
    )


def c_index_difference(table: OutcomeTable, first: str, second: str) -> float:
    return float(table.row(first).c_index - table.row(second).c_index)


def over_dispersed(table: OutcomeTable, method: str) -> bool:
    return bool(table.row(method).calibration.slope > 1.0)
