"""Concordance for time to first incident fracture with treatment-aware censoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class CensoringRule:
    treatment_grace_days: float = 0.0

    def effective(
        self,
        observation_days: npt.NDArray[np.float64],
        events: npt.NDArray[np.bool_],
        treatment_start_days: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        switch = treatment_start_days + self.treatment_grace_days
        capped = np.minimum(observation_days, np.where(np.isfinite(switch), switch, observation_days))
        truncated = np.where(np.isfinite(switch), observation_days >= switch, False)
        event_kept = np.asarray(events, dtype=bool) & ~truncated
        return (np.asarray(capped, dtype=np.float64), event_kept)


def comparable_pairs(
    times: npt.NDArray[np.float64],
    events: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int64]:
    index = np.arange(times.size, dtype=np.int64)
    rows = index[:, None]
    columns = index[None, :]
    upper = rows < columns
    distinct = times[rows] != times[columns]
    first_earlier = times[rows] < times[columns]
    earlier = np.where(first_earlier, rows, columns)
    later = np.where(first_earlier, columns, rows)
    valid = upper & distinct & np.asarray(events, dtype=bool)[earlier]
    if not bool(valid.any()):
        return np.zeros((0, 2), dtype=np.int64)
    selected_earlier = earlier[valid]
    selected_later = later[valid]
    return np.stack([selected_earlier, selected_later], axis=1).astype(np.int64)


def concordance_index(
    times: npt.NDArray[np.float64],
    events: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    rule: CensoringRule | None = None,
    treatment_start_days: npt.NDArray[np.float64] | None = None,
) -> float:
    resolved_rule = CensoringRule() if rule is None else rule
    treatment = (
        np.full(times.size, np.inf, dtype=np.float64)
        if treatment_start_days is None
        else np.asarray(treatment_start_days, dtype=np.float64)
    )
    effective_times, effective_events = resolved_rule.effective(
        np.asarray(times, dtype=np.float64), np.asarray(events, dtype=bool), treatment
    )
    pairs = comparable_pairs(effective_times, effective_events)
    if pairs.size == 0:
        return float("nan")
    first = pairs[:, 0]
    second = pairs[:, 1]
    ordered = np.asarray(scores, dtype=np.float64)[first] > np.asarray(scores, dtype=np.float64)[second]
    tied = np.asarray(scores, dtype=np.float64)[first] == np.asarray(scores, dtype=np.float64)[second]
    return float((ordered.sum() + 0.5 * tied.sum()) / pairs.shape[0])


def event_count(events: npt.NDArray[np.bool_]) -> int:
    return int(np.asarray(events, dtype=bool).sum())


def censored_count(events: npt.NDArray[np.bool_]) -> int:
    return int(np.asarray(events, dtype=bool).size - np.asarray(events, dtype=bool).sum())


def five_year_incidence(
    times: npt.NDArray[np.float64],
    events: npt.NDArray[np.bool_],
    horizon_days: float = 1825.0,
) -> float:
    at_risk = np.asarray(times, dtype=np.float64) > 0.0
    if not bool(at_risk.any()):
        return 0.0
    with_event = np.asarray(events, dtype=bool) & (np.asarray(times, dtype=np.float64) <= horizon_days)
    return float(with_event.sum() / max(1, int(at_risk.sum())))


def incidence_ratio(first: float, second: float) -> float:
    if second <= 0.0:
        return float("nan")
    return float(first / second)
