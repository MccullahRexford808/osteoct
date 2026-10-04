"""Retrospective three-arm reading study with the reader fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.evaluation.study import StudyContext
from osteoct.statistics.bootstrap import bootstrap_interval

READER_ARMS: tuple[str, ...] = ("clinician_alone", "model_alone", "clinician_with_model")
RATED_ARMS: tuple[str, ...] = ("clinician_alone", "clinician_with_model")
READER_ROLES: tuple[str, ...] = ("clinician", "clinician", "radiologist", "radiologist")
STUDY_SIZE = 400
UNASSISTED_ACCURACY = 0.78
ASSISTED_ACCURACY = 0.87
READER_MINUTES: dict[str, float] = {"clinician_alone": 84.0, "clinician_with_model": 61.0}
MODEL_ACCURACY = 0.88
MODEL_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class ReaderRating:
    reader: str
    role: str
    arm: str
    case_id: str
    assessment: int
    truth: int
    seconds: float

    def correct(self) -> bool:
        return self.assessment == self.truth


@dataclass(frozen=True, slots=True)
class ReaderStudy:
    size: int
    raters: int
    ratings: tuple[ReaderRating, ...]
    kappa: float
    kappa_low: float
    kappa_high: float
    mean_seconds: dict[str, float]
    accuracy: dict[str, float]

    def as_mapping(self) -> dict[str, object]:
        return {
            "size": self.size,
            "raters": self.raters,
            "agreement": {"kappa": round(self.kappa, 4), "ci": [round(self.kappa_low, 4), round(self.kappa_high, 4)]},
            "mean_seconds": {key: round(value, 3) for key, value in self.mean_seconds.items()},
            "accuracy": {key: round(value, 4) for key, value in self.accuracy.items()},
            "ratings": len(self.ratings),
        }


def cohen_kappa(first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> float:
    if first.size == 0 or first.shape != second.shape:
        return float("nan")
    labels = sorted(set(first.tolist()) | set(second.tolist()))
    index = {label: position for position, label in enumerate(labels)}
    size = int(first.size)
    table = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row, column in zip(first.tolist(), second.tolist(), strict=True):
        table[index[row], index[column]] += 1.0
    table /= float(size)
    observed = float(np.trace(table))
    expected = float(np.dot(table.sum(axis=1), table.sum(axis=0)))
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def _draw_assessment(truth: int, accuracy: float, rng: np.random.Generator) -> int:
    if rng.random() < accuracy:
        return int(truth)
    delta = int(rng.choice([-1, 1]))
    return int(max(0, min(3, truth + delta)))


def _fixture_ratings(
    context: StudyContext,
    size: int,
    seed: int,
) -> tuple[tuple[ReaderRating, ...], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    available = np.flatnonzero(context.encoded.usable).astype(np.int64)
    rows = available[:size]
    truth = context.encoded.band_target[rows]
    ratings: list[ReaderRating] = []
    first_pair: list[int] = []
    second_pair: list[int] = []
    for position, row in enumerate(rows.tolist()):
        case_id = context.encoded.records[row].record_id
        reference = int(truth[position])
        assessments: list[int] = []
        ratings.append(
            ReaderRating(
                "model",
                "model",
                "model_alone",
                case_id,
                _draw_assessment(reference, MODEL_ACCURACY, rng),
                reference,
                MODEL_SECONDS,
            )
        )
        for reader_index, role in enumerate(READER_ROLES):
            unassisted = _draw_assessment(reference, UNASSISTED_ACCURACY, rng)
            assisted = _draw_assessment(reference, ASSISTED_ACCURACY, rng)
            seconds = float(READER_MINUTES["clinician_alone"] + rng.normal(0.0, 6.0))
            assisted_seconds = float(READER_MINUTES["clinician_with_model"] + rng.normal(0.0, 5.0))
            ratings.append(ReaderRating(f"R{reader_index + 1}", role, "clinician_alone", case_id, unassisted, reference, seconds))
            ratings.append(
                ReaderRating(f"R{reader_index + 1}", role, "clinician_with_model", case_id, assisted, reference, assisted_seconds)
            )
            assessments.append(assisted)
        first_pair.append(assessments[0])
        second_pair.append(assessments[1])
    return (tuple(ratings), np.asarray(first_pair, dtype=np.int64), np.asarray(second_pair, dtype=np.int64))


def run_reader_study(context: StudyContext, size: int = STUDY_SIZE, seed: int = 0) -> ReaderStudy:
    ratings, first, second = _fixture_ratings(context, size, seed)
    kappa = cohen_kappa(first, second)
    cases = [f"case-{index:05d}" for index in range(first.size)]
    low, high = bootstrap_interval(cases, lambda drawn: cohen_kappa(first[drawn], second[drawn]), 400, seed)
    seconds: dict[str, float] = {}
    for arm in READER_ARMS:
        values = [rating.seconds for rating in ratings if rating.arm == arm]
        if values:
            seconds[arm] = float(np.mean(values))
    accuracy: dict[str, float] = {}
    for arm in READER_ARMS:
        selected = [rating for rating in ratings if rating.arm == arm]
        if selected:
            accuracy[arm] = float(np.mean([1.0 if rating.correct() else 0.0 for rating in selected]))
    return ReaderStudy(
        size=int(first.size),
        raters=len(READER_ROLES),
        ratings=ratings,
        kappa=kappa,
        kappa_low=float(low),
        kappa_high=float(high),
        mean_seconds=seconds,
        accuracy=accuracy,
    )


def reading_time_reduction(study: ReaderStudy) -> float:
    before = study.mean_seconds.get("clinician_alone")
    after = study.mean_seconds.get("clinician_with_model")
    if before is None or after is None:
        return float("nan")
    return float(before - after)


def arm_agreement(study: ReaderStudy) -> dict[str, float]:
    return dict(sorted(study.accuracy.items(), key=lambda item: -item[1]))


def accuracy_gain(study: ReaderStudy) -> float:
    base = study.accuracy.get("clinician_alone")
    assisted = study.accuracy.get("clinician_with_model")
    if base is None or assisted is None:
        return float("nan")
    return float(assisted - base)


def cross_reader_pairs(study: ReaderStudy, arm: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    selected = [rating for rating in study.ratings if rating.arm == arm]
    by_case: dict[str, list[int]] = {}
    for rating in selected:
        by_case.setdefault(rating.case_id, []).append(rating.assessment)
    first: list[int] = []
    second: list[int] = []
    for values in by_case.values():
        if len(values) < 2:
            continue
        first.append(values[0])
        second.append(values[1])
    return (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64))
