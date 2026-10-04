"""Net reclassification improvement against a comparator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Reclassification:
    events_up: int
    events_down: int
    non_events_up: int
    non_events_down: int
    cut: float

    def event_share(self) -> float:
        total = self.events_up + self.events_down
        return 0.0 if total == 0 else (self.events_up - self.events_down) / total

    def non_event_share(self) -> float:
        total = self.non_events_up + self.non_events_down
        return 0.0 if total == 0 else (self.non_events_down - self.non_events_up) / total

    def nri(self) -> float:
        return float(self.event_share() + self.non_event_share())

    def as_mapping(self) -> dict[str, float]:
        return {
            "nri": self.nri(),
            "event_share": self.event_share(),
            "non_event_share": self.non_event_share(),
            "cut": self.cut,
        }


def reclassify(
    new_probabilities: npt.NDArray[np.float64],
    old_probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    cut: float,
) -> Reclassification:
    if new_probabilities.shape != old_probabilities.shape or new_probabilities.shape != outcomes.shape:
        raise ValueError("the two risk vectors and the outcome vector must share a shape")
    new_high = np.asarray(new_probabilities, dtype=np.float64) >= cut
    old_high = np.asarray(old_probabilities, dtype=np.float64) >= cut
    events = np.asarray(outcomes, dtype=np.float64) > 0.5
    non_events = ~events
    return Reclassification(
        events_up=int(np.sum(events & new_high & ~old_high)),
        events_down=int(np.sum(events & ~new_high & old_high)),
        non_events_up=int(np.sum(non_events & new_high & ~old_high)),
        non_events_down=int(np.sum(non_events & ~new_high & old_high)),
        cut=cut,
    )


def continuous_nri(
    new_probabilities: npt.NDArray[np.float64],
    old_probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    difference = np.asarray(new_probabilities, dtype=np.float64) - np.asarray(old_probabilities, dtype=np.float64)
    events = np.asarray(outcomes, dtype=np.float64) > 0.5
    if not bool(events.any()) or bool(events.all()):
        return float("nan")
    event_term = float(np.mean(difference[events]))
    non_event_term = -float(np.mean(difference[~events]))
    return event_term + non_event_term


def nri_interval(
    new_probabilities: npt.NDArray[np.float64],
    old_probabilities: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    cut: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    size = int(new_probabilities.shape[0])
    values = np.zeros(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, size, size=size)
        values[index] = reclassify(new_probabilities[draw], old_probabilities[draw], outcomes[draw], cut).nri()
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def risk_category(probabilities: npt.NDArray[np.float64], cut: float) -> npt.NDArray[np.int64]:
    return (np.asarray(probabilities, dtype=np.float64) >= cut).astype(np.int64)


def moved_share(new_probabilities: npt.NDArray[np.float64], old_probabilities: npt.NDArray[np.float64], cut: float) -> float:
    moved = risk_category(new_probabilities, cut) != risk_category(old_probabilities, cut)
    return float(np.mean(moved)) if moved.size else 0.0
