"""Deformity and densitometric status: the dissociation and its prognostic weight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.evaluation.family_ablation import event_time_of
from osteoct.evaluation.study import StudyContext
from osteoct.metrics.survival import five_year_incidence, incidence_ratio
from osteoct.ontology.bands import band_from_t_score
from osteoct.ontology.severity import grade_from_vertebrae
from osteoct.schema.vocabulary import BAND_ORDER, DeformityGrade, DiagnosticBand

DEFORMITY_FLOOR = DeformityGrade.MODERATE


@dataclass(frozen=True, slots=True)
class DissociationStratum:
    label: str
    count: int
    events: int
    five_year_incidence: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "patients": self.count,
            "events": self.events,
            "five_year_incidence": round(self.five_year_incidence, 4),
        }


@dataclass(frozen=True, slots=True)
class DissociationStudy:
    patients: int
    deformed: int
    deformed_densitometrically_normal: int
    normal_without_deformity: int
    dissociated_share: float
    strata: tuple[DissociationStratum, ...]
    incidence_ratio: float
    missed_by_density_rule: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "patients": self.patients,
            "deformed": self.deformed,
            "deformed_and_densitometrically_normal": self.deformed_densitometrically_normal,
            "normal_without_deformity": self.normal_without_deformity,
            "dissociated_share": round(self.dissociated_share, 4),
            "incidence_ratio": round(self.incidence_ratio, 4),
            "missed_by_density_only_rule": self.missed_by_density_rule,
            "strata": [stratum.as_mapping() for stratum in self.strata],
        }


def densitometric_band(context: StudyContext, row: int) -> DiagnosticBand:
    bundle = context.encoded.bundles[row]
    return band_from_t_score(context.attributes.calibration.t_score(bundle.standard_attenuation_hu))


def severity_grade(context: StudyContext, row: int) -> DeformityGrade:
    bundle = context.encoded.bundles[row]
    return grade_from_vertebrae(tuple(measurement.morphology for measurement in bundle.vertebrae), context.attributes.thresholds)


def patient_first_rows(context: StudyContext) -> dict[str, int]:
    selected: dict[str, int] = {}
    for row, record in enumerate(context.encoded.records):
        if not bool(context.encoded.usable[row]) or not record.has_followup():
            continue
        selected.setdefault(record.patient_id, row)
    return selected


def run_dissociation(context: StudyContext, horizon_days: float = 1825.0) -> DissociationStudy:
    first_rows = patient_first_rows(context)
    deformed = 0
    deformed_normal = 0
    normal_without = 0
    dissociated_times: list[float] = []
    dissociated_events: list[float] = []
    reference_times: list[float] = []
    reference_events: list[float] = []
    for patient, row in first_rows.items():
        grade = severity_grade(context, row)
        band = densitometric_band(context, row)
        record = context.encoded.records[row]
        time = event_time_of(record)
        event = 0.0 if record.censored else 1.0
        is_deformed = int(grade.value) >= int(DEFORMITY_FLOOR.value)
        is_normal_density = band in (DiagnosticBand.NORMAL, DiagnosticBand.OSTEOPENIA)
        if is_deformed:
            deformed += 1
            if is_normal_density:
                deformed_normal += 1
                dissociated_times.append(time)
                dissociated_events.append(event)
        elif band is DiagnosticBand.NORMAL:
            normal_without += 1
            reference_times.append(time)
            reference_events.append(event)
        del patient
    dissociated_incidence = five_year_incidence(
        np.asarray(dissociated_times, dtype=np.float64),
        np.asarray(dissociated_events, dtype=np.float64) > 0.5,
        horizon_days,
    )
    reference_incidence = five_year_incidence(
        np.asarray(reference_times, dtype=np.float64),
        np.asarray(reference_events, dtype=np.float64) > 0.5,
        horizon_days,
    )
    strata = (
        DissociationStratum(
            "deformed, densitometrically normal or osteopenic",
            len(dissociated_times),
            int(sum(dissociated_events)),
            dissociated_incidence,
        ),
        DissociationStratum(
            "densitometrically normal without deformity",
            len(reference_times),
            int(sum(reference_events)),
            reference_incidence,
        ),
    )
    share = float(deformed_normal / deformed) if deformed else 0.0
    return DissociationStudy(
        patients=len(first_rows),
        deformed=deformed,
        deformed_densitometrically_normal=deformed_normal,
        normal_without_deformity=normal_without,
        dissociated_share=share,
        strata=strata,
        incidence_ratio=incidence_ratio(dissociated_incidence, reference_incidence),
        missed_by_density_rule=deformed_normal,
    )


def density_only_rule_misses(study: DissociationStudy) -> int:
    return int(study.missed_by_density_rule)


def band_rank_assignment(context: StudyContext, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    ranks = np.zeros(rows.size, dtype=np.int64)
    for position, row in enumerate(rows.tolist()):
        ranks[position] = BAND_ORDER.index(densitometric_band(context, row))
    return ranks
