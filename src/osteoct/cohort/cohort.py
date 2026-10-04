"""Cohort records, partitions and the file loader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from osteoct.schema.records import AcquisitionDescriptor, Examination
from osteoct.schema.vocabulary import DiagnosticBand, DiagnosticContext
from osteoct.utils.io import read_json

PARTITIONS: tuple[str, ...] = ("development", "internal_test", "external")
EXTERNAL_SITE_INDEX = 2


class CohortError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CohortRecord:
    record_id: str
    patient_id: str
    site: str
    region: str
    partition: str
    acquisition: AcquisitionDescriptor
    context: DiagnosticContext
    prior_fractility: bool
    fracture_time_days: float | None
    observation_time_days: float
    censored: bool
    age_years: float
    female: bool
    body_mass_index: float
    glucocorticoid_exposure: bool
    treatment_start_days: float | None = None

    def has_followup(self) -> bool:
        return self.fracture_time_days is not None or self.observation_time_days > 0.0

    def event_time(self) -> float:
        return self.fracture_time_days if self.fracture_time_days is not None else self.observation_time_days


@dataclass(frozen=True, slots=True)
class Cohort:
    records: tuple[CohortRecord, ...]

    def size(self) -> int:
        return len(self.records)

    def partition(self, name: str) -> tuple[CohortRecord, ...]:
        if name not in PARTITIONS:
            raise CohortError(f"unknown partition {name!r}")
        return tuple(record for record in self.records if record.partition == name)

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({record.site for record in self.records}))

    def site_table(self) -> dict[str, int]:
        table: dict[str, int] = {}
        for record in self.records:
            table[record.site] = table.get(record.site, 0) + 1
        return table

    def partition_table(self) -> dict[str, int]:
        table: dict[str, int] = {}
        for record in self.records:
            table[record.partition] = table.get(record.partition, 0) + 1
        return table

    def patient_count(self) -> int:
        return len({record.patient_id for record in self.records})

    def with_followup(self) -> tuple[CohortRecord, ...]:
        return tuple(record for record in self.records if record.has_followup())

    def densitometric_band(self, record: CohortRecord, reference_mean_hu: float, reference_sd_hu: float) -> DiagnosticBand:
        if record.context.reference_t_score is not None:
            return _band_of_t_score(record.context.reference_t_score, record.prior_fractility)
        return DiagnosticBand.NORMAL


def _band_of_t_score(t_score: float, prior_fracture: bool) -> DiagnosticBand:
    from osteoct.ontology.bands import OSTEOPENIA_UPPER_T, OSTEOPOROSIS_UPPER_T

    if t_score > OSTEOPENIA_UPPER_T:
        return DiagnosticBand.NORMAL
    if t_score > OSTEOPOROSIS_UPPER_T:
        return DiagnosticBand.OSTEOPENIA
    return DiagnosticBand.SEVERE_OSTEOPOROSIS if prior_fracture else DiagnosticBand.OSTEOPOROSIS


RECORD_FIELDS: tuple[str, ...] = (
    "record_id",
    "patient_id",
    "site",
    "region",
    "partition",
    "tube_voltage_kvp",
    "scanner_family",
    "reconstruction_kernel",
    "slice_thickness_mm",
    "prior_fractility",
    "treatment_exposure",
    "reference_t_score",
    "fracture_time_days",
    "observation_time_days",
    "censored",
    "treatment_start_days",
    "age_years",
    "female",
    "body_mass_index",
    "glucocorticoid_exposure",
)


def record_from_mapping(payload: Mapping[str, Any]) -> CohortRecord:
    missing = [field for field in RECORD_FIELDS if field not in payload]
    if missing:
        raise CohortError("record is missing fields: " + ", ".join(missing))
    context = DiagnosticContext(
        prior_fragility_fracture=bool(payload["prior_fractility"]),
        treatment_exposure=bool(payload["treatment_exposure"]),
        reference_t_score=None if payload["reference_t_score"] is None else float(payload["reference_t_score"]),
    )
    acquisition = AcquisitionDescriptor(
        tube_voltage_kvp=float(payload["tube_voltage_kvp"]),
        scanner_family=str(payload["scanner_family"]),
        reconstruction_kernel=str(payload["reconstruction_kernel"]),
        slice_thickness_mm=float(payload["slice_thickness_mm"]),
    )
    fracture_time = payload["fracture_time_days"]
    treatment_start = payload["treatment_start_days"]
    return CohortRecord(
        record_id=str(payload["record_id"]),
        patient_id=str(payload["patient_id"]),
        site=str(payload["site"]),
        region=str(payload["region"]),
        partition=str(payload["partition"]),
        acquisition=acquisition,
        context=context,
        prior_fractility=bool(payload["prior_fractility"]),
        fracture_time_days=None if fracture_time is None else float(fracture_time),
        observation_time_days=float(payload["observation_time_days"]),
        censored=bool(payload["censored"]),
        treatment_start_days=None if treatment_start is None else float(treatment_start),
        age_years=float(payload["age_years"]),
        female=bool(payload["female"]),
        body_mass_index=float(payload["body_mass_index"]),
        glucocorticoid_exposure=bool(payload["glucocorticoid_exposure"]),
    )


def load_cohort(path: Path) -> Cohort:
    if not path.exists():
        raise CohortError(f"cohort records file not found: {path.name}")
    payload = read_json(path)
    if isinstance(payload, Mapping):
        raw_records = payload.get("records")
    else:
        raw_records = payload
    if not isinstance(raw_records, list) or not raw_records:
        raise CohortError("the cohort records file carries no records")
    records = tuple(record_from_mapping(entry) for entry in raw_records)
    partitions = {record.partition for record in records}
    unknown = partitions - set(PARTITIONS)
    if unknown:
        raise CohortError("unknown partition labels: " + ", ".join(sorted(unknown)))
    return Cohort(records)


def examination_from_record(record: CohortRecord, volume: npt.NDArray[np.float32]) -> Examination:
    return Examination(
        record_id=record.record_id,
        patient_id=record.patient_id,
        site=record.site,
        region=record.region,
        volume=volume,
        acquisition=record.acquisition,
        context=record.context,
        incident_fracture_time_days=record.fracture_time_days,
        observation_time_days=record.observation_time_days,
        censored=record.censored,
    )
