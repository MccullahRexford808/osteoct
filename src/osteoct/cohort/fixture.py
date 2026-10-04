"""A schema-compatible cohort fixture.

The multicentre cohort that supports the reported clinical quantities is held under
institutional data-usage terms and is not distributed. This module supplies a
schema-compatible stand-in with the same field layout so that every mechanism in the
pipeline is executable and checkable on a machine that has no access to the cohort.
The fixture carries three auxiliary columns that the real record layout does not have --
the latent attenuation, the latent height loss and the dissociation flag -- so that the
recovery properties of the measurement stages can be checked against a known value. No
manuscript table value is produced from this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.acquisition.harmonise import HarmonisationModel
from osteoct.acquisition.scanner import kernel_offset_hu, scanner_family
from osteoct.cohort.cohort import EXTERNAL_SITE_INDEX, PARTITIONS, Cohort, CohortRecord
from osteoct.localisation.morphology import MorphometryParams
from osteoct.localisation.vertebra import LocalisationParams
from osteoct.schema.records import AcquisitionDescriptor
from osteoct.schema.vocabulary import DiagnosticContext

BODY_X_LOW = 7
BODY_X_HIGH = 17
BODY_Y_LOW = 4
BODY_Y_HIGH = 13
FRACTURE_DEPTH_FRACTION = 0.66
POSTERIOR_Y_LOW = 17
POSTERIOR_Y_HIGH = 22
SOFT_TISSUE_HU = 30.0
CORTICAL_SHELL_HU = 520.0
POSTERIOR_ELEMENT_HU = 640.0
SITE_FAMILIES: tuple[str, ...] = ("vendor_a", "vendor_b", "vendor_c")


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    sites: tuple[str, ...] = ("S1", "S2", "S3")
    regions: tuple[str, ...] = ("R1", "R2")
    examinations_per_site: int = 48
    examinations_per_patient: float = 1.4
    bodies: int = 13
    slices_per_body: int = 10
    slice_gap: int = 2
    rows: int = 24
    columns: int = 24
    development_share: float = 0.68
    attenuation_mean_hu: float = 190.0
    attenuation_sd_hu: float = 42.0
    attenuation_clip_hu: tuple[float, float] = (112.0, 300.0)
    texture_grain_hu: float = 11.0
    height_loss_scale: float = 0.16
    height_loss_clip: tuple[float, float] = (0.0, 0.44)
    dissociation_share: float = 0.35
    dissociation_grade_floor: float = 0.28
    dissociation_attenuation_shift_hu: float = 46.0
    dissociation_hazard_ratio: float = 2.6
    baseline_five_year_incidence: float = 0.05
    hazard_attenuation_slope: float = 0.011
    hazard_height_slope: float = 3.1
    prior_fracture_hazard_ratio: float = 1.7
    treatment_hazard_ratio: float = 0.82
    followup_days: float = 1825.0
    first_event_fraction: float = 0.62
    reference_density_share: float = 0.44
    prior_fracture_share: float = 0.17
    treatment_exposure_share: float = 0.23
    glucocorticoid_share: float = 0.08
    age_mean: float = 64.0
    age_sd: float = 11.0
    bmi_mean: float = 25.4
    bmi_sd: float = 4.1
    thickness_mm: tuple[float, ...] = (0.8, 1.0, 1.25, 2.0)
    seed: int = 20260928

    def depth(self) -> int:
        return self.bodies * (self.slices_per_body + self.slice_gap)


@dataclass(frozen=True, slots=True)
class FixtureRecord(CohortRecord):
    latent_attenuation_hu: float = 0.0
    latent_height_loss: float = 0.0
    dissociated: bool = False


@dataclass(frozen=True, slots=True)
class CohortFixture:
    spec: FixtureSpec
    cohort: Cohort
    localisation: LocalisationParams
    morphometry: MorphometryParams
    harmonisation: HarmonisationModel

    def volume_for(self, record: FixtureRecord) -> npt.NDArray[np.float32]:
        if not isinstance(record, FixtureRecord):
            raise TypeError("the fixture builds volumes only for fixture records")
        return construct_volume(self.spec, record)

    def volumes(self) -> tuple[npt.NDArray[np.float32], ...]:
        return tuple(self.volume_for(record) for record in self.records())

    def records(self) -> tuple[FixtureRecord, ...]:
        return tuple(record for record in self.cohort.records if isinstance(record, FixtureRecord))


def spec_family_for_site(site_index: int) -> str:
    return SITE_FAMILIES[site_index % len(SITE_FAMILIES)]


def build_cohort_fixture(
    spec: FixtureSpec | None = None,
    localisation: LocalisationParams | None = None,
    morphometry: MorphometryParams | None = None,
    harmonisation: HarmonisationModel | None = None,
) -> CohortFixture:
    resolved = FixtureSpec() if spec is None else spec
    rng = np.random.default_rng(resolved.seed)
    records: list[FixtureRecord] = []
    for site_index, site in enumerate(resolved.sites):
        region = resolved.regions[site_index % len(resolved.regions)]
        patients = _patient_ids(resolved, site_index)
        for index in range(resolved.examinations_per_site):
            patient_id = patients[int(rng.integers(0, len(patients)))]
            records.append(_draw_record(resolved, rng, f"{site}-{index:05d}", patient_id, site, region, site_index))
    return CohortFixture(
        spec=resolved,
        cohort=Cohort(tuple(records)),
        localisation=LocalisationParams(bone_window_low_hu=55.0) if localisation is None else localisation,
        morphometry=MorphometryParams() if morphometry is None else morphometry,
        harmonisation=HarmonisationModel(0.42, 120.0, 0.9, 2.2, 1.0) if harmonisation is None else harmonisation,
    )


def _patient_ids(spec: FixtureSpec, site_index: int) -> tuple[str, ...]:
    patients = max(1, int(round(spec.examinations_per_site / spec.examinations_per_patient)))
    return tuple(f"{spec.sites[site_index]}-P{index:04d}" for index in range(patients))


def _acquisition(spec: FixtureSpec, site_index: int, rng: np.random.Generator) -> AcquisitionDescriptor:
    family = scanner_family(spec_family_for_site(site_index))
    kernel = family.kernels[int(rng.integers(0, len(family.kernels)))]
    thickness = float(spec.thickness_mm[int(rng.integers(0, len(spec.thickness_mm)))])
    voltage = family.nominal_voltage_kvp + float(rng.choice([-20.0, 0.0, 20.0]))
    return AcquisitionDescriptor(
        tube_voltage_kvp=voltage,
        scanner_family=family.name,
        reconstruction_kernel=kernel,
        slice_thickness_mm=thickness,
    )


def _partition_of(spec: FixtureSpec, site_index: int, draw: float) -> str:
    if site_index >= EXTERNAL_SITE_INDEX:
        return PARTITIONS[2]
    return PARTITIONS[0] if draw < spec.development_share else PARTITIONS[1]


def _draw_record(
    spec: FixtureSpec,
    rng: np.random.Generator,
    record_id: str,
    patient_id: str,
    site: str,
    region: str,
    site_index: int,
) -> FixtureRecord:
    prior_fractility = bool(rng.random() < spec.prior_fracture_share)
    treatment_exposure = bool(rng.random() < spec.treatment_exposure_share)
    dissociated = bool(rng.random() < spec.dissociation_share)
    latent = float(np.clip(rng.normal(spec.attenuation_mean_hu, spec.attenuation_sd_hu), *spec.attenuation_clip_hu))
    if dissociated:
        latent = float(np.clip(latent + spec.dissociation_attenuation_shift_hu, *spec.attenuation_clip_hu))
    height_loss = float(np.clip(abs(rng.normal(0.0, spec.height_loss_scale)), *spec.height_loss_clip))
    if dissociated:
        height_loss = float(min(spec.height_loss_clip[1], max(height_loss, spec.dissociation_grade_floor)))
    reference_t_score = None
    if rng.random() < spec.reference_density_share:
        reference_t_score = float((latent - spec.attenuation_mean_hu) / spec.attenuation_sd_hu)
    event, observed, censored = _draw_followup(spec, rng, latent, height_loss, prior_fractility, treatment_exposure, dissociated)
    treatment_start = float(60.0 + 1500.0 * float(rng.random())) if treatment_exposure else None
    return FixtureRecord(
        record_id=record_id,
        patient_id=patient_id,
        site=site,
        region=region,
        partition=_partition_of(spec, site_index, float(rng.random())),
        acquisition=_acquisition(spec, site_index, rng),
        context=DiagnosticContext(
            prior_fragility_fracture=prior_fractility,
            treatment_exposure=treatment_exposure,
            reference_t_score=reference_t_score,
        ),
        prior_fractility=prior_fractility,
        fracture_time_days=event,
        observation_time_days=observed,
        censored=censored,
        treatment_start_days=treatment_start,
        age_years=float(np.clip(rng.normal(spec.age_mean, spec.age_sd), 22.0, 96.0)),
        female=bool(rng.random() < 0.58),
        body_mass_index=float(np.clip(rng.normal(spec.bmi_mean, spec.bmi_sd), 14.0, 46.0)),
        glucocorticoid_exposure=bool(rng.random() < spec.glucocorticoid_share),
        latent_attenuation_hu=latent,
        latent_height_loss=height_loss,
        dissociated=dissociated,
    )


def _draw_followup(
    spec: FixtureSpec,
    rng: np.random.Generator,
    latent: float,
    height_loss: float,
    prior_fractility: bool,
    treatment_exposure: bool,
    dissociated: bool,
) -> tuple[float | None, float, bool]:
    attenuation_term = spec.hazard_attenuation_slope * (spec.attenuation_mean_hu - latent)
    hazard = spec.baseline_five_year_incidence * float(np.exp(attenuation_term + spec.hazard_height_slope * height_loss))
    if prior_fractility:
        hazard *= spec.prior_fracture_hazard_ratio
    if dissociated:
        hazard *= spec.dissociation_hazard_ratio
    if treatment_exposure:
        hazard *= spec.treatment_hazard_ratio
    hazard = float(min(0.75, max(1e-4, hazard)))
    five_year_probability = 1.0 - float(np.exp(-hazard * 5.0))
    if rng.random() < five_year_probability:
        if rng.random() < spec.first_event_fraction:
            return (float(30.0 + 1100.0 * float(rng.random())), spec.followup_days, False)
        return (float(1200.0 + 600.0 * float(rng.random())), spec.followup_days, False)
    return (None, spec.followup_days, True)


def volume_seed(record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode("utf-8")).digest()[:4], "big")


def construct_volume(spec: FixtureSpec, record: FixtureRecord) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(volume_seed(record.record_id))
    volume = np.full((spec.rows, spec.columns, spec.depth()), SOFT_TISSUE_HU, dtype=np.float64)
    family = scanner_family(record.acquisition.scanner_family)
    kernel_term = kernel_offset_hu(family, record.acquisition.reconstruction_kernel)
    voltage_term = 0.42 * (record.acquisition.tube_voltage_kvp - 120.0)
    observed = family.attenuation_scale * (record.latent_attenuation_hu + voltage_term + kernel_term) + family.attenuation_offset_hu
    removed = _reduced_slices(spec, record.latent_height_loss)
    step = spec.slices_per_body + spec.slice_gap
    for body in range(spec.bodies):
        start = body * step
        stop = start + spec.slices_per_body
        _stamp_body(volume, spec, start, stop, removed, observed, rng)
        _stamp_posterior(volume, spec, start, stop)
    jitter = rng.normal(0.0, 1.4, size=volume.shape)
    return np.asarray(volume + jitter, dtype=np.float32)


def _reduced_slices(spec: FixtureSpec, height_loss: float) -> int:
    return int(round(height_loss * spec.slices_per_body))


def _fracture_rows(spec: FixtureSpec) -> tuple[int, int]:
    depth = BODY_Y_HIGH - BODY_Y_LOW
    return (BODY_Y_LOW, BODY_Y_LOW + max(1, int(round(depth * FRACTURE_DEPTH_FRACTION))))


def _stamp_body(
    volume: npt.NDArray[np.float64],
    spec: FixtureSpec,
    start: int,
    stop: int,
    removed: int,
    observed: float,
    rng: np.random.Generator,
) -> None:
    grain = rng.normal(0.0, spec.texture_grain_hu, size=(BODY_X_HIGH - BODY_X_LOW, BODY_Y_HIGH - BODY_Y_LOW, stop - start))
    volume[BODY_X_LOW:BODY_X_HIGH, BODY_Y_LOW:BODY_Y_HIGH, start:stop] = observed + grain
    _stamp_cortical(volume, start, stop)
    rows = _fracture_rows(spec)
    if removed <= 0 or removed > stop - start - 4:
        return
    volume[BODY_X_LOW:BODY_X_HIGH, rows[0] : rows[1], start : start + removed] = SOFT_TISSUE_HU


def _stamp_cortical(volume: npt.NDArray[np.float64], start: int, stop: int) -> None:
    volume[BODY_X_LOW, BODY_Y_LOW:BODY_Y_HIGH, start:stop] = CORTICAL_SHELL_HU
    volume[BODY_X_HIGH - 1, BODY_Y_LOW:BODY_Y_HIGH, start:stop] = CORTICAL_SHELL_HU
    volume[BODY_X_LOW:BODY_X_HIGH, BODY_Y_LOW, start:stop] = CORTICAL_SHELL_HU
    volume[BODY_X_LOW:BODY_X_HIGH, BODY_Y_HIGH - 1, start:stop] = CORTICAL_SHELL_HU
    volume[BODY_X_LOW:BODY_X_HIGH, BODY_Y_LOW:BODY_Y_HIGH, start] = CORTICAL_SHELL_HU
    volume[BODY_X_LOW:BODY_X_HIGH, BODY_Y_LOW:BODY_Y_HIGH, stop - 1] = CORTICAL_SHELL_HU


def _stamp_posterior(volume: npt.NDArray[np.float64], spec: FixtureSpec, start: int, stop: int) -> None:
    volume[BODY_X_LOW + 2 : BODY_X_HIGH - 2, POSTERIOR_Y_LOW:POSTERIOR_Y_HIGH, start:stop] = POSTERIOR_ELEMENT_HU
