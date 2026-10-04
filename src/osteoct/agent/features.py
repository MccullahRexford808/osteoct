"""Encoded cohort: measurement bundles, descriptor matrices and the scorer batch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.agent.planner import Planner
from osteoct.agent.tools import Toolbox, ToolRegistry
from osteoct.cohort.cohort import CohortRecord
from osteoct.grounding.standard import Grounding
from osteoct.ontology.axioms import AxiomSet
from osteoct.ontology.bands import band_from_t_score
from osteoct.ontology.index import INDEX_ENTRIES
from osteoct.ontology.severity import grade_from_vertebrae
from osteoct.ontology.tiers import tier_from_band
from osteoct.schema.records import MeasurementBundle
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, RiskTier
from osteoct.texture.descriptors import descriptor_names

BAND_ORDER_INDEX: dict[DiagnosticBand, int] = {
    DiagnosticBand.NORMAL: 0,
    DiagnosticBand.OSTEOPENIA: 1,
    DiagnosticBand.OSTEOPOROSIS: 2,
    DiagnosticBand.SEVERE_OSTEOPOROSIS: 3,
}

TIER_ORDER_INDEX: dict[RiskTier, int] = {
    RiskTier.LOW: 0,
    RiskTier.MODERATE: 1,
    RiskTier.HIGH: 2,
    RiskTier.VERY_HIGH: 3,
}


@dataclass(frozen=True, slots=True)
class EncodedCohort:
    records: tuple[CohortRecord, ...]
    bundles: tuple[MeasurementBundle, ...]
    groundings: tuple[Grounding, ...]
    usable: npt.NDArray[np.bool_]
    volumes: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.float64]
    texture: npt.NDArray[np.float64]
    concepts: npt.NDArray[np.float64]
    context: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    band_target: npt.NDArray[np.int64]
    tier_target: npt.NDArray[np.int64]
    severity_target: npt.NDArray[np.int64]
    risk_target: npt.NDArray[np.float64]
    observed_band: tuple[DiagnosticBand, ...]
    graded_severity: tuple[DeformityGrade, ...]

    def size(self) -> int:
        return len(self.records)

    def usable_records(self) -> tuple[CohortRecord, ...]:
        return tuple(record for record, keep in zip(self.records, self.usable.tolist(), strict=True) if keep)

    def texture_names(self) -> tuple[str, ...]:
        return descriptor_names()


def measurement_concept_vocabulary() -> tuple[str, ...]:
    return tuple(f"{entry.system.value}:{entry.key}" for entry in INDEX_ENTRIES)


def concept_vector(grounding: Grounding, vocabulary: tuple[str, ...]) -> npt.NDArray[np.float64]:
    position = {key: index for index, key in enumerate(vocabulary)}
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for key in grounding.concepts:
        slot = position.get(key)
        if slot is not None:
            vector[slot] = 1.0
    return vector


def band_target_of(
    record: CohortRecord,
    attributes: AxiomSet,
    standard_attenuation_hu: float,
) -> DiagnosticBand:
    reference = record.context.reference_t_score
    if reference is not None:
        return band_from_t_score(reference, record.prior_fractility)
    return band_from_t_score(attributes.calibration.t_score(standard_attenuation_hu), record.prior_fractility)


def encode_cohort(
    records: tuple[CohortRecord, ...],
    volumes: npt.NDArray[np.float32],
    toolbox: Toolbox,
    planner: Planner,
    attributes: AxiomSet,
    registry: ToolRegistry,
) -> EncodedCohort:
    vocabulary = measurement_concept_vocabulary()
    texture_names = descriptor_names()
    size = len(records)
    usable = np.zeros(size, dtype=bool)
    bundles: list[MeasurementBundle] = []
    groundings: list[Grounding] = []
    descriptors = np.zeros((size, 3), dtype=np.float64)
    texture = np.zeros((size, len(texture_names)), dtype=np.float64)
    concepts = np.zeros((size, len(vocabulary)), dtype=np.float64)
    context = np.zeros((size, 2), dtype=np.float64)
    band_target = np.zeros(size, dtype=np.int64)
    tier_target = np.zeros(size, dtype=np.int64)
    severity_target = np.zeros(size, dtype=np.int64)
    risk_target = np.zeros(size, dtype=np.float64)
    observed_band: list[DiagnosticBand | None] = [None] * size
    graded_severity: list[DeformityGrade] = [DeformityGrade.NONE] * size
    embeds: list[npt.NDArray[np.float64]] = []
    for index, record in enumerate(records):
        volume = np.asarray(volumes[index], dtype=np.float32)
        regions, morphologies = toolbox.localise(volume, registry)
        if not regions:
            embeds.append(np.zeros((1, planner.bundle.embedding_dim), dtype=np.float64))
            continue
        measurements, pooled_mean, pooled_sd, voxels = toolbox.measure_attenuation(volume, regions, morphologies, registry)
        harmonised = toolbox.harmonise(pooled_mean, pooled_sd, voxels, record.acquisition, registry)
        vector, names = toolbox.extract_texture(volume, regions, registry)
        density = toolbox.density_estimate(
            harmonised.standard_hu,
            harmonised.half_width_hu,
            vector,
            names,
            record.acquisition,
            registry,
        )
        grounding = toolbox.ground_measurements(regions, harmonised.standard_hu, harmonised.half_width_hu, density, registry)
        bundle = MeasurementBundle(
            standard_attenuation_hu=harmonised.standard_hu,
            uncertainty_hu=harmonised.half_width_hu,
            texture_vector=vector,
            density_indicator=density,
            acquisition=record.acquisition,
            vertebrae=measurements,
            retained_texture_names=names,
        )
        band = band_target_of(record, attributes, harmonised.standard_hu)
        tier = tier_from_band(band, record.context, record.acquisition, grounding.concepts, attributes.index, attributes.envelope)
        grade = grade_from_vertebrae(tuple(measurement.morphology for measurement in measurements), attributes.thresholds)
        bundles.append(bundle)
        groundings.append(grounding)
        usable[index] = True
        descriptors[index, 0] = harmonised.standard_hu
        descriptors[index, 1] = harmonised.half_width_hu
        descriptors[index, 2] = density
        for slot, name in enumerate(texture_names):
            texture[index, slot] = dict(zip(names, vector, strict=False)).get(name, 0.0)
        concepts[index, :] = concept_vector(grounding, vocabulary)
        context[index, 0] = 1.0 if record.prior_fractility else 0.0
        context[index, 1] = 1.0 if record.context.treatment_exposure else 0.0
        band_target[index] = BAND_ORDER_INDEX[band]
        tier_target[index] = TIER_ORDER_INDEX[tier]
        severity_target[index] = int(grade.value)
        risk_target[index] = 1.0 if not record.censored else 0.0
        observed_band[index] = band
        graded_severity[index] = grade
        embeds.append(planner.encode_one(volume))
    embedding = np.zeros((size, planner.bundle.embedding_dim), dtype=np.float64)
    for index, block in enumerate(embeds):
        if block.size:
            embedding[index, :] = block.reshape(-1)[: planner.bundle.embedding_dim]
    return EncodedCohort(
        records=records,
        bundles=tuple(bundles),
        groundings=tuple(groundings),
        usable=usable,
        volumes=np.asarray(volumes, dtype=np.float32),
        descriptors=descriptors,
        texture=texture,
        concepts=concepts,
        context=context,
        embedding=embedding,
        band_target=band_target,
        tier_target=tier_target,
        severity_target=severity_target,
        risk_target=risk_target,
        observed_band=tuple(band if band is not None else DiagnosticBand.NORMAL for band in observed_band),
        graded_severity=tuple(graded_severity),
    )


def select_rows(encoded: EncodedCohort, rows: npt.NDArray[np.int64]) -> EncodedCohort:
    keep = rows
    return EncodedCohort(
        records=tuple(encoded.records[int(row)] for row in keep.tolist()),
        bundles=tuple(encoded.bundles[int(row)] for row in keep.tolist()),
        groundings=tuple(encoded.groundings[int(row)] for row in keep.tolist()),
        usable=encoded.usable[keep],
        volumes=encoded.volumes[keep],
        descriptors=encoded.descriptors[keep],
        texture=encoded.texture[keep],
        concepts=encoded.concepts[keep],
        context=encoded.context[keep],
        embedding=encoded.embedding[keep],
        band_target=encoded.band_target[keep],
        tier_target=encoded.tier_target[keep],
        severity_target=encoded.severity_target[keep],
        risk_target=encoded.risk_target[keep],
        observed_band=tuple(encoded.observed_band[int(row)] for row in keep.tolist()),
        graded_severity=tuple(encoded.graded_severity[int(row)] for row in keep.tolist()),
    )


def rows_for_partition(encoded: EncodedCohort, partition: str) -> npt.NDArray[np.int64]:
    selected = [
        index for index, record in enumerate(encoded.records) if record.partition == partition and bool(encoded.usable[index])
    ]
    return np.asarray(selected, dtype=np.int64)


def encode_report(encoded: EncodedCohort) -> dict[str, object]:
    band_counts: dict[str, int] = {}
    for band in encoded.observed_band:
        band_counts[band.value] = band_counts.get(band.value, 0) + 1
    grade_counts: dict[str, int] = {}
    for grade in encoded.graded_severity:
        grade_counts[str(int(grade.value))] = grade_counts.get(str(int(grade.value)), 0) + 1
    return {
        "records": encoded.size(),
        "usable": int(encoded.usable.sum()),
        "texture_width": int(encoded.texture.shape[1]),
        "concept_width": int(encoded.concepts.shape[1]),
        "embedding_width": int(encoded.embedding.shape[1]),
        "band_counts": band_counts,
        "grade_counts": grade_counts,
        "observed_events": int(encoded.risk_target.sum()),
    }
