"""Genant semiquantitative deformity grading from measured vertebral morphology."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.records import MorphologyFeatures
from osteoct.schema.vocabulary import DeformityGrade


@dataclass(frozen=True, slots=True)
class GenantThresholds:
    fracture_floor: float = 0.20
    moderate_floor: float = 0.25
    severe_floor: float = 0.40
    boundary_tolerance: float = 1e-9

    def grade(self, height_loss: float) -> DeformityGrade:
        value = height_loss
        if value < self.fracture_floor - self.boundary_tolerance:
            return DeformityGrade.NONE
        if value < self.moderate_floor - self.boundary_tolerance:
            return DeformityGrade.MILD
        if value < self.severe_floor - self.boundary_tolerance:
            return DeformityGrade.MODERATE
        return DeformityGrade.SEVERE


def grade_from_height_loss(height_loss: float, thresholds: GenantThresholds) -> DeformityGrade:
    return thresholds.grade(max(0.0, min(1.0, height_loss)))


def grade_from_morphology(morphology: MorphologyFeatures, thresholds: GenantThresholds) -> DeformityGrade:
    return grade_from_height_loss(morphology.height_loss(), thresholds)


def grade_from_vertebrae(morphologies: tuple[MorphologyFeatures, ...], thresholds: GenantThresholds) -> DeformityGrade:
    if not morphologies:
        return DeformityGrade.NONE
    return max((grade_from_morphology(morphology, thresholds) for morphology in morphologies), key=lambda grade: int(grade.value))


def is_deformed(morphology: MorphologyFeatures, thresholds: GenantThresholds) -> bool:
    return grade_from_morphology(morphology, thresholds) is not DeformityGrade.NONE


def grade_at_least_two(morphology: MorphologyFeatures, thresholds: GenantThresholds) -> bool:
    return int(grade_from_morphology(morphology, thresholds).value) >= int(DeformityGrade.MODERATE.value)
