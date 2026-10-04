"""Modality-comparable baseline scorers reported beside the constrained agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.ontology.bands import BandCalibration, band_from_t_score
from osteoct.ontology.severity import GenantThresholds, grade_from_vertebrae
from osteoct.schema.records import MeasurementBundle
from osteoct.schema.vocabulary import BAND_ORDER, DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier


@dataclass(frozen=True, slots=True)
class ClinicalVariables:
    age_years: float
    female: bool
    body_mass_index: float
    prior_fragility_fracture: bool
    treatment_exposure: bool
    glucocorticoid_exposure: bool

    def as_vector(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                self.age_years,
                1.0 if self.female else 0.0,
                self.body_mass_index,
                1.0 if self.prior_fragility_fracture else 0.0,
                1.0 if self.treatment_exposure else 0.0,
                1.0 if self.glucocorticoid_exposure else 0.0,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ClinicalRiskModel:
    coefficients: npt.NDArray[np.float64]
    intercept: float

    def probability(self, variables: ClinicalVariables) -> float:
        linear = float(self.intercept + np.dot(self.coefficients, variables.as_vector()))
        return float(1.0 / (1.0 + np.exp(-linear)))

    def band(self, variables: ClinicalVariables, calibration: BandCalibration) -> DiagnosticBand:
        probability = self.probability(variables)
        t_equivalent = _t_from_probability(probability)
        return band_from_t_score(t_equivalent, variables.prior_fragility_fracture)


def _t_from_probability(probability: float) -> float:
    return -2.5 - 6.0 * (probability - 0.5)


@dataclass(frozen=True, slots=True)
class HuThresholdRule:
    calibration: BandCalibration
    thresholds: GenantThresholds
    lower_band_hu: float = 110.0

    def band(self, standard_attenuation_hu: float) -> DiagnosticBand:
        t_score = self.calibration.t_score(standard_attenuation_hu)
        return band_from_t_score(t_score)

    def tier(self, band: DiagnosticBand, context: DiagnosticContext) -> RiskTier:
        prior_fracture, treatment_exposure = context.as_flags()
        base = {
            DiagnosticBand.NORMAL: RiskTier.LOW,
            DiagnosticBand.OSTEOPENIA: RiskTier.MODERATE,
            DiagnosticBand.OSTEOPOROSIS: RiskTier.HIGH,
            DiagnosticBand.SEVERE_OSTEOPOROSIS: RiskTier.VERY_HIGH,
        }[band]
        if prior_fracture:
            return RiskTier.VERY_HIGH
        if treatment_exposure:
            return RiskTier.HIGH if base is RiskTier.MODERATE else base
        return base

    def severity(self, bundle: MeasurementBundle) -> DeformityGrade:
        return grade_from_vertebrae(tuple(measurement.morphology for measurement in bundle.vertebrae), self.thresholds)


def band_index(band: DiagnosticBand) -> int:
    return BAND_ORDER.index(band)


def band_one_hot(band: DiagnosticBand) -> npt.NDArray[np.float64]:
    vector = np.zeros(len(BAND_ORDER), dtype=np.float64)
    vector[band_index(band)] = 1.0
    return vector


def severity_ordinal(grade: DeformityGrade) -> int:
    return int(grade.value)


def risk_probability_from_band(band: DiagnosticBand) -> float:
    return {
        DiagnosticBand.NORMAL: 0.04,
        DiagnosticBand.OSTEOPENIA: 0.09,
        DiagnosticBand.OSTEOPOROSIS: 0.20,
        DiagnosticBand.SEVERE_OSTEOPOROSIS: 0.29,
    }[band]
