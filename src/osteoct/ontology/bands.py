"""ISCD official positions and WHO T-score bands expressed on the harmonised interval."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.vocabulary import BAND_ORDER, DiagnosticBand, DiagnosticContext

OSTEOPENIA_UPPER_T = -1.0
OSTEOPOROSIS_UPPER_T = -2.5


@dataclass(frozen=True, slots=True)
class BandCalibration:
    reference_mean_hu: float
    reference_sd_hu: float
    within_scan_floor_hu: float

    def t_score(self, standard_attenuation_hu: float) -> float:
        return (standard_attenuation_hu - self.reference_mean_hu) / self.reference_sd_hu

    def hu_at_t(self, t_score: float) -> float:
        return self.reference_mean_hu + t_score * self.reference_sd_hu

    def thresholds_hu(self) -> tuple[float, float]:
        return (self.hu_at_t(OSTEOPOROSIS_UPPER_T), self.hu_at_t(OSTEOPENIA_UPPER_T))

    def recomposed_uncertainty(self, harmonisation_uncertainty_hu: float) -> float:
        return max(harmonisation_uncertainty_hu, self.within_scan_floor_hu)


def band_from_t_score(t_score: float, prior_fragility_fracture: bool = False) -> DiagnosticBand:
    if t_score > OSTEOPENIA_UPPER_T:
        return DiagnosticBand.NORMAL
    if t_score > OSTEOPOROSIS_UPPER_T:
        return DiagnosticBand.OSTEOPENIA
    if prior_fragility_fracture:
        return DiagnosticBand.SEVERE_OSTEOPOROSIS
    return DiagnosticBand.OSTEOPOROSIS


def band_bounds_hu(calibration: BandCalibration) -> dict[DiagnosticBand, tuple[float, float]]:
    osteoporosis_threshold, osteopenia_threshold = calibration.thresholds_hu()
    return {
        DiagnosticBand.NORMAL: (osteopenia_threshold, float("inf")),
        DiagnosticBand.OSTEOPENIA: (osteoporosis_threshold, osteopenia_threshold),
        DiagnosticBand.OSTEOPOROSIS: (float("-inf"), osteoporosis_threshold),
        DiagnosticBand.SEVERE_OSTEOPOROSIS: (float("-inf"), osteoporosis_threshold),
    }


def bands_from_interval(
    standard_attenuation_hu: float, uncertainty_hu: float, calibration: BandCalibration
) -> tuple[DiagnosticBand, ...]:
    lower = standard_attenuation_hu - uncertainty_hu
    upper = standard_attenuation_hu + uncertainty_hu
    compatible = [band for band in BAND_ORDER if _interval_overlaps_band(lower, upper, band, calibration)]
    if DiagnosticBand.OSTEOPOROSIS in compatible and DiagnosticBand.SEVERE_OSTEOPOROSIS not in compatible:
        compatible.append(DiagnosticBand.SEVERE_OSTEOPOROSIS)
    return tuple(compatible)


def _interval_overlaps_band(lower: float, upper: float, band: DiagnosticBand, calibration: BandCalibration) -> bool:
    band_lower, band_upper = band_bounds_hu(calibration)[band]
    return not (upper < band_lower or lower > band_upper)


def band_from_interval(standard_attenuation_hu: float, uncertainty_hu: float, calibration: BandCalibration) -> DiagnosticBand:
    t_score = calibration.t_score(standard_attenuation_hu)
    return band_from_t_score(t_score)


def interval_straddles_threshold(standard_attenuation_hu: float, uncertainty_hu: float, calibration: BandCalibration) -> bool:
    return len(bands_from_interval(standard_attenuation_hu, uncertainty_hu, calibration)) > 1


def band_consistent(
    candidate_band: DiagnosticBand,
    standard_attenuation_hu: float,
    uncertainty_hu: float,
    calibration: BandCalibration,
    context: DiagnosticContext,
) -> bool:
    compatible = bands_from_interval(standard_attenuation_hu, uncertainty_hu, calibration)
    if candidate_band not in compatible:
        return False
    if not context.reference_density_available():
        return True
    reference_score = context.reference_t_score
    assert reference_score is not None
    reference_band = band_from_t_score(reference_score, context.prior_fragility_fracture)
    return candidate_band in _reconciled(reference_band, compatible)


def _reconciled(reference_band: DiagnosticBand, compatible: tuple[DiagnosticBand, ...]) -> tuple[DiagnosticBand, ...]:
    if reference_band in compatible:
        return (reference_band,)
    return compatible


def band_rank(band: DiagnosticBand) -> int:
    return BAND_ORDER.index(band)
