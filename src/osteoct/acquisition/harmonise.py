"""Harmonisation of raw attenuation to a standard value with an uncertainty half-width."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.acquisition.scanner import (
    kernel_offset_hu,
    nominal_noise_hu,
    scanner_family,
    voltage_delta_kvp,
)
from osteoct.schema.records import AcquisitionDescriptor


@dataclass(frozen=True, slots=True)
class HarmonisationModel:
    voltage_slope_hu_per_kvp: float
    reference_voltage_kvp: float
    sampling_weight: float
    within_scan_floor_hu: float
    kernel_spread_hu: float

    def corrected_mean(self, mean_hu: float, acquisition: AcquisitionDescriptor) -> float:
        family = scanner_family(acquisition.scanner_family)
        kernel_term = kernel_offset_hu(family, acquisition.reconstruction_kernel) * self.kernel_spread_hu
        corrected = (mean_hu - family.attenuation_offset_hu) / family.attenuation_scale
        corrected -= self.voltage_slope_hu_per_kvp * voltage_delta_kvp(acquisition)
        corrected -= kernel_term
        return corrected

    def uncertainty(
        self,
        sd_hu: float,
        voxels: int,
        acquisition: AcquisitionDescriptor,
    ) -> float:
        family = scanner_family(acquisition.scanner_family)
        sampling = self.sampling_weight * (sd_hu / max(1.0, float(voxels)) ** 0.5)
        voltage = abs(self.voltage_slope_hu_per_kvp * voltage_delta_kvp(acquisition))
        kernel_term = self.kernel_spread_hu * abs(kernel_offset_hu(family, acquisition.reconstruction_kernel))
        noise = 0.5 * nominal_noise_hu(acquisition) / max(1.0, float(voxels)) ** 0.5
        return float((sampling**2 + voltage**2 + kernel_term**2 + noise**2) ** 0.5 + self.within_scan_floor_hu)


@dataclass(frozen=True, slots=True)
class HarmonisedAttenuation:
    standard_hu: float
    half_width_hu: float
    raw_hu: float
    raw_sd_hu: float
    voxels: int
    voltage_term_hu: float
    kernel_term_hu: float

    def interval(self) -> tuple[float, float]:
        return (self.standard_hu - self.half_width_hu, self.standard_hu + self.half_width_hu)


def harmonise(
    mean_hu: float,
    sd_hu: float,
    voxels: int,
    acquisition: AcquisitionDescriptor,
    model: HarmonisationModel,
) -> HarmonisedAttenuation:
    family = scanner_family(acquisition.scanner_family)
    standard = model.corrected_mean(mean_hu, acquisition)
    half_width = model.uncertainty(sd_hu, voxels, acquisition)
    return HarmonisedAttenuation(
        standard_hu=standard,
        half_width_hu=half_width,
        raw_hu=mean_hu,
        raw_sd_hu=sd_hu,
        voxels=int(voxels),
        voltage_term_hu=model.voltage_slope_hu_per_kvp * voltage_delta_kvp(acquisition),
        kernel_term_hu=model.kernel_spread_hu * kernel_offset_hu(family, acquisition.reconstruction_kernel),
    )


def cross_scanner_spread(
    samples: tuple[tuple[AcquisitionDescriptor, float], ...],
    model: HarmonisationModel,
) -> float:
    corrected = [model.corrected_mean(mean_hu, acquisition) for acquisition, mean_hu in samples]
    if len(corrected) < 2:
        return 0.0
    spread = max(corrected) - min(corrected)
    return float(spread)
