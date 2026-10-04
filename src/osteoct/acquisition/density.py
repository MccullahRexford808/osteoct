"""Volumetric density indicator derived from the harmonised interval, texture and acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field

from osteoct.acquisition.scanner import kernel_offset_hu, scanner_family, voltage_delta_kvp
from osteoct.schema.records import AcquisitionDescriptor


@dataclass(frozen=True, slots=True)
class DensityModel:
    intercept: float = 42.0
    attenuation_gain: float = 0.78
    uncertainty_penalty: float = 0.55
    voltage_gain: float = 0.16
    kernel_gain: float = 3.1
    texture_weights: dict[str, float] = field(default_factory=dict)
    texture_scale: float = 1.0

    def predict(
        self,
        standard_attenuation_hu: float,
        uncertainty_hu: float,
        texture_vector: tuple[float, ...],
        texture_names: tuple[str, ...],
        acquisition: AcquisitionDescriptor,
    ) -> float:
        weighted = 0.0
        for name, value in zip(texture_names, texture_vector, strict=False):
            weighted += self.texture_weights.get(name, 0.0) * value
        family = scanner_family(acquisition.scanner_family)
        indicator = (
            self.intercept
            + self.attenuation_gain * standard_attenuation_hu
            - self.uncertainty_penalty * uncertainty_hu
            + self.voltage_gain * voltage_delta_kvp(acquisition)
            + self.kernel_gain * kernel_offset_hu(family, acquisition.reconstruction_kernel)
            + self.texture_scale * weighted
        )
        return float(max(0.0, indicator))


def density_indicator(
    standard_attenuation_hu: float,
    uncertainty_hu: float,
    texture_vector: tuple[float, ...],
    texture_names: tuple[str, ...],
    acquisition: AcquisitionDescriptor,
    model: DensityModel,
) -> float:
    return model.predict(standard_attenuation_hu, uncertainty_hu, texture_vector, texture_names, acquisition)
