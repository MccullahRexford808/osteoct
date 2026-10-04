"""Trabecular attenuation measured inside a region that excludes cortical bone and endplates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from osteoct.schema.records import MorphologyFeatures, VertebraMeasurement, VertebraRegion


@dataclass(frozen=True, slots=True)
class AttenuationParams:
    cortical_erosion_voxels: int = 1
    endplate_exclusion_fraction: float = 0.18
    minimum_trabecular_voxels: int = 24
    marrow_fat_correction_hu: float = 2.5
    hu_window_low: float = -200.0
    hu_window_high: float = 1400.0
    bone_threshold_hu: float = 55.0


@dataclass(frozen=True, slots=True)
class TrabecularSampling:
    mean_hu: float
    sd_hu: float
    voxels: int
    eroded_fraction: float
    excluded_by_window: int


class NoAssessableVoxels(Exception):
    pass


def _roi_slices(region: VertebraRegion) -> tuple[slice, slice, slice]:
    centre = region.centre_index
    extent = region.extent_index
    return tuple(  # type: ignore[return-value]
        slice(
            max(0, centre[axis] - extent[axis] // 2),
            max(0, centre[axis] - extent[axis] // 2) + extent[axis],
        )
        for axis in range(3)
    )


def clipped_roi_shape(region: VertebraRegion, volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    roi = _roi_slices(region)
    return tuple(  # type: ignore[return-value]
        max(1, min(volume_shape[axis], roi[axis].stop) - min(volume_shape[axis], roi[axis].start)) for axis in range(3)
    )


def trabecular_mask(shape: tuple[int, int, int], params: AttenuationParams) -> npt.NDArray[np.bool_]:
    mask = np.ones(shape, dtype=bool)
    erosion = params.cortical_erosion_voxels
    if erosion > 0 and all(length > 2 * erosion for length in shape):
        mask[:erosion, :, :] = False
        mask[-erosion:, :, :] = False
        mask[:, :erosion, :] = False
        mask[:, -erosion:, :] = False
        mask[:, :, :erosion] = False
        mask[:, :, -erosion:] = False
    slab = int(round(shape[2] * params.endplate_exclusion_fraction))
    if slab > 0 and shape[2] > 2 * slab:
        mask[:, :, :slab] = False
        mask[:, :, -slab:] = False
    return mask


def trabecular_region_mask(
    block: npt.NDArray[np.float64],
    params: AttenuationParams,
) -> npt.NDArray[np.bool_]:
    geometric = trabecular_mask(block.shape, params)
    structure = np.ones((3, 3, 3), dtype=bool)
    osseous = np.asarray(block, dtype=np.float64) >= params.bone_threshold_hu
    eroded: npt.NDArray[np.bool_] = np.asarray(ndimage.binary_erosion(osseous, structure=structure, iterations=1), dtype=bool)
    combined: npt.NDArray[np.bool_] = geometric & eroded
    if int(combined.sum()) < params.minimum_trabecular_voxels:
        combined = geometric & osseous
    return combined


def sample_region(
    volume: npt.NDArray[np.float32],
    region: VertebraRegion,
    params: AttenuationParams,
) -> TrabecularSampling:
    roi = _roi_slices(region)
    block = np.asarray(volume[roi], dtype=np.float64)
    mask = trabecular_region_mask(block, params)
    values = block[mask]
    if values.size < params.minimum_trabecular_voxels:
        raise NoAssessableVoxels("region carries fewer trabecular voxels than the assessment floor")
    inside = (values >= params.hu_window_low) & (values <= params.hu_window_high)
    excluded = int(values.size - int(inside.sum()))
    clipped = values[inside]
    if clipped.size < params.minimum_trabecular_voxels:
        raise NoAssessableVoxels("region carries fewer in-window voxels than the assessment floor")
    eroded_fraction = 1.0 - float(mask.sum()) / float(mask.size)
    return TrabecularSampling(
        mean_hu=float(np.mean(clipped)) - params.marrow_fat_correction_hu,
        sd_hu=float(np.std(clipped, ddof=1)) if clipped.size > 1 else 0.0,
        voxels=int(clipped.size),
        eroded_fraction=eroded_fraction,
        excluded_by_window=excluded,
    )


def measure_vertebra(
    volume: npt.NDArray[np.float32],
    region: VertebraRegion,
    params: AttenuationParams,
) -> tuple[float, float]:
    sampling = sample_region(volume, region, params)
    return (sampling.mean_hu, sampling.sd_hu)


def vertebra_record(
    volume: npt.NDArray[np.float32],
    region: VertebraRegion,
    morphology: MorphologyFeatures,
    params: AttenuationParams,
) -> VertebraMeasurement:
    mean_hu, sd_hu = measure_vertebra(volume, region, params)
    return VertebraMeasurement(
        region=region,
        morphology=morphology,
        mean_attenuation_hu=mean_hu,
        attenuation_sd_hu=sd_hu,
    )


def examination_attenuation(
    volume: npt.NDArray[np.float32],
    regions: tuple[VertebraRegion, ...],
    morphologies: tuple[MorphologyFeatures, ...],
    params: AttenuationParams,
) -> tuple[tuple[VertebraMeasurement, ...], float, float, int]:
    if len(regions) != len(morphologies):
        raise ValueError("every region needs one morphology record")
    measurements: list[VertebraMeasurement] = []
    weights: list[float] = []
    for region, morphology in zip(regions, morphologies, strict=True):
        measurements.append(vertebra_record(volume, region, morphology, params))
        weights.append(float(region.voxel_count))
    if not measurements:
        raise NoAssessableVoxels("no assessable vertebra in the examination")
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array = weight_array / float(weight_array.sum())
    means = np.asarray([measurement.mean_attenuation_hu for measurement in measurements], dtype=np.float64)
    sds = np.asarray([measurement.attenuation_sd_hu for measurement in measurements], dtype=np.float64)
    pooled_mean = float(np.sum(weight_array * means))
    pooled_sd = float(np.sqrt(np.sum(weight_array * sds**2)))
    total_voxels = int(sum(measurement.region.voxel_count for measurement in measurements))
    return tuple(measurements), pooled_mean, pooled_sd, total_voxels
