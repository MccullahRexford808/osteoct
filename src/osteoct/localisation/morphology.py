"""Anatomical morphometry of a localised vertebral body."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.schema.records import MorphologyFeatures, VertebraRegion


@dataclass(frozen=True, slots=True)
class MorphometryParams:
    anterior_share: float = 0.34
    posterior_share: float = 0.34
    reference_column_fraction: float = 0.35
    anterior_depth_fraction: float = 0.80


def _roi_slices(region: VertebraRegion) -> tuple[slice, slice, slice]:
    centre = region.centre_index
    extent = region.extent_index
    return tuple(  # type: ignore[return-value]
        slice(max(0, centre[axis] - extent[axis] // 2), max(0, centre[axis] - extent[axis] // 2) + extent[axis])
        for axis in range(3)
    )


def body_rows(mask: npt.NDArray[np.bool_], params: MorphometryParams) -> tuple[int, int]:
    depth = mask.shape[1]
    limit = max(1, int(round(depth * params.anterior_depth_fraction)))
    occupied = mask[:, :limit, :].any(axis=(0, 2))
    rows = np.where(occupied)[0]
    if rows.size == 0:
        return (0, limit)
    return (int(rows.min()), int(rows.max()) + 1)


def band_height(mask: npt.NDArray[np.bool_], row_low: int, row_high: int, reference_fraction: float) -> float:
    depth = mask.shape[1]
    lower = max(0, min(depth - 1, row_low))
    upper = max(lower + 1, min(depth, row_high))
    band = mask[:, lower:upper, :]
    if band.size == 0:
        return 0.0
    heights = np.sort(band.sum(axis=2), axis=None)[::-1]
    keep = max(1, int(round(heights.size * reference_fraction)))
    return float(np.mean(heights[:keep]))


def morphology_from_mask(mask: npt.NDArray[np.bool_], params: MorphometryParams) -> MorphologyFeatures:
    if not bool(mask.any()):
        return MorphologyFeatures(1.0, 1.0, 1.0)
    row_low, row_high = body_rows(mask, params)
    span = max(1, row_high - row_low)
    anterior_high = row_low + max(1, int(round(span * params.anterior_share)))
    posterior_low = row_high - max(1, int(round(span * params.posterior_share)))
    posterior_low = max(anterior_high + 1, posterior_low)
    anterior = band_height(mask, row_low, anterior_high, params.reference_column_fraction)
    middle = band_height(mask, anterior_high, posterior_low, params.reference_column_fraction)
    posterior = band_height(mask, posterior_low, row_high, params.reference_column_fraction)
    reference = max(1.0, posterior)
    return MorphologyFeatures(
        anterior_height_mm=float(anterior / reference * posterior),
        middle_height_mm=float(middle / reference * posterior),
        posterior_height_mm=float(posterior),
    )


def measure_morphology(
    volume: npt.NDArray[np.float32],
    region: VertebraRegion,
    params: MorphometryParams,
    bone_threshold_hu: float,
) -> MorphologyFeatures:
    roi = _roi_slices(region)
    block = np.asarray(volume[roi], dtype=np.float64)
    mask = block >= bone_threshold_hu
    if not bool(mask.any()):
        mask = block >= float(np.percentile(block, 80.0))
    return morphology_from_mask(mask, params)


def heights_from_morphologies(morphologies: tuple[MorphologyFeatures, ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple((item.anterior_height_mm, item.middle_height_mm, item.posterior_height_mm) for item in morphologies)
