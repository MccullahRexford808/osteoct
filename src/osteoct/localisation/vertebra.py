"""Vertebral-body localisation and per-level region construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.acquisition.attenuation import AttenuationParams, clipped_roi_shape, trabecular_mask
from osteoct.localisation.morphology import MorphometryParams, measure_morphology
from osteoct.schema.records import MorphologyFeatures, VertebraRegion
from osteoct.schema.vocabulary import THORACOLUMBAR_LEVELS, VertebralLevel


@dataclass(frozen=True, slots=True)
class LocalisationParams:
    bone_window_low_hu: float = 55.0
    anterior_depth_fraction: float = 0.72
    minimum_body_width: int = 6
    maximum_body_width: int = 44
    minimum_run_y: int = 3
    minimum_slices: int = 2
    minimum_body_voxels: int = 220
    roi_half_extent: tuple[int, int, int] = (8, 7, 6)
    level_span: int = 13


@dataclass(frozen=True, slots=True)
class BodyCandidate:
    centroid: tuple[int, int, int]
    extent: tuple[int, int, int]
    voxel_count: int
    slices: tuple[int, ...]


def _bone_mask(volume: npt.NDArray[np.float32], params: LocalisationParams) -> npt.NDArray[np.bool_]:
    return np.asarray(volume, dtype=np.float64) >= params.bone_window_low_hu


def _slice_body(mask_slice: npt.NDArray[np.bool_], params: LocalisationParams) -> tuple[int, int] | None:
    depth = mask_slice.shape[1]
    limit = max(1, int(round(depth * params.anterior_depth_fraction)))
    anterior = mask_slice[:, :limit]
    counts = anterior.sum(axis=1)
    best: tuple[int, int] | None = None
    start: int | None = None
    for index in range(counts.shape[0] + 1):
        active = index < counts.shape[0] and int(counts[index]) >= params.minimum_run_y
        if active and start is None:
            start = index
        if not active and start is not None:
            width = index - start
            if params.minimum_body_width <= width <= params.maximum_body_width and (best is None or width > best[1] - best[0]):
                best = (start, index)
            start = None
    return best


def _body_extent(
    mask_slice: npt.NDArray[np.bool_], x_range: tuple[int, int], params: LocalisationParams
) -> tuple[int, int, int, int, int]:
    depth = mask_slice.shape[1]
    limit = max(1, int(round(depth * params.anterior_depth_fraction)))
    window = mask_slice[x_range[0] : x_range[1], :limit]
    rows = np.where(window.any(axis=1))[0]
    columns = np.where(window.any(axis=0))[0]
    y_low = int(columns.min())
    y_high = int(columns.max()) + 1
    if rows.size == 0 or columns.size == 0:
        return (x_range[0], x_range[1], 0, 1, 0)
    return (x_range[0], x_range[1], y_low, y_high, int(window.sum()))


def body_candidates(volume: npt.NDArray[np.float32], params: LocalisationParams) -> tuple[BodyCandidate, ...]:
    if volume.ndim != 3:
        raise ValueError("localisation needs a three-dimensional volume")
    mask = _bone_mask(volume, params)
    slices = mask.shape[2]
    per_slice: list[tuple[int, tuple[int, int, int, int, int]] | None] = []
    for z_index in range(slices):
        body = _slice_body(mask[:, :, z_index], params)
        if body is None:
            per_slice.append(None)
            continue
        per_slice.append((z_index, _body_extent(mask[:, :, z_index], body, params)))
    candidates: list[BodyCandidate] = []
    group: list[tuple[int, tuple[int, int, int, int, int]]] = []
    for entry in per_slice:
        if entry is None:
            _flush(group, candidates, params)
            group = []
            continue
        group.append(entry)
    _flush(group, candidates, params)
    return tuple(candidates)


def _flush(
    group: list[tuple[int, tuple[int, int, int, int, int]]],
    candidates: list[BodyCandidate],
    params: LocalisationParams,
) -> None:
    if len(group) < params.minimum_slices:
        return
    z_values = np.asarray([entry[0] for entry in group], dtype=np.float64)
    x_low = np.median([entry[1][0] for entry in group])
    x_high = np.median([entry[1][1] for entry in group])
    y_low = np.median([entry[1][2] for entry in group])
    y_high = np.median([entry[1][3] for entry in group])
    voxels = int(sum(entry[1][4] for entry in group))
    if voxels < params.minimum_body_voxels:
        return
    centre = (
        int(round((x_low + x_high) / 2.0)),
        int(round((y_low + y_high) / 2.0)),
        int(round(float(np.mean(z_values)))),
    )
    extent = (
        max(4, int(round(x_high - x_low))),
        max(4, int(round(y_high - y_low))),
        max(2, int(round(float(z_values.max() - z_values.min()) + 1))),
    )
    candidates.append(BodyCandidate(centre, extent, voxels, tuple(int(value) for value in z_values)))


def assign_levels(
    candidates: tuple[BodyCandidate, ...],
    params: LocalisationParams,
) -> tuple[tuple[VertebralLevel, BodyCandidate], ...]:
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda candidate: candidate.centroid[2])
    span = THORACOLUMBAR_LEVELS[-min(params.level_span, len(THORACOLUMBAR_LEVELS)) :]
    if len(ordered) > len(span):
        ordered = ordered[-len(span) :]
    assigned = list(zip(span[-len(ordered) :], ordered, strict=True))
    return tuple(assigned)


def region_from_candidate(
    candidate: BodyCandidate,
    params: LocalisationParams,
    attenuation: AttenuationParams,
    volume_shape: tuple[int, int, int],
) -> VertebraRegion:
    extent = (
        max(params.roi_half_extent[0] * 2, candidate.extent[0]),
        max(params.roi_half_extent[1] * 2, candidate.extent[1]),
        max(params.roi_half_extent[2] * 2, candidate.extent[2]),
    )
    probe = VertebraRegion(
        level=THORACOLUMBAR_LEVELS[0],
        centre_index=candidate.centroid,
        extent_index=extent,
        voxel_count=candidate.voxel_count,
        trabecular_mask_fraction=0.0,
    )
    mask = trabecular_mask(clipped_roi_shape(probe, volume_shape), attenuation)
    fraction = float(mask.sum()) / float(mask.size)
    return VertebraRegion(
        level=probe.level,
        centre_index=probe.centre_index,
        extent_index=probe.extent_index,
        voxel_count=candidate.voxel_count,
        trabecular_mask_fraction=fraction,
    )


def localise(
    volume: npt.NDArray[np.float32],
    params: LocalisationParams | None = None,
    attenuation: AttenuationParams | None = None,
    morphometry: MorphometryParams | None = None,
) -> tuple[tuple[VertebraRegion, ...], tuple[MorphologyFeatures, ...]]:
    params = LocalisationParams() if params is None else params
    attenuation = AttenuationParams() if attenuation is None else attenuation
    morphometry = MorphometryParams() if morphometry is None else morphometry
    assigned = assign_levels(body_candidates(volume, params), params)
    volume_shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    regions: list[VertebraRegion] = []
    morphologies: list[MorphologyFeatures] = []
    for level, candidate in assigned:
        region = region_from_candidate(candidate, params, attenuation, volume_shape)
        regions.append(
            VertebraRegion(
                level=level,
                centre_index=region.centre_index,
                extent_index=region.extent_index,
                voxel_count=region.voxel_count,
                trabecular_mask_fraction=region.trabecular_mask_fraction,
            )
        )
        morphologies.append(measure_morphology(volume, region, morphometry, params.bone_window_low_hu))
    return (tuple(regions), tuple(morphologies))


def level_span(params: LocalisationParams | None = None) -> tuple[VertebralLevel, ...]:
    resolved = LocalisationParams() if params is None else params
    return THORACOLUMBAR_LEVELS[-min(resolved.level_span, len(THORACOLUMBAR_LEVELS)) :]
