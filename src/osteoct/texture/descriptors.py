"""The retained texture vector: four descriptor families followed by the stability filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from osteoct.texture.glcm import GLCM_FEATURES, glcm_descriptors, quantise
from osteoct.texture.lbp import LBP_FEATURES, lbp_descriptors
from osteoct.texture.runlength import RUNLENGTH_FEATURES, runlength_descriptors
from osteoct.texture.stability import StabilityVerdict, apply_filter, icc_2_1, stability_filter
from osteoct.texture.wavelet import WAVELET_FEATURES, wavelet_descriptors


@dataclass(frozen=True, slots=True)
class TextureParams:
    grey_levels: int = 32
    window_low_hu: float = -200.0
    window_high_hu: float = 1400.0
    wavelet_levels: int = 2
    stability_threshold: float = 0.75


def descriptor_names() -> tuple[str, ...]:
    return GLCM_FEATURES + RUNLENGTH_FEATURES + WAVELET_FEATURES + LBP_FEATURES


def descriptor_vector(block: npt.NDArray[np.float64], params: TextureParams) -> npt.NDArray[np.float64]:
    if block.ndim != 3:
        raise ValueError("texture descriptors need a three-dimensional region")
    if min(block.shape) < 4:
        raise ValueError("the region is smaller than the smallest supported descriptor window")
    quantised = quantise(block, params.window_low_hu, params.window_high_hu, params.grey_levels)
    parts = (
        glcm_descriptors(quantised, params.grey_levels),
        runlength_descriptors(quantised, params.grey_levels),
        wavelet_descriptors(block, params.wavelet_levels),
        lbp_descriptors(block),
    )
    expected = (len(GLCM_FEATURES), len(RUNLENGTH_FEATURES), len(WAVELET_FEATURES), len(LBP_FEATURES))
    for part, size in zip(parts, expected, strict=True):
        if part.shape != (size,):
            raise ValueError("descriptor family returned an unexpected width")
    return np.concatenate(parts).astype(np.float64)


def fit_stability(
    blocks_first: tuple[npt.NDArray[np.float64], ...],
    blocks_second: tuple[npt.NDArray[np.float64], ...],
    params: TextureParams,
) -> StabilityVerdict:
    if len(blocks_first) != len(blocks_second):
        raise ValueError("the repeat set must pair one to one")
    names = descriptor_names()
    first = np.stack([descriptor_vector(block, params) for block in blocks_first]) if blocks_first else np.zeros((0, len(names)))
    second = np.stack([descriptor_vector(block, params) for block in blocks_second]) if blocks_second else np.zeros((0, len(names)))
    return stability_filter(names, first, second, params.stability_threshold)


def retained_texture(
    block: npt.NDArray[np.float64],
    verdict: StabilityVerdict,
    params: TextureParams,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    vector = descriptor_vector(block, params)
    return apply_filter(vector, descriptor_names(), verdict)


def feature_agreement(
    blocks_first: tuple[npt.NDArray[np.float64], ...],
    blocks_second: tuple[npt.NDArray[np.float64], ...],
    params: TextureParams,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    names = descriptor_names()
    if not blocks_first:
        return (names, tuple(0.0 for _ in names))
    first = np.stack([descriptor_vector(block, params) for block in blocks_first])
    second = np.stack([descriptor_vector(block, params) for block in blocks_second])
    values = tuple(icc_2_1(first[:, column], second[:, column]) for column in range(len(names)))
    return (names, values)
