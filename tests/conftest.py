"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from osteoct.acquisition.harmonise import HarmonisationModel
from osteoct.agent.planner import Planner, PlannerConfig
from osteoct.agent.tools import Toolbox, ToolRegistry
from osteoct.evaluation.study import StudyContext, build_study
from osteoct.localisation.morphology import MorphometryParams
from osteoct.localisation.vertebra import LocalisationParams
from osteoct.models.scorer import ScorerBatch
from osteoct.ontology.axioms import AxiomSet
from osteoct.ontology.bands import BandCalibration
from osteoct.ontology.index import build_concept_index
from osteoct.ontology.severity import GenantThresholds
from osteoct.ontology.tiers import ProtocolEnvelope
from osteoct.schema.records import AcquisitionDescriptor, MeasurementBundle, MorphologyFeatures, VertebraMeasurement, VertebraRegion
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier, VertebralLevel
from osteoct.texture.descriptors import TextureParams
from osteoct.utils.config import load_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def smoke_config() -> object:
    return load_experiment(REPOSITORY_ROOT, "_smoke")


@pytest.fixture(scope="session")
def smoke_study(smoke_config: object) -> StudyContext:
    from osteoct.utils.config import ExperimentConfig

    assert isinstance(smoke_config, ExperimentConfig)
    return build_study(smoke_config)


@pytest.fixture()
def attributes() -> AxiomSet:
    return AxiomSet(
        index=build_concept_index(),
        calibration=BandCalibration(240.0, 45.0, 1.5),
        thresholds=GenantThresholds(),
        envelope=ProtocolEnvelope(2.5, 80.0, ("B40f", "B30f", "STANDARD")),
    )


@pytest.fixture()
def toolbox(attributes: AxiomSet) -> Toolbox:
    return Toolbox(
        localisation=LocalisationParams(bone_window_low_hu=55.0, roi_half_extent=(8, 7, 6), level_span=13),
        attenuation=_attenuation_params(),
        morphometry=MorphometryParams(),
        texture=TextureParams(),
        harmonisation=HarmonisationModel(0.42, 120.0, 0.85, 2.2, 1.0),
        density=_density_model(),
        index=attributes.index,
        stability=None,
    )


def _attenuation_params() -> object:
    from osteoct.acquisition.attenuation import AttenuationParams

    return AttenuationParams(bone_threshold_hu=55.0)


def _density_model() -> object:
    from osteoct.acquisition.density import DensityModel

    return DensityModel(texture_weights={"glcm_contrast": -0.037, "rlm_sre": 18.0})


@pytest.fixture()
def measurement_bundle() -> MeasurementBundle:
    region = VertebraRegion(VertebralLevel.L2, (8, 8, 4), (12, 10, 8), 240, 0.62)
    morphology = MorphologyFeatures(7.0, 10.0, 10.0)
    return MeasurementBundle(
        standard_attenuation_hu=200.0,
        uncertainty_hu=3.0,
        texture_vector=(0.31, 0.42, 0.55),
        density_indicator=181.5,
        acquisition=AcquisitionDescriptor(120.0, "vendor_a", "B40f", 1.0),
        vertebrae=(VertebraMeasurement(region, morphology, 200.0, 8.5),),
        retained_texture_names=("glcm_contrast", "rlm_sre", "wavelet_lll_energy"),
    )


@pytest.fixture()
def decision_context() -> DiagnosticContext:
    return DiagnosticContext(prior_fragility_fracture=False, treatment_exposure=False, reference_t_score=None)


@pytest.fixture()
def planner() -> Planner:
    return Planner(PlannerConfig(arm="constrained"))


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


def make_batch(size: int = 4, texture_dim: int = 25, concept_dim: int = 30, embedding_dim: int = 16) -> ScorerBatch:
    return ScorerBatch(
        descriptors=np.linspace(150.0, 240.0, size * 3, dtype=np.float64).reshape(size, 3),
        texture=np.zeros((size, texture_dim), dtype=np.float64),
        concepts=np.zeros((size, concept_dim), dtype=np.float64),
        context=np.zeros((size, 2), dtype=np.float64),
        embedding=np.zeros((size, embedding_dim), dtype=np.float64),
        band_target=np.zeros(size, dtype=np.int64),
        tier_target=np.zeros(size, dtype=np.int64),
        severity_target=np.zeros(size, dtype=np.int64),
        risk_target=np.zeros(size, dtype=np.float64),
    )


def checkered_volume(shape: tuple[int, int, int] = (8, 8, 8)) -> npt.NDArray[np.float32]:
    values = np.indices(shape).sum(axis=0) % 2
    return np.asarray(values * 300.0 + 40.0, dtype=np.float32)


@pytest.fixture()
def score_vectors() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(17)
    labels = (rng.random(120) < 0.45).astype(np.int64)
    scores = rng.normal(0.0, 1.0, size=120) + 1.3 * labels
    return (scores, labels)


@pytest.fixture()
def bands_and_tiers() -> tuple[tuple[DiagnosticBand, ...], tuple[RiskTier, ...], tuple[DeformityGrade, ...]]:
    return (
        (DiagnosticBand.NORMAL, DiagnosticBand.OSTEOPENIA, DiagnosticBand.OSTEOPOROSIS, DiagnosticBand.SEVERE_OSTEOPOROSIS),
        (RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH, RiskTier.VERY_HIGH),
        (DeformityGrade.NONE, DeformityGrade.MILD, DeformityGrade.MODERATE, DeformityGrade.SEVERE),
    )
