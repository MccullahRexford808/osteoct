"""Schema, ontology, acquisition and texture unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from osteoct.acquisition.attenuation import (
    AttenuationParams,
    NoAssessableVoxels,
    sample_region,
    trabecular_mask,
    trabecular_region_mask,
)
from osteoct.acquisition.density import DensityModel, density_indicator
from osteoct.acquisition.harmonise import HarmonisationModel, cross_scanner_spread, harmonise
from osteoct.acquisition.scanner import acquisition_descriptor, kernel_offset_hu, scanner_family, voltage_delta_kvp
from osteoct.grounding.standard import ground_decision, ground_measurements
from osteoct.ontology.axioms import (
    ASSERTED_ELEMENTS,
    EVIDENCE_REQUIREMENTS,
    FAMILY_ORDER,
    FAMILY_WEIGHTS,
    AxiomSet,
    FamilyName,
    conformance_ceiling,
    conformance_score,
    supported_code,
)
from osteoct.ontology.bands import (
    BandCalibration,
    band_consistent,
    band_from_interval,
    band_from_t_score,
    bands_from_interval,
    interval_straddles_threshold,
)
from osteoct.ontology.index import CODE_TO_ELEMENT, INDEX_ENTRIES, build_concept_index, entry_for
from osteoct.ontology.profile import (
    ProfileViolation,
    atom,
    cardinality,
    conjunction,
    el_subsumes,
    existential,
    is_el,
    negation,
    profile_errors,
    require_el,
    universal,
)
from osteoct.ontology.severity import GenantThresholds, grade_from_height_loss, grade_from_morphology, grade_from_vertebrae
from osteoct.ontology.tiers import ProtocolEnvelope, frax_probability, tier_from_band
from osteoct.schema.concepts import TermSystem
from osteoct.schema.records import AcquisitionDescriptor, MorphologyFeatures, VertebraRegion
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier, VertebralLevel
from osteoct.texture.glcm import cooccurrence, glcm_descriptors, harness_features, quantise
from osteoct.texture.lbp import lbp_descriptors, pattern_codes
from osteoct.texture.runlength import run_length_matrix, runlength_descriptors
from osteoct.texture.stability import icc_2_1, stability_filter
from osteoct.texture.wavelet import haar_decompose, wavelet_descriptors


def test_concept_index_is_unique_and_resolvable() -> None:
    index = build_concept_index()
    keys = [concept.key() for concept in index.concepts]
    assert len(keys) == len(set(keys))
    for entry in INDEX_ENTRIES:
        assert index.resolve(entry.key) is not None
        assert index.contains(entry.term.lower())


def test_concept_index_records_system_and_element() -> None:
    index = build_concept_index()
    systems = {concept.system for concept in index.concepts}
    assert systems == {TermSystem.SNOMED_CT, TermSystem.RADLEX, TermSystem.LOINC, TermSystem.DICOM_SR}
    assert set(CODE_TO_ELEMENT) == {entry.key for entry in INDEX_ENTRIES}
    assert entry_for("vertebral_level", TermSystem.RADLEX).system is TermSystem.RADLEX


def test_band_from_t_score_thresholds() -> None:
    assert band_from_t_score(-0.5) is DiagnosticBand.NORMAL
    assert band_from_t_score(-1.5) is DiagnosticBand.OSTEOPENIA
    assert band_from_t_score(-3.0) is DiagnosticBand.OSTEOPOROSIS
    assert band_from_t_score(-3.0, prior_fragility_fracture=True) is DiagnosticBand.SEVERE_OSTEOPOROSIS


def test_band_interval_uses_the_whole_interval() -> None:
    calibration = BandCalibration(240.0, 45.0, 1.5)
    low_threshold, high_threshold = calibration.thresholds_hu()
    compatible = bands_from_interval(high_threshold, 6.0, calibration)
    assert len(compatible) >= 2
    assert interval_straddles_threshold(high_threshold, 6.0, calibration)
    assert not interval_straddles_threshold(high_threshold + 30.0, 2.0, calibration)
    assert band_from_interval(high_threshold + 30.0, 2.0, calibration) is DiagnosticBand.NORMAL


def test_band_consistency_uses_the_reference_when_present() -> None:
    calibration = BandCalibration(240.0, 45.0, 1.5)
    context = DiagnosticContext(reference_t_score=0.5)
    assert band_consistent(DiagnosticBand.NORMAL, 260.0, 3.0, calibration, context)
    assert not band_consistent(DiagnosticBand.OSTEOPOROSIS, 260.0, 3.0, calibration, context)


@pytest.mark.parametrize(
    ("band", "prior", "treatment", "expected"),
    [
        (DiagnosticBand.NORMAL, False, False, RiskTier.LOW),
        (DiagnosticBand.OSTEOPENIA, False, False, RiskTier.MODERATE),
        (DiagnosticBand.OSTEOPOROSIS, False, False, RiskTier.HIGH),
        (DiagnosticBand.SEVERE_OSTEOPOROSIS, False, False, RiskTier.VERY_HIGH),
        (DiagnosticBand.OSTEOPENIA, True, False, RiskTier.HIGH),
        (DiagnosticBand.OSTEOPOROSIS, False, True, RiskTier.MODERATE),
    ],
)
def test_tier_derivation(band: DiagnosticBand, prior: bool, treatment: bool, expected: RiskTier) -> None:
    attributes_index = build_concept_index()
    envelope = ProtocolEnvelope(2.5, 80.0, ("B40f",))
    acquisition = AcquisitionDescriptor(120.0, "vendor_a", "B40f", 1.0)
    concepts = (entry_for("risk_tier", TermSystem.LOINC).key,)
    context = DiagnosticContext(prior_fragility_fracture=prior, treatment_exposure=treatment)
    assert tier_from_band(band, context, acquisition, concepts, attributes_index, envelope) is expected


def test_tier_demotes_outside_the_protocol_envelope() -> None:
    index = build_concept_index()
    envelope = ProtocolEnvelope(1.0, 80.0, ("B40f",))
    concepts = (entry_for("risk_tier", TermSystem.LOINC).key,)
    thin = AcquisitionDescriptor(120.0, "vendor_a", "B40f", 2.0)
    thick = AcquisitionDescriptor(120.0, "vendor_a", "B40f", 0.8)
    context = DiagnosticContext()
    assert tier_from_band(DiagnosticBand.OSTEOPOROSIS, context, thin, concepts, index, envelope) is RiskTier.MODERATE
    assert tier_from_band(DiagnosticBand.OSTEOPOROSIS, context, thick, concepts, index, envelope) is RiskTier.HIGH


def test_frax_probability_is_monotone_in_severity() -> None:
    values = [
        frax_probability(DiagnosticBand.OSTEOPOROSIS, tier, prior, treatment)
        for tier, prior, treatment in (
            (RiskTier.HIGH, False, False),
            (RiskTier.HIGH, True, False),
            (RiskTier.VERY_HIGH, True, False),
        )
    ]
    assert values[0] < values[1] < values[2]


def test_genant_grading_boundaries() -> None:
    thresholds = GenantThresholds()
    assert grade_from_height_loss(0.10, thresholds) is DeformityGrade.NONE
    assert grade_from_height_loss(0.20, thresholds) is DeformityGrade.MILD
    assert grade_from_height_loss(0.24, thresholds) is DeformityGrade.MILD
    assert grade_from_height_loss(0.30, thresholds) is DeformityGrade.MODERATE
    assert grade_from_height_loss(0.42, thresholds) is DeformityGrade.SEVERE


def test_genant_grading_reads_the_worst_vertebra() -> None:
    thresholds = GenantThresholds()
    morphology = (
        MorphologyFeatures(10.0, 10.0, 10.0),
        MorphologyFeatures(6.0, 10.0, 10.0),
        MorphologyFeatures(9.5, 9.0, 10.0),
    )
    assert grade_from_vertebrae(morphology, thresholds) is DeformityGrade.SEVERE
    assert grade_from_morphology(morphology[0], thresholds) is DeformityGrade.NONE


def test_profile_accepts_conjunction_and_existential() -> None:
    expression = conjunction(
        existential("derived_from", atom("HarmonisedInterval")),
        existential("linked_to", atom("Measurement")),
    )
    assert is_el(expression)
    require_el(expression)


@pytest.mark.parametrize("expression", [negation(atom("X")), universal("role", atom("X")), cardinality("role", 2)])
def test_profile_rejects_non_el_constructors(expression: object) -> None:
    with pytest.raises(ProfileViolation):
        require_el(expression)  # type: ignore[arg-type]
    assert profile_errors(expression)  # type: ignore[arg-type]


def test_role_hierarchy_subsumption() -> None:
    definitions = {
        "Band": existential("derived_from", atom("HarmonisedInterval")),
        "Tier": existential("derived_from", atom("Band")),
    }
    assert el_subsumes(definitions["Tier"], "HarmonisedInterval", definitions, (("derived_from", "derived_from"),))


def test_trabecular_mask_excludes_the_shell_and_endplates() -> None:
    params = AttenuationParams()
    mask = trabecular_mask((10, 10, 10), params)
    assert not mask[0, :, :].any()
    assert not mask[:, :, 0].any()
    assert mask[5, 5, 5]
    assert 0.0 < float(mask.mean()) < 1.0


def test_attenuation_measurement_recovers_a_constant_block() -> None:
    block = np.full((12, 12, 10), 180.0, dtype=np.float32)
    block[0, :, :] = 900.0
    block[-1, :, :] = 900.0
    region = VertebraRegion(VertebralLevel.L1, (10, 10, 6), (12, 12, 10), 1200, 0.6)
    sampling = sample_region(block, region, AttenuationParams(marrow_fat_correction_hu=0.0))
    assert abs(sampling.mean_hu - 180.0) < 1.0
    assert sampling.voxels > 24


def test_attenuation_raises_when_too_few_voxels() -> None:
    block = np.full((5, 5, 4), 180.0, dtype=np.float32)
    region = VertebraRegion(VertebralLevel.L1, (8, 8, 4), (30, 30, 30), 40, 0.4)
    with pytest.raises(NoAssessableVoxels):
        sample_region(block, region, AttenuationParams(minimum_trabecular_voxels=200))


def test_trabecular_region_mask_falls_back_to_osseous() -> None:
    block = np.full((12, 12, 10), 20.0, dtype=np.float64)
    block[4:8, 4:8, 3:7] = 190.0
    mask = trabecular_region_mask(block, AttenuationParams(bone_threshold_hu=55.0))
    assert mask.sum() > 0
    assert bool(mask[5, 5, 5])


def test_harmonisation_recovers_the_standard_value() -> None:
    model = HarmonisationModel(0.42, 120.0, 0.85, 2.2, 1.0)
    acquisition = AcquisitionDescriptor(140.0, "vendor_c", "FC50", 1.0)
    family = scanner_family("vendor_c")
    kernel_term = kernel_offset_hu(family, "FC50")
    voltage_term = 0.42 * voltage_delta_kvp(acquisition)
    latent = 205.0
    observed = family.attenuation_scale * (latent + voltage_term + kernel_term) + family.attenuation_offset_hu
    result = harmonise(observed, 9.0, 240, acquisition, model)
    assert abs(result.standard_hu - latent) < 1e-6
    assert result.half_width_hu > 0.0
    assert result.interval()[0] < result.standard_hu < result.interval()[1]


def test_cross_scanner_spread_narrows_after_harmonisation() -> None:
    model = HarmonisationModel(0.42, 120.0, 0.85, 2.2, 1.0)
    latent = 205.0
    acquisitions = (
        AcquisitionDescriptor(120.0, "vendor_a", "B30f", 1.0),
        AcquisitionDescriptor(140.0, "vendor_d", "Br60d", 1.25),
    )
    raw: list[float] = []
    corrected: list[tuple[AcquisitionDescriptor, float]] = []
    for acquisition in acquisitions:
        family = scanner_family(acquisition.scanner_family)
        kernel_term = kernel_offset_hu(family, acquisition.reconstruction_kernel)
        voltage_term = 0.42 * voltage_delta_kvp(acquisition)
        observed = family.attenuation_scale * (latent + voltage_term + kernel_term) + family.attenuation_offset_hu
        raw.append(observed)
        corrected.append((acquisition, observed))
    assert cross_scanner_spread(tuple(corrected), model) < (max(raw) - min(raw)) / 10.0


def test_descriptor_from_scanner_is_stable() -> None:
    descriptor = acquisition_descriptor(120.0, "vendor_a", "B40f", 1.0)
    assert descriptor.signature().startswith("120kVp")
    assert len(descriptor.as_vector()) == 3


def test_density_indicator_is_positive_and_monotone() -> None:
    model = DensityModel(texture_weights={"glcm_contrast": -0.037}, texture_scale=1.0)
    acquisition = AcquisitionDescriptor(120.0, "vendor_a", "B40f", 1.0)
    low = density_indicator(120.0, 4.0, (0.3,), ("glcm_contrast",), acquisition, model)
    high = density_indicator(240.0, 4.0, (0.3,), ("glcm_contrast",), acquisition, model)
    assert 0.0 < low < high


def test_ground_measurements_binds_every_required_element() -> None:
    region = VertebraRegion(VertebralLevel.L1, (8, 8, 4), (12, 10, 8), 240, 0.6)
    grounding = ground_measurements((region,), 200.0, 3.0, 180.0)
    assert grounding.concepts
    assert grounding.binding_for("trabecular_attenuation")
    assert grounding.binding_for("vertebral_level")


def test_ground_decision_adds_the_band_and_history_bindings() -> None:
    index = build_concept_index()
    region = VertebraRegion(VertebralLevel.L1, (8, 8, 4), (12, 10, 8), 240, 0.6)
    base = ground_measurements((region,), 200.0, 3.0, 180.0)
    decision = ground_decision(
        base, DiagnosticBand.OSTEOPENIA, RiskTier.MODERATE, DeformityGrade.MODERATE, False, True, True, index
    )
    assert decision.binding_for("diagnostic_band")
    assert decision.binding_for("deformity_grade")
    assert len(decision.concepts) > len(base.concepts)


def test_axiom_families_and_weights() -> None:
    assert tuple(FAMILY_WEIGHTS) == FAMILY_ORDER
    assert tuple(FAMILY_WEIGHTS.values()) != ()
    assert abs(sum(FAMILY_WEIGHTS.values()) - 1.0) < 1e-12
    assert set(ASSERTED_ELEMENTS) <= set(EVIDENCE_REQUIREMENTS) | {"diagnostic_band", "risk_tier", "deformity_grade"}


def test_conformance_ceiling_falls_when_a_family_is_removed(attributes: AxiomSet) -> None:
    full = conformance_ceiling(attributes)
    assert abs(full - 1.0) < 1e-12
    for family in FAMILY_ORDER:
        reduced = attributes.without_family(family)
        assert conformance_ceiling(reduced) < full
        assert abs(conformance_ceiling(reduced) - (1.0 - FAMILY_WEIGHTS[family])) < 1e-12


def index_key(element: str, system: TermSystem = TermSystem.LOINC) -> str:
    entry = entry_for(element, system)
    return f"{entry.system.value}:{entry.key}"


def test_conformance_score_uses_the_five_family_denominator(attributes: AxiomSet, measurement_bundle: object) -> None:
    from osteoct.schema.records import CandidateDecision, MeasurementBundle

    assert isinstance(measurement_bundle, MeasurementBundle)
    candidate = CandidateDecision(
        band=DiagnosticBand.OSTEOPENIA,
        tier=RiskTier.MODERATE,
        severity=grade_from_vertebrae(tuple(m.morphology for m in measurement_bundle.vertebrae), attributes.thresholds),
        concepts=(index_key("trabecular_attenuation"),),
        evidence=(),
        probability=0.3,
        context=DiagnosticContext(),
    )
    identity_only = attributes.with_only((FamilyName.IDENTITY,))
    assert conformance_score(identity_only, candidate, measurement_bundle) == pytest.approx(FAMILY_WEIGHTS[FamilyName.IDENTITY])
    assert not supported_code(candidate, attributes, measurement_bundle)


def test_quantise_clips_to_the_window() -> None:
    block = np.asarray([[-100.0, 0.0, 700.0, 5000.0]], dtype=np.float64)
    levels = quantise(block, 0.0, 1400.0, 8)
    assert levels.min() >= 0
    assert levels.max() <= 7


def test_cooccurrence_is_normalised_and_symmetric() -> None:
    quantised = np.asarray([[[0, 1, 1], [2, 2, 0]], [[3, 3, 1], [0, 2, 2]]], dtype=np.int32)
    matrix = cooccurrence(quantised, (0, 0, 1), 4)
    assert abs(float(matrix.sum()) - 1.0) < 1e-12
    assert np.allclose(matrix, matrix.T)


def test_glcm_descriptors_have_the_declared_width() -> None:
    rng = np.random.default_rng(2)
    block = rng.normal(180.0, 20.0, size=(8, 8, 8))
    quantised = quantise(block, 0.0, 400.0, 16)
    descriptors = glcm_descriptors(quantised, 16)
    assert descriptors.shape == (5,)
    assert np.all(np.isfinite(descriptors))


def test_harness_contrast_on_a_two_level_matrix() -> None:
    matrix = np.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    features = harness_features(matrix)
    assert features[0] == pytest.approx(1.0)
    assert features[2] == pytest.approx(float(np.sqrt(0.5)))


def test_run_length_matrix_counts_runs() -> None:
    block = np.asarray([[[0, 0, 1, 1, 1, 2, 2, 0]]], dtype=np.int32)
    matrix = run_length_matrix(block, 2, 3)
    assert float(matrix.sum()) == 4.0
    assert matrix[0, 1] == 1.0
    assert matrix[1, 2] == 1.0


def test_runlength_descriptors_have_the_declared_width() -> None:
    rng = np.random.default_rng(4)
    block = rng.integers(0, 4, size=(8, 8, 8)).astype(np.int32)
    descriptors = runlength_descriptors(block, 4)
    assert descriptors.shape == (5,)
    assert np.all(np.isfinite(descriptors))


def test_haar_decomposition_energy_is_conserved() -> None:
    rng = np.random.default_rng(6)
    block = rng.normal(0.0, 1.0, size=(8, 8, 8))
    bands = haar_decompose(block, 1)
    total = sum(float(np.sum(values**2)) for values in bands.values())
    assert total == pytest.approx(float(np.sum(block**2)), rel=1e-9)


def test_wavelet_descriptors_are_non_negative() -> None:
    rng = np.random.default_rng(8)
    block = rng.normal(180.0, 25.0, size=(8, 8, 8))
    descriptors = wavelet_descriptors(block, 2)
    assert descriptors.shape == (10,)
    assert float(descriptors[-2]) >= 0.0


def test_pattern_codes_count_superseding_neighbours() -> None:
    block = np.zeros((3, 3, 3), dtype=np.float64)
    block[1, 1, 1] = 1.0
    codes = pattern_codes(block)
    assert int(codes[1, 1, 1]) == 0
    assert int(codes[0, 0, 0]) > 0


def test_lbp_descriptors_have_the_declared_width() -> None:
    rng = np.random.default_rng(10)
    block = rng.normal(0.0, 1.0, size=(6, 6, 6))
    descriptors = lbp_descriptors(block)
    assert descriptors.shape == (5,)
    assert 0.0 <= float(descriptors[3]) <= 1.0


def test_stability_filter_keeps_agreeing_descriptors() -> None:
    first = np.asarray([[1.0, 5.0], [2.0, 5.1], [3.0, 4.9], [4.0, 5.0]], dtype=np.float64)
    second = np.asarray([[1.01, 30.0], [2.02, 12.0], [2.98, 22.0], [4.03, 40.0]], dtype=np.float64)
    verdict = stability_filter(("stable", "unstable"), first, second, 0.75)
    assert verdict.retained == ("stable",)
    assert verdict.dropped == ("unstable",)
    assert abs(verdict.drop_share() - 0.5) < 1e-12


def test_icc_is_one_for_identical_repeats() -> None:
    values = np.asarray([0.4, 1.1, 1.9, 2.6], dtype=np.float64)
    assert icc_2_1(values, values) == pytest.approx(1.0)


def test_icc_is_negative_for_opposed_repeats() -> None:
    first = np.asarray([0.4, 1.1, 1.9, 2.6], dtype=np.float64)
    second = -first
    assert icc_2_1(first, second) < 0.0
