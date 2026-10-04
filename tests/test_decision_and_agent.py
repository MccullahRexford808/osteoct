"""Decision-layer, agent and model unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from osteoct.agent.features import (
    BAND_ORDER_INDEX,
    TIER_ORDER_INDEX,
    concept_vector,
    measurement_concept_vocabulary,
    rows_for_partition,
)
from osteoct.agent.pipeline import Policy, clinical_risk, policy_from_arm, run_agent, structural_risk, summarise_outcomes
from osteoct.agent.planner import Planner, PlannerConfig, build_planner, planner_report
from osteoct.agent.tools import TOOL_INPUTS, TOOL_ORDER, ToolName, ToolRegistry, tool_report
from osteoct.cohort.cohort import examination_from_record
from osteoct.decision.abstention import REASON_DESCRIPTIONS, TERMINATION_REASONS, abstain, describe, tally
from osteoct.decision.conformance import record_conformance, record_supported, summarise
from osteoct.decision.evidence import attach, chain_complete, chain_depth, close_chain, history_tokens, measurement_tokens
from osteoct.decision.operator import constrain, enumerate_admissible, repair_sequence, unconstrained
from osteoct.decision.repair import apply_repair, is_repairable, repair_loop, resolve_identifiers
from osteoct.grounding.standard import concept_key, ground_measurements
from osteoct.models.arms import ARM_SPECS, OBSERVED_ARMS, SUBSTITUTED_ARMS, arm_names, build_arm, encoder_names, is_constrained_arm
from osteoct.models.baselines import ClinicalRiskModel, ClinicalVariables, HuThresholdRule, band_one_hot, risk_probability_from_band
from osteoct.models.blocks import (
    CompactVolumeEncoder3D,
    ConvEncoder3D,
    DescriptorEncoder,
    ShiftedWindowEncoder3D,
    WindowAttention3D,
)
from osteoct.models.scorer import BAND_CLASSES, SEVERITY_CLASSES, TIER_CLASSES, ScorerSpec, features_to_tensors, zero_features
from osteoct.ontology.axioms import AxiomSet, FamilyName, Violation
from osteoct.ontology.bands import BandCalibration, band_from_interval
from osteoct.ontology.index import entry_for
from osteoct.ontology.severity import grade_from_vertebrae
from osteoct.ontology.tiers import tier_from_band
from osteoct.schema.concepts import TermSystem
from osteoct.schema.records import CandidateDecision, DecisionRecord, MeasurementBundle
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier, VertebralLevel
from osteoct.serialisation.dcm_sr import build_sr_tree, concept_sequence, leaf_count, serialise_record, serialise_sr, sr_digest
from osteoct.texture.descriptors import descriptor_names
from tests.conftest import make_batch


def attenuation_key() -> str:
    entry = entry_for("trabecular_attenuation", TermSystem.LOINC)
    return concept_key(entry.system, entry.key)


def _candidate(context: DiagnosticContext | None = None) -> CandidateDecision:
    return CandidateDecision(
        band=DiagnosticBand.NORMAL,
        tier=RiskTier.LOW,
        severity=DeformityGrade.NONE,
        concepts=(attenuation_key(),),
        evidence=(),
        probability=0.17,
        context=DiagnosticContext() if context is None else context,
    )


def test_tool_inventory_matches_the_manuscript_order() -> None:
    assert tuple(tool.value for tool in TOOL_ORDER) == (
        "LOCALISE",
        "ATTENUATION",
        "HARMONISE",
        "TEXTURE",
        "STABILITYFILTER",
        "DENSITYESTIMATE",
        "GROUNDTOSTANDARD",
        "SCORE",
        "CONSTRAIN",
        "SERIALISE",
    )
    assert set(TOOL_INPUTS) == set(TOOL_ORDER)


def test_registry_counts_calls_and_errors() -> None:
    registry = ToolRegistry()
    registry.record(ToolName.SCORE)
    registry.record(ToolName.SCORE)
    registry.record_error(ToolName.CONSTRAIN, "band")
    assert registry.summary() == {"SCORE": 2}
    assert registry.total() == 2
    assert registry.error_paths() == {"CONSTRAIN:band": 1}


def test_resolve_identifiers_repairs_a_lower_case_key(attributes: AxiomSet) -> None:
    candidate = _candidate()
    broken = CandidateDecision(
        band=candidate.band,
        tier=candidate.tier,
        severity=candidate.severity,
        concepts=("osteoct-3003",),
        evidence=(),
        probability=candidate.probability,
        context=candidate.context,
    )
    resolved = resolve_identifiers(broken, attributes.index)
    assert attributes.unresolved(resolved) == ()


def test_resolve_identifiers_leaves_unknown_tokens(attributes: AxiomSet) -> None:
    broken = CandidateDecision(
        band=DiagnosticBand.NORMAL,
        tier=RiskTier.LOW,
        severity=DeformityGrade.NONE,
        concepts=("nothing-like-this",),
        evidence=(),
        probability=0.1,
        context=DiagnosticContext(),
    )
    resolved = resolve_identifiers(broken, attributes.index)
    assert attributes.unresolved(resolved) == ("nothing-like-this",)


def test_repair_loop_follows_the_fixed_family_order(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    violating = CandidateDecision(
        band=DiagnosticBand.SEVERE_OSTEOPOROSIS,
        tier=RiskTier.VERY_HIGH,
        severity=DeformityGrade.SEVERE,
        concepts=(attenuation_key(),),
        evidence=(),
        probability=0.4,
        context=DiagnosticContext(),
    )
    trace = repair_loop(attributes, violating, measurement_bundle, 8)
    families = [step.family.value for step in trace.steps]
    order = {"band": 0, "tier": 1, "sev": 2, "ev": 3}
    assert families == sorted(families, key=lambda label: order[label])
    assert {"band", "sev", "ev"} <= set(families)


def test_repair_loop_respects_the_budget(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    violating = CandidateDecision(
        band=DiagnosticBand.SEVERE_OSTEOPOROSIS,
        tier=RiskTier.VERY_HIGH,
        severity=DeformityGrade.SEVERE,
        concepts=(attenuation_key(),),
        evidence=(),
        probability=0.4,
        context=DiagnosticContext(),
    )
    tight = repair_loop(attributes, violating, measurement_bundle, 1)
    loose = repair_loop(attributes, violating, measurement_bundle, 6)
    assert tight.repairs() == 1
    assert tight.budget_exhausted
    assert loose.repairs() > tight.repairs()


def test_apply_repair_derives_the_band_from_the_interval(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    repaired = apply_repair(FamilyName.BAND, _candidate(), measurement_bundle, attributes)
    expected = band_from_interval(
        measurement_bundle.standard_attenuation_hu, measurement_bundle.uncertainty_hu, attributes.calibration
    )
    assert repaired.band is expected


def test_apply_repair_derives_the_severity_from_morphology(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    repaired = apply_repair(FamilyName.SEVERITY, _candidate(), measurement_bundle, attributes)
    assert repaired.severity is grade_from_vertebrae(
        tuple(m.morphology for m in measurement_bundle.vertebrae), attributes.thresholds
    )


def test_apply_repair_derives_the_tier_from_the_band(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    repaired = apply_repair(FamilyName.TIER, _candidate(), measurement_bundle, attributes)
    expected = tier_from_band(
        repaired.band,
        repaired.context,
        measurement_bundle.acquisition,
        repaired.concepts,
        attributes.index,
        attributes.envelope,
    )
    assert repaired.tier is expected


def test_apply_repair_attaches_the_required_evidence(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    repaired = apply_repair(FamilyName.EVIDENCE, _candidate(), measurement_bundle, attributes)
    required = attributes.required_evidence(repaired)
    assert required
    assert set(required) <= set(repaired.evidence)


def test_evidence_family_is_soft_and_others_are_hard() -> None:
    from osteoct.ontology.axioms import FAMILY_TABLE

    soft = {name.value for name, family in FAMILY_TABLE.items() if not family.is_hard()}
    hard = {name.value for name, family in FAMILY_TABLE.items() if family.is_hard()}
    assert soft == {"ev"}
    assert hard == {"id", "band", "tier", "sev"}


def test_is_repairable_excludes_the_identity_family(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    identity_only = attributes.with_only((FamilyName.IDENTITY,))
    unresolved = CandidateDecision(
        band=DiagnosticBand.NORMAL,
        tier=RiskTier.LOW,
        severity=DeformityGrade.NONE,
        concepts=("unknown",),
        evidence=(),
        probability=0.1,
        context=DiagnosticContext(),
    )
    assert not is_repairable(identity_only, unresolved, measurement_bundle)
    assert is_repairable(attributes, _candidate(), measurement_bundle)


def test_operator_abstains_on_an_unresolved_identifier(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    unresolved = CandidateDecision(
        band=DiagnosticBand.NORMAL,
        tier=RiskTier.LOW,
        severity=DeformityGrade.NONE,
        concepts=("totally-unknown",),
        evidence=(),
        probability=0.1,
        context=DiagnosticContext(),
    )
    outcome = constrain(unresolved, measurement_bundle, attributes, 6)
    assert not outcome.emitted()
    assert outcome.reason() == "unresolved_identifier"
    assert outcome.trace.repairs() == 0


def test_operator_admits_a_repaired_record(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    outcome = constrain(_candidate(), measurement_bundle, attributes, 6)
    assert outcome.emitted()
    admitted = outcome.admitted
    assert admitted is not None
    assert 0.0 < admitted.conformance <= 1.0
    assert admitted.repairs > 0
    assert chain_complete(admitted.candidate.evidence)


def test_operator_with_no_budget_abstains_with_the_violated_family(
    attributes: AxiomSet, measurement_bundle: MeasurementBundle
) -> None:
    outcome = constrain(_candidate(), measurement_bundle, attributes, 0)
    assert not outcome.emitted()
    assert outcome.reason() in {"band", "tier", "sev", "ev", "budget_exhausted"}


def test_unconstrained_arm_is_scored_but_not_restricted(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    admitted = unconstrained(_candidate(), measurement_bundle, attributes)
    assert admitted.repairs == 0
    assert 0.0 <= admitted.conformance < 1.0


def test_repair_sequence_reports_the_families(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    sequence = repair_sequence(_candidate(), measurement_bundle, attributes, 6)
    assert sequence
    assert all(label in {"band", "tier", "sev", "ev"} for label in sequence)


def test_enumerate_admissible_filters_candidates(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    consistent = CandidateDecision(
        band=band_from_interval(
            measurement_bundle.standard_attenuation_hu, measurement_bundle.uncertainty_hu, attributes.calibration
        ),
        tier=tier_from_band(
            band_from_interval(
                measurement_bundle.standard_attenuation_hu, measurement_bundle.uncertainty_hu, attributes.calibration
            ),
            DiagnosticContext(),
            measurement_bundle.acquisition,
            (entry_for("trabecular_attenuation", TermSystem.LOINC).key,),
            attributes.index,
            attributes.envelope,
        ),
        severity=grade_from_vertebrae(tuple(m.morphology for m in measurement_bundle.vertebrae), attributes.thresholds),
        concepts=(entry_for("trabecular_attenuation", TermSystem.LOINC).key,),
        evidence=(),
        probability=0.2,
        context=DiagnosticContext(),
    )
    admissible = enumerate_admissible((consistent, _candidate()), measurement_bundle, attributes)
    assert consistent in admissible


def test_abstention_reason_registry() -> None:
    assert set(TERMINATION_REASONS) == set(REASON_DESCRIPTIONS)
    record = abstain("no_assessable_vertebra", "localise")
    assert record.reason == "no_assessable_vertebra"
    assert describe("band")
    with pytest.raises(KeyError):
        abstain("not-a-reason", "localise")


def test_abstention_tally() -> None:
    counted = tally(("band", "band", "sev"))
    assert counted.total() == 3
    assert counted.share("band", 4) == pytest.approx(0.5)


def test_evidence_tokens_include_levels_and_history() -> None:
    context = DiagnosticContext(prior_fragility_fracture=True, treatment_exposure=True, reference_t_score=-2.0)
    bundle = make_bundle_for_tokens()
    tokens = measurement_tokens(bundle, context)
    assert "m_att_star" in tokens
    assert "reference_density" in tokens
    assert "prior_fracture_state" in history_tokens(context)
    assert "treatment_exposure_state" in history_tokens(context)


def make_bundle_for_tokens() -> MeasurementBundle:
    from osteoct.schema.records import AcquisitionDescriptor, VertebraMeasurement, VertebraRegion

    region = VertebraRegion(VertebralLevel.L1, (8, 8, 4), (12, 10, 8), 240, 0.6)
    from osteoct.schema.records import MorphologyFeatures

    return MeasurementBundle(
        standard_attenuation_hu=200.0,
        uncertainty_hu=3.0,
        texture_vector=(0.1,),
        density_indicator=180.0,
        acquisition=AcquisitionDescriptor(120.0, "vendor_a", "B40f", 1.0),
        vertebrae=(VertebraMeasurement(region, MorphologyFeatures(10.0, 10.0, 10.0), 200.0, 8.0),),
    )


def test_evidence_chain_helpers() -> None:
    tokens = attach(("m_att",), ("u_att", "m_att"))
    assert tokens == ("m_att", "u_att")
    closed = close_chain(tokens, make_bundle_for_tokens())
    assert chain_complete(closed)
    assert chain_depth(closed) >= 2


def test_conformance_summary_counts_abstentions() -> None:
    from osteoct.decision.conformance import ConformanceSummary

    summary = ConformanceSummary(
        emitted=3,
        abstained=1,
        processed=4,
        mean_conformance=0.9,
        unsupported_rate=0.0,
        abstention_rate=0.25,
        family_violation_rates={"ev": 0.0},
    )
    mapping = summary.as_mapping()
    assert mapping["abstention_rate"] == 0.25
    assert mapping["violation_rate_ev"] == 0.0


def test_scorer_forward_shapes(planner: Planner) -> None:
    output = planner.forward(make_batch(5))
    assert tuple(output.band_logits.shape) == (5, BAND_CLASSES)
    assert tuple(output.tier_logits.shape) == (5, TIER_CLASSES)
    assert tuple(output.severity_logits.shape) == (5, SEVERITY_CLASSES)
    assert tuple(output.risk_logit.shape) == (5,)


def test_scorer_heads_decode_to_vocabulary(planner: Planner) -> None:
    output = planner.forward(make_batch(1))
    assert output.band(0) in BAND_ORDER_INDEX
    assert output.tier(0) in TIER_ORDER_INDEX
    assert 0.0 <= output.probability(0) <= 1.0


def test_scorer_spec_input_width_matches_assembly() -> None:
    spec = ScorerSpec(descriptor_dim=3, texture_dim=25, concept_dim=30, embedding_dim=16)
    assert spec.input_dim() == 2 + 3 + 25 + 30 + 16
    ablated = ScorerSpec(descriptor_dim=3, texture_dim=25, concept_dim=30, embedding_dim=0, use_embedding=False, use_texture=False)
    assert ablated.input_dim() == 2 + 3 + 30


def test_zero_features_preserves_declared_width() -> None:
    batch = zero_features(3, 25, 30, 16, 3)
    assert batch.size() == 3
    assert batch.width() == 3 + 25 + 30 + 16
    tensors = features_to_tensors(batch, torch.device("cpu"))
    assert tuple(tensors.concepts.shape) == (3, 30)


def test_arm_registry_covers_the_manuscript_architectures() -> None:
    assert arm_names() == tuple(ARM_SPECS)
    encoder_arms = [name for name in OBSERVED_ARMS if name != "tool_free_scorer"]
    assert len(encoder_arms) == 3
    for name in encoder_arms:
        assert ARM_SPECS[name].encoder is not None
    assert ARM_SPECS["tool_free_scorer"].encoder is None
    assert ARM_SPECS["text_only_planner"].use_texture is False
    assert encoder_names() == ("3d_conv", "shifted_window", "foundation_contract")
    assert is_constrained_arm("constrained")
    assert not is_constrained_arm("tool_free_scorer")
    assert set(SUBSTITUTED_ARMS) <= set(ARM_SPECS)


@pytest.mark.parametrize(
    "arm", ["constrained", "convolutional", "shifted_window", "foundation_encoder", "tool_free_scorer", "text_only_planner"]
)
def test_arm_builds_and_runs(arm: str) -> None:
    bundle = build_arm(arm)
    batch = make_batch(3, embedding_dim=bundle.embedding_dim)
    tensors = features_to_tensors(batch, torch.device("cpu"))
    decoded = bundle.scorer(tensors)
    assert tuple(decoded.band_logits.shape) == (3, BAND_CLASSES)


def test_encoder_shapes() -> None:
    volume = torch.zeros((2, 1, 16, 16, 16))
    for encoder in (ConvEncoder3D(1, 16, 2), ShiftedWindowEncoder3D(1, 16, 1, 4, 2), CompactVolumeEncoder3D(1, 16, 1, 2)):
        embedding = encoder(volume)
        assert tuple(embedding.shape) == (2, 16)


def test_window_attention_with_shift_is_accepted() -> None:
    module = WindowAttention3D(8, 2, 4, shift=2)
    tokens = torch.randn(3, 8, 8)
    assert tuple(module(tokens).shape) == (3, 8, 8)
    with pytest.raises(ValueError):
        WindowAttention3D(8, 2, 4, shift=9)


def test_descriptor_encoder_projection() -> None:
    encoder = DescriptorEncoder(5, 7)
    assert tuple(encoder(torch.randn(4, 5)).shape) == (4, 7)


def test_baseline_rules_are_monotone() -> None:
    calibration = BandCalibration(240.0, 45.0, 1.5)
    thresholds = grade_from_thresholds()
    rule = HuThresholdRule(calibration, thresholds)
    assert rule.band(260.0) is DiagnosticBand.NORMAL
    assert rule.band(70.0) is DiagnosticBand.OSTEOPOROSIS
    assert rule.tier(DiagnosticBand.OSTEOPENIA, DiagnosticContext(prior_fragility_fracture=True)) is RiskTier.VERY_HIGH
    assert risk_probability_from_band(DiagnosticBand.OSTEOPOROSIS) > risk_probability_from_band(DiagnosticBand.NORMAL)
    assert float(band_one_hot(DiagnosticBand.OSTEOPOROSIS).sum()) == 1.0


def grade_from_thresholds() -> object:
    from osteoct.ontology.severity import GenantThresholds

    return GenantThresholds()


def test_clinical_risk_model_is_monotone_in_age() -> None:
    coefficients = np.asarray([0.03, 0.4, -0.02, 0.8, 0.2, 0.5], dtype=np.float64)
    model = ClinicalRiskModel(coefficients=coefficients, intercept=-3.0)
    young = ClinicalVariables(45.0, True, 26.0, False, False, False)
    old = ClinicalVariables(80.0, True, 26.0, False, False, False)
    assert model.probability(old) > model.probability(young)


def test_policy_from_arm_and_labels() -> None:
    assert policy_from_arm("constrained") is Policy.CONSTRAINED
    assert policy_from_arm("unconstrained") is Policy.UNCONSTRAINED
    assert policy_from_arm("regularised_only") is Policy.REGULARISED
    assert Policy.CONSTRAINED.enforces_operator()
    assert not Policy.UNCONSTRAINED.enforces_operator()
    assert Policy.CONSTRAINED.label().startswith("OSTEO-CT")


def test_structural_risk_rises_with_severity() -> None:
    low = structural_risk(DiagnosticBand.OSTEOPENIA, RiskTier.MODERATE, DeformityGrade.NONE, False, False)
    high = structural_risk(DiagnosticBand.OSTEOPENIA, RiskTier.MODERATE, DeformityGrade.SEVERE, False, False)
    assert high > low


def test_clinical_risk_is_bounded() -> None:
    for probability in (0.0, 0.5, 1.0):
        value = clinical_risk(
            probability, DiagnosticBand.SEVERE_OSTEOPOROSIS, RiskTier.VERY_HIGH, DeformityGrade.SEVERE, DiagnosticContext()
        )
        assert 0.0 < value <= 0.99


def test_summarise_outcomes_over_an_empty_run() -> None:
    log = summarise_outcomes(())
    assert log.values["processed"] == 0.0
    assert log.values["mean_conformance"] == 0.0


def test_concept_vector_marks_the_grounded_entries() -> None:
    vocabulary = measurement_concept_vocabulary()
    grounding = ground_measurements((), 200.0, 3.0, 180.0)
    vector = concept_vector(grounding, vocabulary)
    assert float(vector.sum()) >= 4.0
    assert int(vector.size) == len(vocabulary)


def test_planner_report_is_serialisable(planner: Planner) -> None:
    report = planner_report(planner)
    assert report["structure"]["arm"] == "constrained"
    assert isinstance(report["configuration"], dict)


def test_build_planner_defaults_to_the_constrained_arm() -> None:
    planner = build_planner()
    assert planner.bundle.name == "constrained"
    assert planner.bundle.encoder is not None


def test_planner_state_round_trips(planner: Planner) -> None:
    state = planner.state()
    other = Planner(PlannerConfig(arm="constrained"))
    other.load_state(state)
    planner.eval()
    other.eval()
    first = planner.forward(make_batch(2))
    second = other.forward(make_batch(2))
    assert torch.allclose(first.band_logits, second.band_logits)
    assert torch.allclose(first.risk_logit, second.risk_logit)


def test_encoder_encoding_shape(planner: Planner) -> None:
    volumes = np.zeros((2, 8, 8, 16), dtype=np.float32)
    embedding = planner.encode(volumes)
    assert embedding.shape == (2, planner.bundle.embedding_dim)


def test_encode_cohort_produces_consistent_widths(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    encoded = smoke_study.encoded
    assert encoded.texture.shape[1] == len(descriptor_names())
    assert encoded.concepts.shape[1] == len(measurement_concept_vocabulary())
    assert encoded.descriptors.shape[1] == 3
    assert rows_for_partition(encoded, "external").size >= 0


def test_run_agent_returns_a_record_or_an_abstention(smoke_study: object, planner: Planner, registry: ToolRegistry) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")[:1]
    assert rows.size
    row = int(rows[0])
    fixture_records = smoke_study.fixture.records()
    examination = examination_from_record(smoke_study.encoded.records[row], smoke_study.fixture.volume_for(fixture_records[row]))
    outcome = run_agent(
        examination, smoke_study.toolbox, planner, smoke_study.attributes, Policy.CONSTRAINED, smoke_study.budget, registry
    )
    assert outcome.emitted() or outcome.abstention is not None
    if outcome.emitted():
        assert outcome.serialised
        assert 0.0 < outcome.conformance <= 1.0
    assert tool_report(registry)["total_calls"] > 0


def test_serialisation_tree_carries_every_binding(measurement_bundle: MeasurementBundle) -> None:
    record = DecisionRecord(
        record_id="X-1",
        band=DiagnosticBand.OSTEOPENIA,
        tier=RiskTier.MODERATE,
        severity=DeformityGrade.MODERATE,
        conformance=0.9,
        bindings=ground_measurements((), 200.0, 3.0, 180.0).bindings,
        evidence=("m_att",),
        repairs=2,
    )
    tree = build_sr_tree(record, measurement_bundle)
    text = serialise_sr(tree)
    assert leaf_count(tree) >= 10
    assert "CONTAINER" in text
    assert text == serialise_record(record, measurement_bundle)
    assert sr_digest(text) == sr_digest(serialise_record(record, measurement_bundle))
    assert "DICOM SR:OSTEOCT-4001" in concept_sequence(tree)


def test_violation_label() -> None:
    violation = Violation(FamilyName.BAND, "detail")
    assert violation.label() == "band"


def test_record_conformance_and_support(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    candidate = _candidate()
    repaired = apply_repair(FamilyName.EVIDENCE, candidate, measurement_bundle, attributes)
    assert record_supported(attributes, repaired, measurement_bundle)
    assert 0.0 < record_conformance(attributes, repaired, measurement_bundle) <= 1.0


def test_summarise_counts_family_violations(attributes: AxiomSet, measurement_bundle: MeasurementBundle) -> None:
    candidate = apply_repair(FamilyName.EVIDENCE, _candidate(), measurement_bundle, attributes)
    record = DecisionRecord(
        record_id="Y-1",
        band=candidate.band,
        tier=candidate.tier,
        severity=candidate.severity,
        conformance=record_conformance(attributes, candidate, measurement_bundle),
        bindings=(),
        evidence=candidate.evidence,
        repairs=1,
    )
    summary = summarise(attributes, (record,), (candidate,), (measurement_bundle,), 3)
    assert summary.emitted == 1
    assert summary.abstained == 2
    assert 0.0 <= summary.abstention_rate <= 1.0
