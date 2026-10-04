"""The end-to-end agent of Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
import torch

from osteoct.agent.features import measurement_concept_vocabulary
from osteoct.agent.planner import Planner
from osteoct.agent.tools import Toolbox, ToolName, ToolRegistry
from osteoct.decision.abstention import abstain
from osteoct.decision.evidence import history_tokens, measurement_tokens
from osteoct.decision.repair import RepairTrace
from osteoct.models.scorer import ScorerBatch
from osteoct.ontology.axioms import AxiomSet
from osteoct.ontology.tiers import frax_probability
from osteoct.schema.records import (
    Abstention,
    CandidateDecision,
    DecisionRecord,
    Examination,
    MeasurementBundle,
    RunLog,
)
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier
from osteoct.texture.descriptors import descriptor_names

SEVERITY_BONUS: dict[DeformityGrade, float] = {
    DeformityGrade.NONE: 1.0,
    DeformityGrade.MILD: 1.12,
    DeformityGrade.MODERATE: 1.42,
    DeformityGrade.SEVERE: 1.85,
}

MODEL_RISK_BLEND: float = 0.5


class Policy(str, Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"
    REGULARISED = "regularised"

    def enforces_operator(self) -> bool:
        return self is Policy.CONSTRAINED

    def label(self) -> str:
        return {
            Policy.CONSTRAINED: "OSTEO-CT (constrained)",
            Policy.UNCONSTRAINED: "Tool-using agent, no ontology layer",
            Policy.REGULARISED: "Hard regulariser, same axioms",
        }[self]


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    record: DecisionRecord | None
    abstention: Abstention | None
    candidate: CandidateDecision | None
    bundle: MeasurementBundle | None
    conformance: float
    repairs: int
    repair_sequence: tuple[str, ...]
    risk_score: float
    serialised: str

    def emitted(self) -> bool:
        return self.record is not None

    def reason(self) -> str:
        return "" if self.abstention is None else self.abstention.reason


def structural_risk(
    band: DiagnosticBand,
    tier: RiskTier,
    severity: DeformityGrade,
    prior_fragility_fracture: bool,
    treatment_exposure: bool,
) -> float:
    base = frax_probability(band, tier, prior_fragility_fracture, treatment_exposure)
    return float(min(0.98, base * SEVERITY_BONUS[severity]))


def clinical_risk(
    model_probability: float, band: DiagnosticBand, tier: RiskTier, severity: DeformityGrade, context: DiagnosticContext
) -> float:
    prior_fracture, treatment_exposure = context.as_flags()
    structural = structural_risk(band, tier, severity, prior_fracture, treatment_exposure)
    blended = MODEL_RISK_BLEND * model_probability + (1.0 - MODEL_RISK_BLEND) * structural
    return float(min(0.99, max(0.001, blended)))


def build_candidate_batch(
    planner: Planner,
    toolbox: Toolbox,
    bundle: MeasurementBundle,
    concepts: tuple[str, ...],
    context_vector: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float32],
    embedding: npt.NDArray[np.float64],
) -> ScorerBatch:
    vocabulary = measurement_concept_vocabulary()
    position = {key: index for index, key in enumerate(vocabulary)}
    concept_row = np.zeros((1, len(vocabulary)), dtype=np.float64)
    for key in concepts:
        slot = position.get(key)
        if slot is not None:
            concept_row[0, slot] = 1.0
    texture_row = np.zeros((1, len(descriptor_names())), dtype=np.float64)
    values = dict(zip(bundle.retained_texture_names, bundle.texture_vector, strict=False))
    for slot, name in enumerate(descriptor_names()):
        texture_row[0, slot] = values.get(name, 0.0)
    del volume
    return ScorerBatch(
        descriptors=np.asarray(
            [[bundle.standard_attenuation_hu, bundle.uncertainty_hu, bundle.density_indicator]], dtype=np.float64
        ),
        texture=texture_row,
        concepts=concept_row,
        context=context_vector.reshape(1, -1),
        embedding=embedding.reshape(1, -1),
        band_target=np.zeros(1, dtype=np.int64),
        tier_target=np.zeros(1, dtype=np.int64),
        severity_target=np.zeros(1, dtype=np.int64),
        risk_target=np.zeros(1, dtype=np.float64),
    )


def run_agent(
    examination: Examination,
    toolbox: Toolbox,
    planner: Planner,
    attributes: AxiomSet,
    policy: Policy,
    budget: int,
    registry: ToolRegistry,
) -> AgentOutcome:
    volume = np.asarray(examination.volume, dtype=np.float32)
    regions, morphologies = toolbox.localise(volume, registry)
    if not regions:
        return _empty(abstain("no_assessable_vertebra", "localise"))
    measurements, pooled_mean, pooled_sd, voxels = toolbox.measure_attenuation(volume, regions, morphologies, registry)
    harmonised = toolbox.harmonise(pooled_mean, pooled_sd, voxels, examination.acquisition, registry)
    vector, names = toolbox.extract_texture(volume, regions, registry)
    vector, names = toolbox.stability_applied(vector, names, registry)
    density = toolbox.density_estimate(
        harmonised.standard_hu,
        harmonised.half_width_hu,
        vector,
        names,
        examination.acquisition,
        registry,
    )
    measurement_grounding = toolbox.ground_measurements(
        regions,
        harmonised.standard_hu,
        harmonised.half_width_hu,
        density,
        registry,
    )
    bundle = MeasurementBundle(
        standard_attenuation_hu=harmonised.standard_hu,
        uncertainty_hu=harmonised.half_width_hu,
        texture_vector=vector,
        density_indicator=density,
        acquisition=examination.acquisition,
        vertebrae=measurements,
        retained_texture_names=names,
    )
    context = examination.context
    context_vector = np.asarray(
        [1.0 if context.prior_fragility_fracture else 0.0, 1.0 if context.treatment_exposure else 0.0],
        dtype=np.float64,
    )
    embedding = planner.encode_one(volume).reshape(-1)
    registry.record(ToolName.SCORE)
    batch = build_candidate_batch(planner, toolbox, bundle, measurement_grounding.concepts, context_vector, volume, embedding)
    with torch.no_grad():
        output = planner.forward(batch)
    evidence = measurement_tokens(bundle, context) + history_tokens(context)
    candidate = CandidateDecision(
        band=output.band(0),
        tier=output.tier(0),
        severity=output.severity(0),
        concepts=measurement_grounding.concepts,
        evidence=tuple(dict.fromkeys(evidence)),
        probability=output.probability(0),
        context=context,
    )
    outcome = toolbox.operate(candidate, bundle, attributes, budget, registry, enforced=policy.enforces_operator())
    if not outcome.emitted():
        return _empty(outcome.abstention, bundle=bundle, candidate=candidate, trace=outcome.trace)
    admitted = outcome.admitted
    assert admitted is not None
    grounding = toolbox.ground_decision(
        measurement_grounding,
        admitted.candidate.band,
        admitted.candidate.tier,
        admitted.candidate.severity,
        context,
        registry,
    )
    record = DecisionRecord(
        record_id=examination.record_id,
        band=admitted.candidate.band,
        tier=admitted.candidate.tier,
        severity=admitted.candidate.severity,
        conformance=admitted.conformance,
        bindings=grounding.bindings,
        evidence=admitted.candidate.evidence,
        repairs=admitted.repairs,
    )
    serialised = toolbox.serialise(record, bundle, registry)
    risk = clinical_risk(admitted.candidate.probability, record.band, record.tier, record.severity, context)
    return AgentOutcome(
        record=record,
        abstention=None,
        candidate=admitted.candidate,
        bundle=bundle,
        conformance=admitted.conformance,
        repairs=admitted.repairs,
        repair_sequence=tuple(step.family.value for step in outcome.trace.steps),
        risk_score=risk,
        serialised=serialised,
    )


def run_cohort(
    examinations: tuple[Examination, ...],
    toolbox: Toolbox,
    planner: Planner,
    attributes: AxiomSet,
    policy: Policy,
    budget: int,
    registry: ToolRegistry,
) -> tuple[AgentOutcome, ...]:
    return tuple(run_agent(examination, toolbox, planner, attributes, policy, budget, registry) for examination in examinations)


def summarise_outcomes(outcomes: tuple[AgentOutcome, ...]) -> RunLog:
    log = RunLog(seeds=())
    emitted = [outcome for outcome in outcomes if outcome.emitted()]
    log.add("processed", float(len(outcomes)))
    log.add("emitted", float(len(emitted)))
    log.add("abstained", float(len(outcomes) - len(emitted)))
    log.add("mean_conformance", float(np.mean([outcome.conformance for outcome in emitted])) if emitted else 0.0)
    log.add("mean_repairs", float(np.mean([outcome.repairs for outcome in emitted])) if emitted else 0.0)
    log.add("mean_risk", float(np.mean([outcome.risk_score for outcome in emitted])) if emitted else 0.0)
    return log


def _empty(
    abstention: Abstention | None,
    bundle: MeasurementBundle | None = None,
    candidate: CandidateDecision | None = None,
    trace: RepairTrace | None = None,
) -> AgentOutcome:
    repairs = 0 if trace is None else trace.repairs()
    sequence = () if trace is None else tuple(step.family.value for step in trace.steps)
    resolved = abstention if abstention is not None else abstain("budget_exhausted", "operate")
    return AgentOutcome(
        record=None,
        abstention=resolved,
        candidate=candidate,
        bundle=bundle,
        conformance=0.0,
        repairs=repairs,
        repair_sequence=sequence,
        risk_score=0.5,
        serialised="",
    )


def policy_from_arm(arm: str) -> Policy:
    if arm == "regularised_only":
        return Policy.REGULARISED
    if arm == "unconstrained":
        return Policy.UNCONSTRAINED
    return Policy.CONSTRAINED
