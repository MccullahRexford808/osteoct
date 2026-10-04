"""Study assembly: configuration to cohort fixture, ontology, toolbox and arm execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from osteoct.acquisition.attenuation import AttenuationParams
from osteoct.acquisition.density import DensityModel
from osteoct.acquisition.harmonise import HarmonisationModel
from osteoct.agent.features import EncodedCohort, encode_cohort, rows_for_partition
from osteoct.agent.pipeline import Policy, clinical_risk
from osteoct.agent.planner import Planner, PlannerConfig
from osteoct.agent.tools import Toolbox, ToolRegistry
from osteoct.cohort.fixture import CohortFixture, FixtureSpec, build_cohort_fixture
from osteoct.decision.abstention import describe
from osteoct.decision.evidence import history_tokens, measurement_tokens
from osteoct.localisation.morphology import MorphometryParams
from osteoct.localisation.vertebra import LocalisationParams
from osteoct.models.scorer import ScorerBatch
from osteoct.ontology.axioms import FAMILY_ORDER, AxiomSet, FamilyName, supported_code
from osteoct.ontology.bands import BandCalibration, band_from_t_score
from osteoct.ontology.index import build_concept_index
from osteoct.ontology.severity import GenantThresholds
from osteoct.ontology.tiers import ProtocolEnvelope
from osteoct.schema.records import CandidateDecision
from osteoct.schema.vocabulary import DiagnosticBand, RiskTier
from osteoct.texture.descriptors import TextureParams
from osteoct.training.seeds import set_seed
from osteoct.training.trainer import TrainConfig, TrainingResult, train_arm
from osteoct.utils.config import ExperimentConfig

PARTITION_DEVELOPMENT = "development"
PARTITION_INTERNAL = "internal_test"
PARTITION_EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class ArmRecord:
    row: int
    record_id: str
    patient_id: str
    site: str
    band: int
    tier: int
    severity: int
    risk: float
    conformance: float
    emitted: bool
    unsupported: bool
    repairs: int
    reason: str
    violated_families: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "band": self.band,
            "tier": self.tier,
            "severity": self.severity,
            "risk": round(self.risk, 6),
            "conformance": round(self.conformance, 6),
            "emitted": self.emitted,
            "repairs": self.repairs,
            "reason": self.reason,
            "violated_families": list(self.violated_families),
        }


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    policy: Policy
    rows: npt.NDArray[np.int64]
    per_seed: tuple[tuple[ArmRecord, ...], ...]
    ensemble: tuple[ArmRecord, ...]

    def emitted(self) -> tuple[ArmRecord, ...]:
        return tuple(record for record in self.ensemble if record.emitted)

    def conformance_per_seed(self) -> tuple[float, ...]:
        values: list[float] = []
        for records in self.per_seed:
            emitted = [record.conformance for record in records if record.emitted]
            values.append(float(np.mean(emitted)) if emitted else 0.0)
        return tuple(values)

    def unsupported_per_seed(self) -> tuple[float, ...]:
        values: list[float] = []
        for records in self.per_seed:
            emitted = [record for record in records if record.emitted]
            share = float(np.mean([1.0 if record.unsupported else 0.0 for record in emitted])) if emitted else 0.0
            values.append(share)
        return tuple(values)

    def abstention_per_seed(self) -> tuple[float, ...]:
        return tuple(
            float(np.mean([0.0 if record.emitted else 1.0 for record in records])) if records else 0.0 for records in self.per_seed
        )

    def mean_repairs(self) -> float:
        emitted = self.emitted()
        return float(np.mean([record.repairs for record in emitted])) if emitted else 0.0

    def reasons(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for record in self.ensemble:
            if not record.emitted:
                tally[record.reason] = tally.get(record.reason, 0) + 1
        return tally

    def summary(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "policy": self.policy.value,
            "records": int(self.rows.size),
            "emitted": len(self.emitted()),
            "conformance": _mean_sd(self.conformance_per_seed()),
            "unsupported": _mean_sd(self.unsupported_per_seed()),
            "abstention": _mean_sd(self.abstention_per_seed()),
            "mean_repairs": round(self.mean_repairs(), 6),
            "abstention_reasons": self.reasons(),
        }


def _mean_sd(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0}
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


@dataclass(frozen=True, slots=True)
class StudyContext:
    fixture: CohortFixture
    attributes: AxiomSet
    toolbox: Toolbox
    encoded: EncodedCohort
    registry: ToolRegistry
    budget: int
    training: dict[str, TrainingResult] = field(default_factory=dict)

    def rows(self, partition: str) -> npt.NDArray[np.int64]:
        return rows_for_partition(self.encoded, partition)

    def patient_ids(self, rows: npt.NDArray[np.int64]) -> tuple[str, ...]:
        return tuple(self.encoded.records[int(row)].patient_id for row in rows.tolist())


def axiom_set_from_config(config: ExperimentConfig) -> AxiomSet:
    section = config.section("ontology")
    calibration = BandCalibration(
        reference_mean_hu=float(section["calibration"]["reference_mean_hu"]),
        reference_sd_hu=float(section["calibration"]["reference_sd_hu"]),
        within_scan_floor_hu=float(section["calibration"]["within_scan_floor_hu"]),
    )
    thresholds = GenantThresholds(
        fracture_floor=float(section["genant"]["fracture_floor"]),
        moderate_floor=float(section["genant"]["moderate_floor"]),
        severe_floor=float(section["genant"]["severe_floor"]),
    )
    envelope = ProtocolEnvelope(
        maximum_slice_thickness_mm=float(section["envelope"]["maximum_slice_thickness_mm"]),
        minimum_tube_voltage_kvp=float(section["envelope"]["minimum_tube_voltage_kvp"]),
        accepted_kernels=tuple(str(value) for value in section["envelope"]["accepted_kernels"]),
    )
    index = build_concept_index()
    active = frozenset(FamilyName(label) for label in section.get("active_families", [family.value for family in FAMILY_ORDER]))
    return AxiomSet(index=index, calibration=calibration, thresholds=thresholds, envelope=envelope, active=active)


def toolbox_from_config(config: ExperimentConfig, attributes: AxiomSet) -> Toolbox:
    acquisition = config.section("acquisition")
    cohort = config.section("cohort")
    texture = cohort["texture"]
    localisation = LocalisationParams(
        bone_window_low_hu=float(cohort["bone_window_low_hu"]),
        roi_half_extent=(
            int(cohort["roi_half_extent"][0]),
            int(cohort["roi_half_extent"][1]),
            int(cohort["roi_half_extent"][2]),
        ),
        level_span=int(cohort["level_span"]),
    )
    attenuation = AttenuationParams(
        cortical_erosion_voxels=int(cohort["cortical_erosion_voxels"]),
        endplate_exclusion_fraction=float(cohort["endplate_exclusion_fraction"]),
        minimum_trabecular_voxels=int(cohort["minimum_trabecular_voxels"]),
        marrow_fat_correction_hu=float(cohort["marrow_fat_correction_hu"]),
        bone_threshold_hu=float(cohort["bone_window_low_hu"]),
    )
    morphometry = MorphometryParams(
        anterior_share=float(cohort["anterior_share"]),
        posterior_share=float(cohort["posterior_share"]),
        reference_column_fraction=float(cohort["reference_column_fraction"]),
    )
    texture_params = TextureParams(
        grey_levels=int(texture["grey_levels"]),
        window_low_hu=float(texture["window_low_hu"]),
        window_high_hu=float(texture["window_high_hu"]),
        wavelet_levels=int(texture["wavelet_levels"]),
        stability_threshold=float(texture["stability_threshold"]),
    )
    harmonisation = HarmonisationModel(
        voltage_slope_hu_per_kvp=float(acquisition["voltage_slope_hu_per_kvp"]),
        reference_voltage_kvp=float(acquisition["reference_voltage_kvp"]),
        sampling_weight=float(acquisition["sampling_weight"]),
        within_scan_floor_hu=float(acquisition["within_scan_floor_hu"]),
        kernel_spread_hu=float(acquisition["kernel_spread_hu"]),
    )
    density = DensityModel(
        intercept=float(acquisition["density"]["intercept"]),
        attenuation_gain=float(acquisition["density"]["attenuation_gain"]),
        uncertainty_penalty=float(acquisition["density"]["uncertainty_penalty"]),
        voltage_gain=float(acquisition["density"]["voltage_gain"]),
        kernel_gain=float(acquisition["density"]["kernel_gain"]),
        texture_weights={str(key): float(value) for key, value in acquisition["density"]["texture_weights"].items()},
        texture_scale=float(acquisition["density"]["texture_scale"]),
    )
    return Toolbox(
        localisation=localisation,
        attenuation=attenuation,
        morphometry=morphometry,
        texture=texture_params,
        harmonisation=harmonisation,
        density=density,
        index=attributes.index,
        stability=None,
    )


def fixture_from_config(config: ExperimentConfig, toolbox: Toolbox) -> CohortFixture:
    section = config.section("cohort")
    spec = FixtureSpec(
        sites=tuple(str(value) for value in section.get("sites", ["S1", "S2", "S3"])),
        regions=tuple(str(value) for value in section.get("regions", ["R1", "R2"])),
        examinations_per_site=int(section.get("examinations_per_site", 48)),
        bodies=int(section.get("bodies", 13)),
        slices_per_body=int(section.get("slices_per_body", 10)),
        slice_gap=int(section.get("slice_gap", 2)),
        rows=int(section.get("rows", 24)),
        columns=int(section.get("columns", 24)),
        development_share=float(section.get("development_share", 0.68)),
        attenuation_mean_hu=float(section.get("attenuation_mean_hu", 190.0)),
        attenuation_sd_hu=float(section.get("attenuation_sd_hu", 42.0)),
        dissociation_share=float(section.get("dissociation_share", 0.35)),
        dissociation_hazard_ratio=float(section.get("dissociation_hazard_ratio", 2.4)),
        seed=int(section.get("fixture_seed", 20260928)),
    )
    return build_cohort_fixture(
        spec,
        localisation=toolbox.localisation,
        morphometry=toolbox.morphometry,
        harmonisation=toolbox.harmonisation,
    )


def train_config_from_config(config: ExperimentConfig) -> TrainConfig:
    section = config.section("train")
    from osteoct.training.optim import OptimConfig
    from osteoct.training.precision import PrecisionConfig
    from osteoct.training.schedule import ScheduleConfig

    return TrainConfig(
        seeds=tuple(int(value) for value in section["seeds"]),
        epochs=int(section["epochs"]),
        batch_size=int(section["batch_size"]),
        grad_accum=int(section["grad_accum"]),
        world_size=int(section.get("world_size", 1)),
        optimizer=OptimConfig(
            name=str(section["optimizer"]["name"]),
            learning_rate=float(section["optimizer"]["learning_rate"]),
            encoder_learning_rate=float(section["optimizer"].get("encoder_learning_rate", 1e-4)),
            weight_decay=float(section["optimizer"]["weight_decay"]),
            gradient_clip=float(section["optimizer"]["gradient_clip"]),
        ),
        schedule=ScheduleConfig(
            name=str(section["schedule"]["name"]),
            warmup_steps=int(section["schedule"]["warmup_steps"]),
            total_steps=int(section["schedule"]["total_steps"]),
            minimum_ratio=float(section["schedule"].get("minimum_ratio", 0.02)),
        ),
        precision=PrecisionConfig(name=str(section["precision"]["name"])),
        ema_decay=float(section.get("ema_decay", 0.999)),
        early_stopping_patience=int(section.get("early_stopping_patience", 12)),
        regulariser_scale=float(section.get("regulariser_scale", 0.0)),
        train_encoder=bool(section.get("train_encoder", True)),
        validation_fraction=float(section.get("validation_fraction", 0.22)),
    )


def build_study(config: ExperimentConfig, run_directory: Path | None = None) -> StudyContext:
    set_seed(int(config.get("cohort.fixture_seed", 0)))
    attributes = axiom_set_from_config(config)
    toolbox = toolbox_from_config(config, attributes)
    fixture = fixture_from_config(config, toolbox)
    registry = ToolRegistry()
    records = fixture.records()
    volumes = fixture.volumes()
    planner = Planner(PlannerConfig(arm=str(config.get("arm", config.get("model.arm", "constrained")))))
    encoded = encode_cohort(records, np.asarray(volumes, dtype=np.float32), toolbox, planner, attributes, registry)
    budget = int(config.get("pipeline.constrain_budget", 6))
    return StudyContext(
        fixture=fixture,
        attributes=attributes,
        toolbox=toolbox,
        encoded=encoded,
        registry=registry,
        budget=budget,
    )


def train_arms(
    context: StudyContext,
    config: ExperimentConfig,
    arms: tuple[str, ...],
    run_directory: Path | None = None,
) -> dict[str, TrainingResult]:
    train_config = train_config_from_config(config)
    results: dict[str, TrainingResult] = {}
    for arm in arms:
        if arm in RULE_ARMS or arm in results:
            continue
        directory = None if run_directory is None else run_directory / arm
        results[arm] = train_arm(context.encoded, arm, train_config, context.attributes, directory)
    return results


def candidate_for_arm(
    context: StudyContext,
    planner: Planner,
    row: int,
) -> tuple[CandidateDecision, float]:
    record = context.encoded.records[row]
    grounding = context.encoded.groundings[row]
    bundle = context.encoded.bundles[row]
    batch = ScorerBatch(
        descriptors=context.encoded.descriptors[row : row + 1],
        texture=context.encoded.texture[row : row + 1],
        concepts=context.encoded.concepts[row : row + 1],
        context=context.encoded.context[row : row + 1],
        embedding=context.encoded.embedding[row : row + 1],
        band_target=context.encoded.band_target[row : row + 1],
        tier_target=context.encoded.tier_target[row : row + 1],
        severity_target=context.encoded.severity_target[row : row + 1],
        risk_target=context.encoded.risk_target[row : row + 1],
    )
    planner.eval()
    with torch.no_grad():
        output = planner.forward(batch)
    evidence = measurement_tokens(bundle, record.context) + history_tokens(record.context)
    candidate = CandidateDecision(
        band=output.band(0),
        tier=output.tier(0),
        severity=output.severity(0),
        concepts=grounding.concepts,
        evidence=tuple(dict.fromkeys(evidence)),
        probability=output.probability(0),
        context=record.context,
    )
    return (candidate, output.probability(0))


def execute_arm(
    context: StudyContext,
    arm: str,
    policy: Policy,
    rows: npt.NDArray[np.int64],
    training: TrainingResult,
    axioms: AxiomSet | None = None,
) -> ArmResult:
    active = context.attributes if axioms is None else axioms
    per_seed: list[tuple[ArmRecord, ...]] = []
    ensemble_records: list[ArmRecord] = []
    for run in training.runs:
        planner = Planner(PlannerConfig(arm=arm))
        planner.load_state(run.state)
        per_seed.append(_records_for_run(context, planner, policy, rows, active))
    if not per_seed:
        return ArmResult(arm=arm, policy=policy, rows=rows, per_seed=(), ensemble=())
    for position, row in enumerate(rows.tolist()):
        seed_records = [records[position] for records in per_seed]
        emitted = [record for record in seed_records if record.emitted]
        if not emitted:
            ensemble_records.append(seed_records[0])
            continue
        ensemble_records.append(
            ArmRecord(
                row=row,
                record_id=seed_records[0].record_id,
                patient_id=seed_records[0].patient_id,
                site=seed_records[0].site,
                band=_modal([record.band for record in emitted]),
                tier=_modal([record.tier for record in emitted]),
                severity=_modal([record.severity for record in emitted]),
                risk=float(np.mean([record.risk for record in emitted])),
                conformance=float(np.mean([record.conformance for record in emitted])),
                emitted=True,
                unsupported=bool(any(record.unsupported for record in emitted)),
                repairs=_modal([record.repairs for record in emitted]),
                reason="",
                violated_families=tuple(sorted({family for record in emitted for family in record.violated_families})),
            )
        )
    return ArmResult(arm=arm, policy=policy, rows=rows, per_seed=tuple(per_seed), ensemble=tuple(ensemble_records))


def _records_for_run(
    context: StudyContext,
    planner: Planner,
    policy: Policy,
    rows: npt.NDArray[np.int64],
    axioms: AxiomSet,
) -> tuple[ArmRecord, ...]:
    records: list[ArmRecord] = []
    for row in rows.tolist():
        candidate, _ = candidate_for_arm(context, planner, row)
        bundle = context.encoded.bundles[row]
        source = context.encoded.records[row]
        outcome = context.toolbox.operate(
            candidate, bundle, axioms, context.budget, context.registry, enforced=policy.enforces_operator()
        )
        if outcome.emitted() and outcome.admitted is not None:
            admitted = outcome.admitted
            unsupported = not supported_code(admitted.candidate, axioms, bundle)
            risk = clinical_risk(
                admitted.candidate.probability,
                admitted.candidate.band,
                admitted.candidate.tier,
                admitted.candidate.severity,
                source.context,
            )
            records.append(
                ArmRecord(
                    row=row,
                    record_id=source.record_id,
                    patient_id=source.patient_id,
                    site=source.site,
                    band=_band_index(admitted.candidate.band),
                    tier=_tier_index(admitted.candidate.tier),
                    severity=int(admitted.candidate.severity.value),
                    risk=risk,
                    conformance=admitted.conformance,
                    emitted=True,
                    unsupported=unsupported,
                    repairs=admitted.repairs,
                    reason="",
                    violated_families=tuple(
                        sorted({violation.family.value for violation in axioms.violations(admitted.candidate, bundle)})
                    ),
                )
            )
            continue
        reason = outcome.reason() or "budget_exhausted"
        records.append(
            ArmRecord(
                row=row,
                record_id=source.record_id,
                patient_id=source.patient_id,
                site=source.site,
                band=-1,
                tier=-1,
                severity=-1,
                risk=0.5,
                conformance=0.0,
                emitted=False,
                unsupported=True,
                repairs=outcome.trace.repairs(),
                reason=reason,
            )
        )
    return tuple(records)


def _modal(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return int(best[0])


def _band_index(band: object) -> int:
    from osteoct.agent.features import BAND_ORDER_INDEX
    from osteoct.schema.vocabulary import DiagnosticBand

    assert isinstance(band, DiagnosticBand)
    return BAND_ORDER_INDEX[band]


def _tier_index(tier: object) -> int:
    from osteoct.agent.features import TIER_ORDER_INDEX
    from osteoct.schema.vocabulary import RiskTier

    assert isinstance(tier, RiskTier)
    return TIER_ORDER_INDEX[tier]


def reason_text(reason: str) -> str:
    return describe(reason)


RULE_ARMS: tuple[str, ...] = ("hu_threshold_rule", "frax_no_bmd", "clinical_variable_model")


def rule_candidate(context: StudyContext, row: int, arm: str) -> CandidateDecision:
    from osteoct.decision.evidence import history_tokens, measurement_tokens
    from osteoct.models.baselines import ClinicalRiskModel, ClinicalVariables, HuThresholdRule, risk_probability_from_band
    from osteoct.ontology.tiers import frax_probability

    record = context.encoded.records[row]
    bundle = context.encoded.bundles[row]
    grounding = context.encoded.groundings[row]
    calibration = context.attributes.calibration
    thresholds = context.attributes.thresholds
    rule = HuThresholdRule(calibration, thresholds)
    if arm == "hu_threshold_rule":
        band = rule.band(bundle.standard_attenuation_hu)
        tier = rule.tier(band, record.context)
        severity = rule.severity(bundle)
        probability = risk_probability_from_band(band)
    elif arm == "frax_no_bmd":
        prior_fracture, treatment_exposure = record.context.as_flags()
        base = ClinicalVariables(
            age_years=record.age_years,
            female=record.female,
            body_mass_index=record.body_mass_index,
            prior_fragility_fracture=prior_fracture,
            treatment_exposure=treatment_exposure,
            glucocorticoid_exposure=record.glucocorticoid_exposure,
        )
        probability = frax_probability(
            _band_from_probability(base, calibration),
            RiskTier.MODERATE,
            prior_fracture,
            treatment_exposure,
        )
        band = _band_from_probability(base, calibration)
        tier = rule.tier(band, record.context)
        severity = rule.severity(bundle)
    else:
        base = ClinicalVariables(
            age_years=record.age_years,
            female=record.female,
            body_mass_index=record.body_mass_index,
            prior_fragility_fracture=record.context.prior_fragility_fracture,
            treatment_exposure=record.context.treatment_exposure,
            glucocorticoid_exposure=record.glucocorticoid_exposure,
        )
        model = ClinicalRiskModel(coefficients=CLINICAL_COEFFICIENTS, intercept=CLINICAL_INTERCEPT)
        probability = model.probability(base)
        band = model.band(base, calibration)
        tier = rule.tier(band, record.context)
        severity = rule.severity(bundle)
    decision = context.toolbox.ground_decision(grounding, band, tier, severity, record.context, context.registry)
    evidence = measurement_tokens(bundle, record.context) + history_tokens(record.context)
    return CandidateDecision(
        band=band,
        tier=tier,
        severity=severity,
        concepts=decision.concepts,
        evidence=tuple(dict.fromkeys(evidence)),
        probability=float(probability),
        context=record.context,
    )


CLINICAL_COEFFICIENTS = np.asarray([0.031, 0.42, -0.028, 0.86, 0.24, 0.53], dtype=np.float64)
CLINICAL_INTERCEPT = -2.9


def _band_from_probability(variables: object, calibration: object) -> DiagnosticBand:
    from osteoct.models.baselines import ClinicalVariables

    assert isinstance(variables, ClinicalVariables)
    score = -0.004 * variables.age_years - 0.03 * variables.body_mass_index + (0.35 if variables.prior_fragility_fracture else 0.0)
    return band_from_t_score(float(score), variables.prior_fragility_fracture)


def execute_rule_arm(
    context: StudyContext,
    arm: str,
    policy: Policy,
    rows: npt.NDArray[np.int64],
    axioms: AxiomSet | None = None,
) -> ArmResult:
    active = context.attributes if axioms is None else axioms
    records: list[ArmRecord] = []
    for row in rows.tolist():
        candidate = rule_candidate(context, row, arm)
        bundle = context.encoded.bundles[row]
        source = context.encoded.records[row]
        outcome = context.toolbox.operate(
            candidate, bundle, active, context.budget, context.registry, enforced=policy.enforces_operator()
        )
        if outcome.emitted() and outcome.admitted is not None:
            admitted = outcome.admitted
            unsupported = not supported_code(admitted.candidate, active, bundle)
            records.append(
                ArmRecord(
                    row=row,
                    record_id=source.record_id,
                    patient_id=source.patient_id,
                    site=source.site,
                    band=_band_index(admitted.candidate.band),
                    tier=_tier_index(admitted.candidate.tier),
                    severity=int(admitted.candidate.severity.value),
                    risk=clinical_risk(
                        admitted.candidate.probability,
                        admitted.candidate.band,
                        admitted.candidate.tier,
                        admitted.candidate.severity,
                        source.context,
                    ),
                    conformance=admitted.conformance,
                    emitted=True,
                    unsupported=unsupported,
                    repairs=admitted.repairs,
                    reason="",
                    violated_families=tuple(
                        sorted({violation.family.value for violation in active.violations(admitted.candidate, bundle)})
                    ),
                )
            )
            continue
        reason = outcome.reason() or "budget_exhausted"
        records.append(
            ArmRecord(
                row=row,
                record_id=source.record_id,
                patient_id=source.patient_id,
                site=source.site,
                band=-1,
                tier=-1,
                severity=-1,
                risk=0.5,
                conformance=0.0,
                emitted=False,
                unsupported=True,
                repairs=outcome.trace.repairs(),
                reason=reason,
            )
        )
    frozen = tuple(records)
    return ArmResult(arm=arm, policy=policy, rows=rows, per_seed=(frozen,), ensemble=frozen)


def arm_result(
    context: StudyContext,
    arm: str,
    policy: Policy,
    rows: npt.NDArray[np.int64],
    training: dict[str, TrainingResult],
    axioms: AxiomSet | None = None,
) -> ArmResult:
    if arm in RULE_ARMS:
        return execute_rule_arm(context, arm, policy, rows, axioms)
    if arm not in training:
        raise KeyError(f"arm {arm!r} has no trained run and is not a rule arm")
    return execute_arm(context, arm, policy, rows, training[arm], axioms)
