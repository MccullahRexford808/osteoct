"""Training and threshold selection for one arm."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from osteoct.agent.features import EncodedCohort
from osteoct.agent.planner import Planner, PlannerConfig
from osteoct.losses.total import ObjectiveWeights, compose
from osteoct.metrics.discrimination import auc
from osteoct.models.scorer import ScorerFeatures
from osteoct.ontology.axioms import AxiomSet
from osteoct.ontology.bands import band_from_interval
from osteoct.ontology.severity import grade_from_vertebrae
from osteoct.ontology.tiers import tier_from_band
from osteoct.training.checkpoint import CheckpointPayload, content_digest, save_checkpoint
from osteoct.training.optim import OptimConfig, backpropagate, build_optimizer, clip_gradients
from osteoct.training.precision import PrecisionConfig, device_type_of
from osteoct.training.schedule import ScheduleConfig, WarmupSchedule
from osteoct.training.seeds import set_seed
from osteoct.utils.config import flatten
from osteoct.utils.io import atomic_write_json, atomic_write_text

THRESHOLD_GRID: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 8
    grad_accum: int = 2
    world_size: int = 1
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    ema_decay: float = 0.999
    early_stopping_patience: int = 12
    regulariser_scale: float = 0.0
    train_encoder: bool = True
    validation_fraction: float = 0.22

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * self.world_size

    def as_mapping(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "world_size": self.world_size,
            "effective_batch": self.effective_batch(),
            "optimizer": self.optimizer.as_mapping(),
            "schedule": self.schedule.as_mapping(),
            "precision": self.precision.as_mapping(),
            "ema_decay": self.ema_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "regulariser_scale": self.regulariser_scale,
            "train_encoder": self.train_encoder,
            "validation_fraction": self.validation_fraction,
        }


@dataclass(frozen=True, slots=True)
class SeedRun:
    seed: int
    epochs_run: int
    best_epoch: int
    best_metric: float
    history: tuple[float, ...]
    band_accuracy: float
    risk_auc: float
    state: dict[str, dict[str, Tensor]]

    def digest(self) -> str:
        return content_digest(self.state)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    arm: str
    config: TrainConfig
    planner_config: PlannerConfig
    runs: tuple[SeedRun, ...]
    thresholds: tuple[float, ...]

    def digests(self) -> tuple[str, ...]:
        return tuple(run.digest() for run in self.runs)

    def mean_history(self) -> tuple[float, ...]:
        if not self.runs:
            return ()
        width = min(len(run.history) for run in self.runs)
        matrix = np.asarray([run.history[:width] for run in self.runs], dtype=np.float64)
        return tuple(float(value) for value in matrix.mean(axis=0))


def patient_split(
    encoded: EncodedCohort,
    rows: npt.NDArray[np.int64],
    fraction: float,
    seed: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    patients = sorted({encoded.records[int(row)].patient_id for row in rows.tolist()})
    if len(patients) < 2 or fraction <= 0.0:
        return (rows, rows)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    cut = max(1, int(round(len(patients) * fraction)))
    held = {patients[int(order[index])] for index in range(cut)}
    validation = [int(row) for row in rows.tolist() if encoded.records[int(row)].patient_id in held]
    training = [int(row) for row in rows.tolist() if encoded.records[int(row)].patient_id not in held]
    if not validation or not training:
        return (rows, rows)
    return (np.asarray(training, dtype=np.int64), np.asarray(validation, dtype=np.int64))


def build_features(
    encoded: EncodedCohort,
    rows: npt.NDArray[np.int64],
    planner: Planner,
    train_encoder: bool,
) -> ScorerFeatures:
    device = planner.device
    descriptors = torch.as_tensor(encoded.descriptors[rows], dtype=torch.float32, device=device)
    texture = torch.as_tensor(encoded.texture[rows], dtype=torch.float32, device=device)
    concepts = torch.as_tensor(encoded.concepts[rows], dtype=torch.float32, device=device)
    context = torch.as_tensor(encoded.context[rows], dtype=torch.float32, device=device)
    if planner.bundle.encoder is not None:
        volumes: Tensor = torch.as_tensor(encoded.volumes[rows], dtype=torch.float32, device=device).unsqueeze(1)
        encoded_embedding: Tensor = planner.bundle.encoder(volumes)
        embedding = encoded_embedding if train_encoder else encoded_embedding.detach()
    else:
        embedding = torch.zeros((int(rows.size), 0), dtype=torch.float32, device=device)
    return ScorerFeatures(descriptors=descriptors, texture=texture, concepts=concepts, context=context, embedding=embedding)


def derived_targets(
    encoded: EncodedCohort,
    rows: npt.NDArray[np.int64],
    attributes: AxiomSet,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    band_labels = np.zeros(rows.size, dtype=np.int64)
    tier_labels = np.zeros(rows.size, dtype=np.int64)
    severity_labels = np.zeros(rows.size, dtype=np.int64)
    evidence = np.zeros(rows.size, dtype=np.float64)
    for position, row in enumerate(rows.tolist()):
        record = encoded.records[row]
        bundle = encoded.bundles[row]
        grounding = encoded.groundings[row]
        band = band_from_interval(bundle.standard_attenuation_hu, bundle.uncertainty_hu, attributes.calibration)
        tier = tier_from_band(band, record.context, record.acquisition, grounding.concepts, attributes.index, attributes.envelope)
        grade = grade_from_vertebrae(tuple(measurement.morphology for measurement in bundle.vertebrae), attributes.thresholds)
        band_labels[position] = _band_index(band)
        tier_labels[position] = _tier_index(tier)
        severity_labels[position] = int(grade.value)
        evidence[position] = 1.0
    return (band_labels, tier_labels, severity_labels, evidence)


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


def predict(
    planner: Planner,
    encoded: EncodedCohort,
    rows: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    planner.eval()
    band = np.zeros(rows.size, dtype=np.int64)
    tier = np.zeros(rows.size, dtype=np.int64)
    severity = np.zeros(rows.size, dtype=np.int64)
    risk = np.zeros(rows.size, dtype=np.float64)
    with torch.no_grad():
        features = build_features(encoded, rows, planner, train_encoder=False)
        output = planner.bundle.scorer(features)
    band[:] = torch.argmax(output.band_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    tier[:] = torch.argmax(output.tier_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    severity[:] = torch.argmax(output.severity_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
    risk[:] = torch.sigmoid(output.risk_logit).detach().cpu().numpy().astype(np.float64)
    return (band, tier, severity, risk)


def select_thresholds(probabilities: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64], grid: tuple[float, ...]) -> float:
    from osteoct.metrics.utility import net_benefit

    if probabilities.size == 0:
        return grid[len(grid) // 2]
    values = [net_benefit(probabilities, outcomes, threshold) for threshold in grid]
    return float(grid[int(np.argmax(np.asarray(values, dtype=np.float64)))])


def train_seed(
    encoded: EncodedCohort,
    arm: str,
    config: TrainConfig,
    attributes: AxiomSet,
    seed: int,
    run_directory: Path | None = None,
) -> SeedRun:
    set_seed(seed)
    planner = Planner(PlannerConfig(arm=arm))
    planner.train(True)
    train_rows, validation_rows = patient_split(encoded, _development_rows(encoded), config.validation_fraction, seed)
    planner.bundle.scorer.fit_normalisation(build_features(encoded, train_rows, planner, train_encoder=False))
    optimizer = build_optimizer(
        planner.bundle.scorer.parameters(),
        () if planner.bundle.encoder is None else planner.bundle.encoder.parameters(),
        config.optimizer,
    )
    schedule = WarmupSchedule(config.schedule, config.optimizer.learning_rate)
    objectives = (
        ObjectiveWeights(regulariser_scale=config.regulariser_scale) if config.regulariser_scale > 0.0 else ObjectiveWeights()
    )
    device_type = device_type_of(planner.device)
    rng = np.random.default_rng(seed)
    history: list[float] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state = planner.state()
    step = 0
    stale = 0
    epochs_run = 0
    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        planner.train(True)
        order = rng.permutation(train_rows.size)
        shuffled = train_rows[order]
        optimizer.zero_grad(set_to_none=True)
        for position in range(0, shuffled.size, config.batch_size):
            batch_rows = shuffled[position : position + config.batch_size]
            features = build_features(encoded, batch_rows, planner, config.train_encoder)
            with config.precision.autocast(device_type):
                output = planner.bundle.scorer(features)
                band_labels, tier_labels, severity_labels, evidence = derived_targets(encoded, batch_rows, attributes)
                loss, breakdown = compose(
                    output.band_logits,
                    output.tier_logits,
                    output.severity_logits,
                    output.risk_logit,
                    torch.as_tensor(encoded.band_target[batch_rows], dtype=torch.long, device=planner.device),
                    torch.as_tensor(encoded.tier_target[batch_rows], dtype=torch.long, device=planner.device),
                    torch.as_tensor(encoded.severity_target[batch_rows], dtype=torch.long, device=planner.device),
                    torch.as_tensor(encoded.risk_target[batch_rows], dtype=torch.float32, device=planner.device),
                    objectives,
                    derived_band=band_labels,
                    derived_tier=tier_labels,
                    graded_severity=severity_labels,
                    evidence_complete=evidence,
                )
                scaled = loss / float(config.grad_accum)
            backpropagate(scaled)
            if (position // max(1, config.batch_size) + 1) % config.grad_accum == 0:
                clip_gradients(planner.parameters(), config.optimizer.gradient_clip)
                for group in optimizer.param_groups:
                    group["lr"] = schedule.learning_rate(step) * float(group.get("scale", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            history.append(breakdown.total)
        planner.eval()
        validation = predict(planner, encoded, validation_rows)
        metric = _validation_metric(encoded, validation_rows, validation)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = planner.state()
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    planner.load_state(best_state)
    test_band, _, _, test_risk = predict(planner, encoded, validation_rows)
    band_accuracy = float(np.mean(test_band == encoded.band_target[validation_rows])) if validation_rows.size else 0.0
    risk_auc = (
        auc(test_risk, (encoded.risk_target[validation_rows] > 0.5).astype(np.int64)) if validation_rows.size else float("nan")
    )
    if run_directory is not None:
        payload = CheckpointPayload(
            seed=seed,
            step=step,
            epoch=best_epoch,
            arm=arm,
            run="train_seed",
            state=best_state,
            metric=best_metric,
            extra={
                "schedule": config.schedule.name,
                "precision": config.precision.name,
                "learning_rate": f"{config.optimizer.learning_rate:g}",
            },
        )
        digest = save_checkpoint(run_directory / f"seed_{seed}.pt", payload)
        atomic_write_text(run_directory / f"seed_{seed}.digest", digest + "\n")
    return SeedRun(
        seed=seed,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_metric=best_metric,
        history=tuple(history),
        band_accuracy=band_accuracy,
        risk_auc=float(risk_auc),
        state=best_state,
    )


def _validation_metric(
    encoded: EncodedCohort,
    rows: npt.NDArray[np.int64],
    prediction: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]],
) -> float:
    if rows.size == 0:
        return 0.0
    band = prediction[0]
    return float(np.mean(band == encoded.band_target[rows]))


def _development_rows(encoded: EncodedCohort) -> npt.NDArray[np.int64]:
    selected = [
        index for index, record in enumerate(encoded.records) if record.partition == "development" and bool(encoded.usable[index])
    ]
    return np.asarray(selected, dtype=np.int64)


def train_arm(
    encoded: EncodedCohort,
    arm: str,
    config: TrainConfig,
    attributes: AxiomSet,
    run_directory: Path | None = None,
) -> TrainingResult:
    runs: list[SeedRun] = []
    for seed in config.seeds:
        directory = None if run_directory is None else run_directory / f"seed_{seed}"
        runs.append(train_seed(encoded, arm, config, attributes, seed, directory))
    thresholds = ensemble_thresholds(encoded, arm, config, tuple(runs))
    return TrainingResult(
        arm=arm,
        config=config,
        planner_config=PlannerConfig(arm=arm),
        runs=tuple(runs),
        thresholds=thresholds,
    )


def ensemble_thresholds(
    encoded: EncodedCohort,
    arm: str,
    config: TrainConfig,
    runs: tuple[SeedRun, ...],
) -> tuple[float, ...]:
    rows = _development_rows(encoded)
    if rows.size == 0 or not runs:
        return tuple(1.0 for _ in config.seeds)
    probabilities = np.zeros((len(runs), rows.size), dtype=np.float64)
    for index, run in enumerate(runs):
        planner = Planner(PlannerConfig(arm=arm))
        planner.load_state(run.state)
        probabilities[index, :] = predict(planner, encoded, rows)[3]
    mean_probability = probabilities.mean(axis=0)
    threshold = select_thresholds(mean_probability, encoded.risk_target[rows], THRESHOLD_GRID)
    return tuple(threshold for _ in runs)


def ensemble_predictions(
    encoded: EncodedCohort,
    arm: str,
    runs: tuple[SeedRun, ...],
    rows: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not runs:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    band = np.zeros((len(runs), rows.size), dtype=np.float64)
    severity = np.zeros((len(runs), rows.size), dtype=np.float64)
    risk = np.zeros((len(runs), rows.size), dtype=np.float64)
    for index, run in enumerate(runs):
        planner = Planner(PlannerConfig(arm=arm))
        planner.load_state(run.state)
        prediction = predict(planner, encoded, rows)
        band[index, :] = prediction[0].astype(np.float64)
        severity[index, :] = prediction[2].astype(np.float64)
        risk[index, :] = prediction[3]
    return (band.mean(axis=0), band.std(axis=0), severity.mean(axis=0), risk.mean(axis=0))


def write_training_artefacts(result: TrainingResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "run.json",
        {
            "arm": result.arm,
            "config": result.config.as_mapping(),
            "planner": result.planner_config.as_mapping(),
            "thresholds": list(result.thresholds),
            "seeds": [
                {
                    "seed": run.seed,
                    "epochs_run": run.epochs_run,
                    "best_epoch": run.best_epoch,
                    "best_metric": run.best_metric,
                    "band_accuracy": run.band_accuracy,
                    "risk_auc": run.risk_auc,
                    "digest": run.digest(),
                }
                for run in result.runs
            ],
        },
    )
    lines = [
        f"arm {result.arm}",
        f"seeds {len(result.runs)}",
        f"epochs_cap {result.config.epochs}",
        f"effective_batch {result.config.effective_batch()}",
        f"mean_best_epoch {float(np.mean([run.best_epoch for run in result.runs])):.3f}",
        f"mean_band_accuracy {float(np.mean([run.band_accuracy for run in result.runs])):.4f}",
        f"thresholds {','.join(f'{value:.2f}' for value in result.thresholds)}",
        f"digests {','.join(run.digest()[:12] for run in result.runs)}",
    ]
    atomic_write_text(directory / "run_summary.txt", "\n".join(lines) + "\n")
    return directory


def config_fingerprint(config: TrainConfig) -> str:
    payload = flatten(config.as_mapping())
    text = "|".join(f"{key}={payload[key]}" for key in sorted(payload))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
