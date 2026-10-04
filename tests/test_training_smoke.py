"""Training-loop smoke tests: three steps on the smoke configuration, checkpoint round trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from osteoct.agent.features import rows_for_partition
from osteoct.agent.planner import Planner, PlannerConfig
from osteoct.losses.total import ObjectiveWeights, compose
from osteoct.models.scorer import ScorerBatch
from osteoct.training.checkpoint import CheckpointPayload, content_digest, load_checkpoint, save_checkpoint, state_matches
from osteoct.training.ema import EmaTracker
from osteoct.training.optim import OptimConfig, backpropagate, build_optimizer, clip_gradients, parameter_norm
from osteoct.training.precision import PrecisionConfig, resolve_precision
from osteoct.training.schedule import ScheduleConfig, WarmupSchedule, apply_learning_rate, schedule_curve
from osteoct.training.seeds import derive_seed, rng_for, seed_list, seed_report, set_seed
from osteoct.training.trainer import (
    THRESHOLD_GRID,
    TrainConfig,
    build_features,
    config_fingerprint,
    derived_targets,
    patient_split,
    predict,
    select_thresholds,
    train_seed,
)
from tests.conftest import make_batch


def test_set_seed_reproduces_the_stream() -> None:
    first = set_seed(11)
    draws_first = first.random(5).tolist()
    second = set_seed(11)
    draws_second = second.random(5).tolist()
    assert draws_first == draws_second
    third = set_seed(12)
    assert third.random(5).tolist() != draws_first


def test_seed_list_and_derivation() -> None:
    seeds = seed_list(3, 5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    assert seed_list(3, 5) == seeds
    assert derive_seed(3, "band") == derive_seed(3, "band")
    assert float(rng_for(3, "band").random()) == float(rng_for(3, "band").random())
    assert seed_report(seeds)["count"] == 5
    with pytest.raises(ValueError):
        seed_list(1, 0)


def test_warmup_then_cosine_decay() -> None:
    schedule = WarmupSchedule(ScheduleConfig(name="cosine", warmup_steps=10, total_steps=110, minimum_ratio=0.05), 1.0)
    assert schedule.factor(0) == pytest.approx(1.0 / 10.0)
    assert schedule.factor(9) == pytest.approx(1.0)
    curve = schedule.curve(120)
    assert curve[0] < curve[9]
    assert curve[-1] < curve[9]
    assert min(curve) >= 0.05 - 1e-12
    assert schedule_curve(schedule, 5).shape == (5,)


@pytest.mark.parametrize("name", ["cosine", "step", "linear", "constant"])
def test_all_schedules_return_a_positive_factor(name: str) -> None:
    schedule = WarmupSchedule(ScheduleConfig(name=name, warmup_steps=4, total_steps=40, minimum_ratio=0.01), 2e-4)
    assert all(schedule.learning_rate(step) > 0.0 for step in range(45))
    with pytest.raises(KeyError):
        WarmupSchedule(ScheduleConfig(name="mystery", warmup_steps=1, total_steps=4), 1.0).factor(2)


def test_schedule_rejects_an_empty_horizon() -> None:
    with pytest.raises(ValueError):
        WarmupSchedule(ScheduleConfig(warmup_steps=2, total_steps=0), 1.0)


def test_apply_learning_rate_updates_every_group() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(model.parameters(), (), OptimConfig(learning_rate=1e-3))
    apply_learning_rate(optimizer, 5e-4)
    assert all(group["lr"] == 5e-4 for group in optimizer.param_groups)


def test_optimizer_builder_rejects_an_unknown_name() -> None:
    model = torch.nn.Linear(2, 2)
    with pytest.raises(KeyError):
        build_optimizer(model.parameters(), (), OptimConfig(name="rmsprop"))


def test_precision_configuration() -> None:
    assert not resolve_precision("fp32").uses_scaler()
    assert resolve_precision("bf16").as_mapping()["name"] == "bf16"
    assert resolve_precision("fp16").uses_scaler()
    with pytest.raises(KeyError):
        PrecisionConfig(name="int8")


def test_gradient_helpers() -> None:
    model = torch.nn.Linear(4, 1)
    output = model(torch.randn(3, 4)).sum()
    backpropagate(output)
    norm = clip_gradients(model.parameters(), 1.0)
    assert norm > 0.0
    assert parameter_norm(model.parameters()) > 0.0
    assert clip_gradients((), 1.0) == 0.0


def test_checkpoint_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    model = torch.nn.Linear(3, 2)
    state = {"scorer": {key: value.detach().clone() for key, value in model.state_dict().items()}}
    payload = CheckpointPayload(seed=5, step=9, epoch=4, arm="constrained", run="unit", state=state, metric=0.5, extra={"k": "v"})
    digest = save_checkpoint(root / "ckpt.pt", payload)
    restored = load_checkpoint(root / "ckpt.pt")
    assert restored.seed == 5
    assert restored.step == 9
    assert restored.extra["k"] == "v"
    assert content_digest(restored.state) == digest
    assert state_matches(restored.state, state)


def test_checkpoint_digest_ignores_container_metadata(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    model = torch.nn.Linear(2, 2)
    state = {"scorer": {key: value.detach().clone() for key, value in model.state_dict().items()}}
    payload = CheckpointPayload(seed=1, step=1, epoch=1, arm="a", run="r", state=state, metric=0.0, extra={})
    first = save_checkpoint(root / "a.pt", payload)
    second = save_checkpoint(root / "b.pt", payload)
    assert first == second


def test_ema_tracks_and_restores() -> None:
    model = torch.nn.Linear(2, 2)
    tracker = EmaTracker([model], decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    tracker.update()
    assert tracker.divergence() > 0.0
    tracker.copy_to([model])
    restored = [parameter.detach() for parameter in model.parameters()]
    for before, after in zip(initial, restored, strict=True):
        assert not torch.equal(before, after)


def test_patient_split_is_disjoint(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")
    if rows.size < 4:
        pytest.skip("the smoke cohort is too small for a split")
    training_rows, validation_rows = patient_split(smoke_study.encoded, rows, 0.34, 0)
    train_patients = {smoke_study.encoded.records[int(row)].patient_id for row in training_rows.tolist()}
    validation_patients = {smoke_study.encoded.records[int(row)].patient_id for row in validation_rows.tolist()}
    assert not train_patients & validation_patients


def test_derived_targets_align_with_the_axioms(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")[:4]
    if not rows.size:
        pytest.skip("the smoke cohort carries no development row")
    band_labels, tier_labels, severity_labels, evidence = derived_targets(smoke_study.encoded, rows, smoke_study.attributes)
    assert band_labels.shape == rows.shape
    assert tier_labels.shape == rows.shape
    assert severity_labels.shape == rows.shape
    assert evidence.shape == rows.shape
    assert set(np.unique(band_labels).tolist()) <= {0, 1, 2, 3}


def test_feature_builder_matches_the_scorer_width(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")[:3]
    planner = Planner(PlannerConfig(arm="constrained"))
    tensors = build_features(smoke_study.encoded, rows, planner, train_encoder=False)
    assert int(tensors.embedding.shape[1]) == planner.bundle.embedding_dim
    output = planner.bundle.scorer(tensors)
    assert int(output.risk_logit.shape[0]) == int(rows.size)


def test_predict_returns_one_value_per_row(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")[:5]
    planner = Planner(PlannerConfig(arm="constrained"))
    band, tier, severity, risk = predict(planner, smoke_study.encoded, rows)
    assert band.shape == tier.shape == severity.shape == risk.shape == rows.shape
    assert np.all((risk >= 0.0) & (risk <= 1.0))


def test_threshold_selection_respects_the_grid() -> None:
    probabilities = np.linspace(0.0, 1.0, 50)
    outcomes = (probabilities > 0.6).astype(np.float64)
    chosen = select_thresholds(probabilities, outcomes, THRESHOLD_GRID)
    assert chosen in THRESHOLD_GRID


def test_config_fingerprint_is_stable() -> None:
    config = TrainConfig(seeds=(0,), epochs=2)
    assert config_fingerprint(config) == config_fingerprint(TrainConfig(seeds=(0,), epochs=2))
    assert config_fingerprint(config) != config_fingerprint(TrainConfig(seeds=(0,), epochs=3))
    assert config.effective_batch() == config.batch_size * config.grad_accum * config.world_size


@pytest.mark.slow
def test_three_step_training_closes_the_loop(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    config = TrainConfig(seeds=(0,), epochs=2, batch_size=4, grad_accum=1, early_stopping_patience=2, validation_fraction=0.3)
    run = train_seed(smoke_study.encoded, "constrained", config, smoke_study.attributes, 0)
    assert run.epochs_run >= 1
    assert run.history
    assert np.isfinite(run.best_metric)
    state = run.state
    assert content_digest(state) == run.digest()


def test_training_on_one_batch_lowers_the_objective(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    token_rows = rows_for_partition(smoke_study.encoded, "development")[:8]
    if token_rows.size < 4:
        pytest.skip("the smoke cohort is too small")
    config = TrainConfig(seeds=(0,), epochs=8, batch_size=4, grad_accum=1, early_stopping_patience=8, validation_fraction=0.0)
    run = train_seed(smoke_study.encoded, "shifted_window", config, smoke_study.attributes, 0)
    assert run.history[-1] < run.history[0]


def test_loss_parts_sum_to_the_total(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = rows_for_partition(smoke_study.encoded, "development")[:4]
    if not rows.size:
        pytest.skip("the smoke cohort carries no development row")
    planner = Planner(PlannerConfig(arm="constrained"))
    tensors = build_features(smoke_study.encoded, rows, planner, train_encoder=False)
    planner.bundle.scorer.fit_normalisation(tensors)
    output = planner.bundle.scorer(tensors)
    labels = derived_targets(smoke_study.encoded, rows, smoke_study.attributes)
    batch = ScorerBatch(
        descriptors=smoke_study.encoded.descriptors[rows],
        texture=smoke_study.encoded.texture[rows],
        concepts=smoke_study.encoded.concepts[rows],
        context=smoke_study.encoded.context[rows],
        embedding=smoke_study.encoded.embedding[rows],
        band_target=smoke_study.encoded.band_target[rows],
        tier_target=smoke_study.encoded.tier_target[rows],
        severity_target=smoke_study.encoded.severity_target[rows],
        risk_target=smoke_study.encoded.risk_target[rows],
    )
    del batch
    objective, breakdown = compose(
        output.band_logits,
        output.tier_logits,
        output.severity_logits,
        output.risk_logit,
        torch.as_tensor(smoke_study.encoded.band_target[rows], dtype=torch.long),
        torch.as_tensor(smoke_study.encoded.tier_target[rows], dtype=torch.long),
        torch.as_tensor(smoke_study.encoded.severity_target[rows], dtype=torch.long),
        torch.as_tensor(smoke_study.encoded.risk_target[rows], dtype=torch.float32),
        ObjectiveWeights(),
        derived_band=labels[0],
        derived_tier=labels[1],
        graded_severity=labels[2],
        evidence_complete=labels[3],
    )
    parts = breakdown.as_mapping()
    assert abs(parts["supervised"] + 0.2 * parts["calibration"] - parts["total"]) < 1e-4
    assert float(objective.detach()) == pytest.approx(parts["total"], abs=1e-4)


def test_batch_width_matches_the_scorer() -> None:
    batch = make_batch(4)
    assert batch.width() == 3 + 25 + 30 + 16
    sliced = batch.slice(np.asarray([0, 2], dtype=np.int64))
    assert sliced.size() == 2
    assert sliced.width() == batch.width()
