"""Shared orchestration for the command-line entry points."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from osteoct.agent.pipeline import Policy
from osteoct.evaluation.dissociation import run_dissociation
from osteoct.evaluation.family_ablation import run_family_ablation
from osteoct.evaluation.findings import assemble_findings
from osteoct.evaluation.main_comparison import run_main_comparison
from osteoct.evaluation.outcome_validation import run_outcome_validation
from osteoct.evaluation.reader_study import run_reader_study
from osteoct.evaluation.report import EvaluationBundle, write_artefacts
from osteoct.evaluation.silent_mode import run_silent_mode
from osteoct.evaluation.study import StudyContext, build_study, train_arms
from osteoct.training.seeds import configure_determinism
from osteoct.training.trainer import TrainingResult
from osteoct.utils.config import ExperimentConfig, load_experiment
from osteoct.utils.logging import configure, get_logger

LOGGER = get_logger("cli")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("experiment", nargs="?", default="main", help="name of the experiment configuration file")
    parser.add_argument(
        "--override", action="append", default=[], help="dotted key=value override applied after the defaults merge"
    )
    parser.add_argument("--root", default=None, help="repository root; resolved from the package when omitted")
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve(args: argparse.Namespace) -> tuple[Path, ExperimentConfig]:
    root = repository_root() if args.root is None else Path(args.root)
    configure(args.log_level)
    config = load_experiment(root, args.experiment, tuple(args.override))
    return (root, config)


@dataclass(frozen=True, slots=True)
class ArmSelection:
    arm: str
    method: str
    policy: Policy

    def as_tuple(self) -> tuple[str, str, Policy]:
        return (self.arm, self.method, self.policy)


def _selection(entries: object) -> tuple[ArmSelection, ...]:
    if not isinstance(entries, list):
        return ()
    selections: list[ArmSelection] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        selections.append(
            ArmSelection(
                arm=str(entry["arm"]),
                method=str(entry["method"]),
                policy=Policy(str(entry.get("policy", "constrained"))),
            )
        )
    return tuple(selections)


def _entries(config: ExperimentConfig, key: str) -> object:
    nested = config.get(f"pipeline.{key}")
    return config.get(key) if nested is None else nested


def comparison_arms(config: ExperimentConfig) -> tuple[ArmSelection, ...]:
    return _selection(_entries(config, "comparison_arms"))


def substitution_arms(config: ExperimentConfig) -> tuple[ArmSelection, ...]:
    return _selection(_entries(config, "substitution_arms"))


def outcome_arms(config: ExperimentConfig) -> tuple[ArmSelection, ...]:
    return _selection(_entries(config, "outcome_arms"))


def all_trained_arms(config: ExperimentConfig) -> tuple[str, ...]:
    names: list[str] = []
    for selection in comparison_arms(config) + substitution_arms(config) + outcome_arms(config):
        if selection.arm not in names:
            names.append(selection.arm)
    reference = str(config.get("reference_arm", "constrained"))
    if reference not in names:
        names.append(reference)
    return tuple(names)


def train_all(context: StudyContext, config: ExperimentConfig, run_directory: Path | None) -> dict[str, TrainingResult]:
    return train_arms(context, config, all_trained_arms(config), run_directory)


def run_evaluation(
    root: Path,
    config: ExperimentConfig,
    context: StudyContext,
    training: dict[str, TrainingResult],
) -> EvaluationBundle:
    configure_determinism()
    started = time.monotonic()
    replicates = int(config.get("bootstrap_replicates", 2000))
    seed = int(config.get("pipeline.bootstrap_seed", 0)) if config.get("pipeline.bootstrap_seed") is not None else 0
    reference_method = str(config.get("reference_method", "OSTEO-CT (constrained)"))
    comparison, _ = run_main_comparison(
        context, training, tuple(item.as_tuple() for item in comparison_arms(config)), replicates, seed
    )
    LOGGER.info("main comparison completed with %d rows", len(comparison.rows))
    ablation = run_family_ablation(
        context,
        training,
        tuple(item.as_tuple() for item in substitution_arms(config)),
        str(config.get("reference_arm", "constrained")),
        Policy.CONSTRAINED,
    )
    LOGGER.info("constraint ablation completed with %d rows", len(ablation.rows))
    outcome_specification = tuple(item.as_tuple() for item in outcome_arms(config))
    outcomes, _ = run_outcome_validation(
        context,
        training,
        outcome_specification,
        str(config.get("clinical_comparator_method", "Tool-using agent, no ontology layer")),
        replicates,
        seed,
    )
    LOGGER.info("clinical-outcome validation completed with %d rows", len(outcomes.rows))
    readers = run_reader_study(context, int(config.get("reading_study_size", 400)), seed)
    silent = run_silent_mode(
        context,
        str(config.get("reference_arm", "constrained")),
        Policy.CONSTRAINED,
        training,
        float(config.get("escalation_threshold", 0.20)),
    )
    dissociation = run_dissociation(context)
    findings = assemble_findings(comparison, ablation, outcomes, dissociation, reference_method)
    return EvaluationBundle(
        experiment=config.name,
        comparison=comparison,
        ablation=ablation,
        outcomes=outcomes,
        readers=readers,
        silent=silent,
        dissociation=dissociation,
        findings=findings,
        seeds=max((len(result.runs) for result in training.values()), default=0),
        compute_seconds=time.monotonic() - started,
    )


def prepare(
    config: ExperimentConfig,
) -> StudyContext:
    return build_study(config)


def emit(bundle: EvaluationBundle, root: Path) -> tuple[Path, Path]:
    return write_artefacts(bundle, root)
