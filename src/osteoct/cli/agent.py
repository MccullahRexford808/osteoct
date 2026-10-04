"""Command-line entry point that runs the ten-tool agent over one cohort partition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from osteoct.agent.pipeline import AgentOutcome, Policy, run_agent, summarise_outcomes
from osteoct.agent.planner import Planner, PlannerConfig
from osteoct.agent.tools import ToolRegistry, tool_report
from osteoct.cli.support import LOGGER, add_common_arguments, prepare, resolve, train_all
from osteoct.cohort.cohort import examination_from_record
from osteoct.evaluation.study import PARTITION_INTERNAL, StudyContext
from osteoct.serialisation.dcm_sr import sr_digest
from osteoct.training.trainer import TrainingResult
from osteoct.utils.io import atomic_write_text


@dataclass(frozen=True, slots=True)
class AgentRun:
    partition: str
    processed: int
    emitted: int
    abstained: int
    mean_conformance: float
    reasons: dict[str, int]
    tools: dict[str, object]
    first_record: str
    log: dict[str, float]

    def inventory(self) -> tuple[str, ...]:
        names = self.tools["inventory"]
        return tuple(str(name) for name in names) if isinstance(names, list) else ()

    def as_text(self) -> str:
        lines = [
            f"partition {self.partition}",
            f"processed {self.processed}",
            f"emitted {self.emitted}",
            f"abstained {self.abstained}",
            f"mean_conformance {self.mean_conformance:.6f}",
            f"abstention_reasons {self.reasons}",
            f"tool_inventory {','.join(str(name) for name in self.inventory())}",
            f"tool_calls {self.tools['calls']}",
            f"total_tool_calls {self.tools['total_calls']}",
            f"error_paths {self.tools['error_paths']}",
            "",
            "log",
            " ".join(f"{key}={value:.6f}" for key, value in sorted(self.log.items())),
            "",
            "first emitted record",
            self.first_record,
            "",
            f"first_serialised_digest {sr_digest(self.first_record) if self.first_record else ''}",
        ]
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osteoct-agent", description="run the OSTEO-CT agent over a cohort partition")
    add_common_arguments(parser)
    parser.add_argument("--partition", default=PARTITION_INTERNAL, choices=[PARTITION_INTERNAL, "external", "development"])
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--out", default="agent_run.txt")
    return parser


def agent_run(
    context: StudyContext,
    state: dict[str, dict[str, Tensor]],
    arm: str,
    partition: str,
    limit: int,
) -> AgentRun:
    planner = Planner(PlannerConfig(arm=arm))
    planner.load_state(state)
    registry = ToolRegistry()
    rows = context.rows(partition)[: max(1, limit)]
    fixture_records = context.fixture.records()
    outcomes: list[AgentOutcome] = []
    for row in rows.tolist():
        volume = context.fixture.volume_for(fixture_records[row])
        examination = examination_from_record(context.encoded.records[row], volume)
        outcomes.append(
            run_agent(examination, context.toolbox, planner, context.attributes, Policy.CONSTRAINED, context.budget, registry)
        )
    frozen = tuple(outcomes)
    emitted = [outcome for outcome in frozen if outcome.emitted()]
    log = summarise_outcomes(frozen)
    return AgentRun(
        partition=partition,
        processed=len(frozen),
        emitted=len(emitted),
        abstained=len(frozen) - len(emitted),
        mean_conformance=float(np.mean([outcome.conformance for outcome in emitted])) if emitted else 0.0,
        reasons=abstention_tally(frozen),
        tools=tool_report(registry),
        first_record=next((outcome.serialised for outcome in emitted), ""),
        log={key: float(value) for key, value in log.values.items()},
    )


def abstention_tally(outcomes: Sequence[AgentOutcome]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for outcome in outcomes:
        reason = outcome.reason()
        if reason:
            tally[reason] = tally.get(reason, 0) + 1
    return dict(sorted(tally.items()))


def first_seed_state(training: TrainingResult) -> dict[str, dict[str, Tensor]]:
    return training.runs[0].state


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root, config = resolve(args)
    context = prepare(config)
    arm = str(config.get("arm", "constrained"))
    training = train_all(context, config, None)
    if arm not in training:
        LOGGER.error("arm %s is a rule arm and exposes no agent state", arm)
        return 2
    run = agent_run(context, first_seed_state(training[arm]), arm, args.partition, args.limit)
    atomic_write_text(root / args.out, run.as_text())
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
