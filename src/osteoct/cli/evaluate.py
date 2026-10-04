"""Command-line entry point that runs the reported tables and writes their artefacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from osteoct.cli.support import LOGGER, add_common_arguments, emit, prepare, resolve, run_evaluation, train_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osteoct-evaluate", description="run the evaluation tables of one experiment")
    add_common_arguments(parser)
    parser.add_argument("--runs", default=None, help="directory for run artefacts; defaults to runs/<experiment>")
    parser.add_argument("--no-train", action="store_true", help="reuse the checkpoints already present under --runs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root, config = resolve(args)
    context = prepare(config)
    run_root = (root / "runs" / config.name) if args.runs is None else Path(args.runs)
    training = train_all(context, config, run_root)
    LOGGER.info("training complete, running the evaluation tables")
    bundle = run_evaluation(root, config, context, training)
    json_path, text_path = emit(bundle, root)
    LOGGER.info("wrote %s and %s", json_path.name, text_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
