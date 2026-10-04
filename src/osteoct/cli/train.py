"""Command-line entry point that trains the arms of one experiment."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from osteoct.cli.support import LOGGER, add_common_arguments, all_trained_arms, prepare, resolve, train_all
from osteoct.training.trainer import config_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osteoct-train", description="train the scorer arms of one experiment")
    add_common_arguments(parser)
    parser.add_argument("--runs", default=None, help="directory for run artefacts; defaults to runs/<experiment>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root, config = resolve(args)
    context = prepare(config)
    run_root = (root / "runs" / config.name) if args.runs is None else __import__("pathlib").Path(args.runs)
    training = train_all(context, config, run_root)
    LOGGER.info("trained %d arms: %s", len(training), ", ".join(sorted(training)))
    lines = [f"experiment {config.name}", f"arms {len(training)}", f"arm list {', '.join(all_trained_arms(config))}"]
    for arm, result in sorted(training.items()):
        lines.append(
            f"  {arm}: seeds={len(result.runs)} "
            f"epochs={[run.epochs_run for run in result.runs]} "
            f"fingerprint={config_fingerprint(result.config)[:12]}"
        )
    summary = "\n".join(lines) + "\n"
    sys.stdout.write(summary)
    (run_root / "runs_summary.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "runs_summary.txt").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
