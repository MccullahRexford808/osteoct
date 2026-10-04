"""Module entry point allowing `python -m osteoct`."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from osteoct.cli import evaluate

    return evaluate.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
