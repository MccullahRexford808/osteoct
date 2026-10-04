#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT="${1:-main}"
RUNS="${RUNS_DIR:-${REPO_ROOT}/runs/${EXPERIMENT}}"

"${PYTHON_BIN}" -m osteoct.cli.evaluate "${EXPERIMENT}" --runs "${RUNS}"
