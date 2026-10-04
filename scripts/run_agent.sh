#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT="${1:-main}"
PARTITION="${2:-internal_test}"
LIMIT="${3:-32}"
OUT="${4:-agent_run.txt}"

"${PYTHON_BIN}" -m osteoct.cli.agent "${EXPERIMENT}" --partition "${PARTITION}" --limit "${LIMIT}" --out "${OUT}"
