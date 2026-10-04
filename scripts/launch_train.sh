#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT="${1:-main}"
NPROC="${NPROC_PER_NODE:-1}"

if [[ "${NPROC}" -gt 1 ]] && command -v torchrun >/dev/null 2>&1; then
  torchrun --standalone --nproc_per_node="${NPROC}" -m osteoct.cli.train "${EXPERIMENT}"
else
  "${PYTHON_BIN}" -m osteoct.cli.train "${EXPERIMENT}"
fi
