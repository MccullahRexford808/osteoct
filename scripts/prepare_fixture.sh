#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

COHORT_ROOT="${COHORT_ROOT:-${REPO_ROOT}/cohort}"
mkdir -p "${COHORT_ROOT}"

"${PYTHON_BIN}" - "${REPO_ROOT}" "${COHORT_ROOT}" <<'PY'
import sys
from pathlib import Path

from osteoct.cohort.fixture import FixtureSpec, build_cohort_fixture
from osteoct.utils.config import load_experiment
from osteoct.utils.io import atomic_write_json

repository_root = Path(sys.argv[1])
cohort_root = Path(sys.argv[2])
config = load_experiment(repository_root, "main")
section = config.section("cohort")
fixture = build_cohort_fixture(
    FixtureSpec(
        sites=tuple(str(value) for value in section["sites"]),
        regions=tuple(str(value) for value in section["regions"]),
        examinations_per_site=int(section["examinations_per_site"]),
        bodies=int(section["bodies"]),
        slices_per_body=int(section["slices_per_body"]),
        slice_gap=int(section["slice_gap"]),
        rows=int(section["rows"]),
        columns=int(section["columns"]),
        development_share=float(section["development_share"]),
        attenuation_mean_hu=float(section["attenuation_mean_hu"]),
        attenuation_sd_hu=float(section["attenuation_sd_hu"]),
        dissociation_share=float(section["dissociation_share"]),
        dissociation_hazard_ratio=float(section["dissociation_hazard_ratio"]),
        seed=int(section["fixture_seed"]),
    )
)
records = [
    {
        "record_id": record.record_id,
        "patient_id": record.patient_id,
        "site": record.site,
        "region": record.region,
        "partition": record.partition,
        "latent_attenuation_hu": round(record.latent_attenuation_hu, 6),
        "latent_height_loss": round(record.latent_height_loss, 6),
        "dissociated": record.dissociated,
    }
    for record in fixture.records()
]
payload = {"cohort": fixture.cohort.partition_table(), "sites": fixture.cohort.site_table(), "records": records}
atomic_write_json(cohort_root / "manifest.json", payload)
print("records", len(records))
print("partitions", fixture.cohort.partition_table())
print("sites", fixture.cohort.site_table())
PY

echo "manifest written under ${COHORT_ROOT}/manifest.json"
