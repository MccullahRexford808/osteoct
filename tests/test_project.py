"""Configuration, cohort-record and command-line support tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from osteoct.agent.features import rows_for_partition
from osteoct.agent.pipeline import Policy
from osteoct.cli.support import (
    all_trained_arms,
    comparison_arms,
    outcome_arms,
    prepare,
    substitution_arms,
)
from osteoct.cohort.cohort import RECORD_FIELDS, Cohort, CohortError, load_cohort, record_from_mapping
from osteoct.evaluation.study import PARTITION_EXTERNAL, PARTITION_INTERNAL
from osteoct.ontology.axioms import FAMILY_ORDER, FamilyName
from osteoct.schema.records import AcquisitionDescriptor
from osteoct.schema.vocabulary import DiagnosticBand, DiagnosticContext
from osteoct.utils.config import (
    ConfigError,
    deep_merge,
    discovered_experiments,
    flatten,
    load_experiment,
    load_yaml,
    parse_override,
    set_dotted,
)
from osteoct.utils.io import atomic_write_json, counts_by_suffix, directory_digest


def test_every_experiment_configuration_loads(repository_root: Path) -> None:
    names = discovered_experiments(repository_root)
    assert "main" in names
    assert "_smoke" in names
    assert any(name.startswith("ablation_without_") for name in names)
    for name in names:
        config = load_experiment(repository_root, name)
        assert config.name == name
        assert config.section("ontology")["active_families"]
        assert config.section("cohort")["bodies"] >= 1


def test_per_family_ablation_configurations_drop_one_family(repository_root: Path) -> None:
    labels = {
        FamilyName.IDENTITY: "identity",
        FamilyName.BAND: "band",
        FamilyName.TIER: "tier",
        FamilyName.SEVERITY: "severity",
        FamilyName.EVIDENCE: "evidence",
    }
    for family, label in labels.items():
        config = load_experiment(repository_root, f"ablation_without_{label}")
        active = set(config.section("ontology")["active_families"])
        assert family.value not in active
        assert active < {value.value for value in FAMILY_ORDER}


def test_main_configuration_carries_the_reported_design_constants(repository_root: Path) -> None:
    config = load_experiment(repository_root, "main")
    assert config.get("evaluation.pre_specified_conformance") == 0.90
    assert config.get("evaluation.mcid_concordance") == 0.030
    assert config.get("evaluation.utility_margin_points") == 5.0
    assert config.get("evaluation.conformance_margin_points") == 15.0
    assert len(config.get("train.seeds")) == 5
    assert config.get("pipeline.constrain_budget") >= 1
    assert config.get("arm") == "constrained"


def test_config_overrides_are_applied_after_the_defaults_merge(repository_root: Path) -> None:
    config = load_experiment(repository_root, "main", ("train.epochs=3", "pipeline.constrain_budget=2", "model.dropout=0.0"))
    assert config.get("train.epochs") == 3
    assert config.get("pipeline.constrain_budget") == 2
    assert config.get("model.dropout") == 0.0
    assert config.get("model.arm") == "constrained"


def test_override_parsing_and_coercion() -> None:
    assert parse_override("a=1") == ("a", 1)
    assert parse_override("a=1.5") == ("a", 1.5)
    assert parse_override("a=true") == ("a", True)
    assert parse_override("a=none") == ("a", None)
    assert parse_override("a=[1, 2]") == ("a", [1, 2])
    with pytest.raises(ConfigError):
        parse_override("a")


def test_deep_merge_and_flatten() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}
    flat = flatten({"a": {"b": 1}, "c": 2})
    assert flat == {"a.b": 1, "c": 2}
    target: dict[str, object] = {}
    set_dotted(target, "x.y.z", 4)
    assert target == {"x": {"y": {"z": 4}}}


def test_missing_configuration_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml(Path("does_not_exist.yaml"))
    with pytest.raises(ConfigError):
        load_experiment(tmp_path, "not_an_experiment")


def test_selection_helpers_cover_every_configured_arm(repository_root: Path) -> None:
    config = load_experiment(repository_root, "main")
    methods = [selection.method for selection in comparison_arms(config)]
    assert "OSTEO-CT (constrained)" in methods
    assert "Tool-using agent, no ontology layer" in methods
    assert {selection.policy for selection in comparison_arms(config)} <= set(Policy)
    assert all(selection.method for selection in substitution_arms(config))
    assert all(selection.method for selection in outcome_arms(config))
    assert "constrained" in all_trained_arms(config)
    assert "hu_threshold_rule" in all_trained_arms(config)


def test_prepare_builds_a_study(repository_root: Path) -> None:
    config = load_experiment(repository_root, "_smoke")
    context = prepare(config)
    assert context.encoded.size() > 0
    assert context.budget >= 1
    assert context.rows(PARTITION_INTERNAL).size >= 0
    assert context.rows(PARTITION_EXTERNAL).size >= 0


def hand_authored_record(
    record_id: str = "V-0001",
    patient_id: str = "V-P0001",
    partition: str = "development",
    reference_t_score: float | None = -1.4,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "site": "S1",
        "region": "R1",
        "partition": partition,
        "tube_voltage_kvp": 120.0,
        "scanner_family": "vendor_a",
        "reconstruction_kernel": "B40f",
        "slice_thickness_mm": 1.0,
        "prior_fractility": False,
        "treatment_exposure": False,
        "reference_t_score": reference_t_score,
        "fracture_time_days": None,
        "observation_time_days": 1825.0,
        "censored": True,
        "treatment_start_days": None,
        "age_years": 66.0,
        "female": True,
        "body_mass_index": 24.0,
        "glucocorticoid_exposure": False,
    }


def test_record_mapping_requires_every_field() -> None:
    mapping = hand_authored_record()
    assert set(RECORD_FIELDS) == set(mapping)
    parsed = record_from_mapping(mapping)
    assert parsed.record_id == "V-0001"
    assert parsed.prior_fractility is False
    assert parsed.context.reference_t_score == -1.4
    assert parsed.treatment_start_days is None
    with pytest.raises(CohortError):
        record_from_mapping({key: value for key, value in mapping.items() if key != "site"})


def test_record_mapping_accepts_an_absent_reference_score() -> None:
    parsed = record_from_mapping(hand_authored_record(reference_t_score=None))
    assert not parsed.context.reference_density_available()
    assert parsed.context.reference_t_score is None


def test_cohort_loader_rejects_an_unknown_partition(tmp_path: Path) -> None:
    mapping = hand_authored_record(partition="unknown_partition")
    path = tmp_path / "cohort.json"
    atomic_write_json(path, {"records": [mapping]})
    with pytest.raises(CohortError):
        load_cohort(path)
    with pytest.raises(CohortError):
        load_cohort(tmp_path / "missing.json")


def test_cohort_loader_accepts_a_conforming_file(tmp_path: Path) -> None:
    records = [
        hand_authored_record("V-0001", "V-P0001", "development"),
        hand_authored_record("V-0002", "V-P0002", "internal_test"),
        hand_authored_record("V-0003", "V-P0002", "external"),
    ]
    path = tmp_path / "cohort.json"
    atomic_write_json(path, {"records": records})
    cohort = load_cohort(path)
    assert cohort.size() == 3
    assert cohort.patient_count() == 2
    assert cohort.partition_table() == {"development": 1, "internal_test": 1, "external": 1}
    assert cohort.site_table() == {"S1": 3}
    assert cohort.sites() == ("S1",)
    assert len(cohort.with_followup()) == 3


def test_cohort_helpers_report_tables() -> None:
    records = tuple(
        record_from_mapping(hand_authored_record(f"V-{index:04d}", f"V-P{index % 2:04d}", partition))
        for index, partition in enumerate(("development", "development", "internal_test", "external"))
    )
    cohort = Cohort(records)
    assert cohort.size() == 4
    assert sum(cohort.partition_table().values()) == 4
    assert sum(cohort.site_table().values()) == 4
    assert cohort.patient_count() == 2
    first = records[0]
    assert isinstance(first.acquisition, AcquisitionDescriptor)
    assert isinstance(first.context, DiagnosticContext)
    assert cohort.densitometric_band(first, 240.0, 45.0) in set(DiagnosticBand)


def test_smoke_study_matches_the_dependency_order(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    encoded = smoke_study.encoded
    rows = rows_for_partition(encoded, "development")
    assert rows.size
    first = int(rows[0])
    bundle = encoded.bundles[first]
    assert bundle.standard_attenuation_hu > 0.0
    assert bundle.uncertainty_hu > 0.0
    assert bundle.texture_vector
    assert bundle.retained_texture_names
    assert bool(encoded.usable[first])


def test_io_helpers(repository_root: Path, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b\n", encoding="utf-8")
    counts = counts_by_suffix(tmp_path)
    assert counts[".txt"] == 1
    assert counts[".py"] == 1
    size, digest, entries = directory_digest(tmp_path)
    assert size == 2
    assert len(digest) == 64
    assert set(entries) == {"a.txt", "b.py"}
    assert (repository_root / "configs" / "experiment" / "main.yaml").exists()
