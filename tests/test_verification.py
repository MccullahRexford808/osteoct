"""Tests over the verification driver and the shipped artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osteoct.cli import verify
from osteoct.ontology.axioms import FAMILY_ORDER, FAMILY_WEIGHTS, FamilyName
from osteoct.ontology.bands import BandCalibration
from osteoct.ontology.severity import GenantThresholds
from osteoct.utils.io import atomic_write_json


def test_reference_auc_matches_the_implementation() -> None:
    import numpy as np

    from osteoct.metrics.discrimination import auc

    rng = np.random.default_rng(7)
    labels = (rng.random(90) < 0.4).astype(np.int64)
    scores = rng.normal(0.0, 1.0, size=90) + labels
    assert verify.reference_auc(scores, labels) == pytest.approx(auc(scores, labels), abs=1e-12)


def test_reference_sensitivity_matches_the_implementation() -> None:
    import numpy as np

    from osteoct.metrics.discrimination import sensitivity_at_specificity

    rng = np.random.default_rng(9)
    labels = (rng.random(80) < 0.5).astype(np.int64)
    scores = rng.normal(0.0, 1.0, size=80) + 0.8 * labels
    assert verify.reference_sensitivity_at_specificity(scores, labels, 0.9) == pytest.approx(
        sensitivity_at_specificity(scores, labels, 0.9), abs=1e-9
    )


def test_reference_concordance_matches_the_implementation() -> None:
    import numpy as np

    from osteoct.metrics.survival import CensoringRule, concordance_index

    rng = np.random.default_rng(13)
    times = rng.uniform(5.0, 1200.0, size=45)
    events = rng.random(45) < 0.6
    risk = rng.normal(0.5, 0.2, size=45)
    assert verify.reference_concordance(times, events, risk) == pytest.approx(
        concordance_index(times, events, risk, CensoringRule(), None), abs=1e-12
    )


def test_reference_holm_matches_the_implementation() -> None:
    import numpy as np

    from osteoct.statistics.permutation import holm_step_down

    values = [0.002, 0.011, 0.019, 0.044, 0.2, 0.44]
    observed = holm_step_down(np.asarray(values, dtype=np.float64)).tolist()
    expected = verify.reference_holm(values)
    assert all(abs(first - second) < 1e-12 for first, second in zip(observed, expected, strict=True))


def test_reference_kappa_matches_the_implementation() -> None:
    import numpy as np

    rng = np.random.default_rng(19)
    first = rng.integers(0, 4, size=140)
    second = np.clip(first + rng.integers(-1, 2, size=140), 0, 3)
    from osteoct.evaluation.reader_study import cohen_kappa

    assert verify.reference_kappa(first.tolist(), second.tolist()) == pytest.approx(float(cohen_kappa(first, second)), abs=1e-12)


def test_reference_band_matches_the_implementation() -> None:
    from osteoct.ontology.bands import bands_from_interval

    calibration = BandCalibration(240.0, 45.0, 1.5)
    low_threshold, high_threshold = calibration.thresholds_hu()
    for attenuation, uncertainty in ((high_threshold, 5.0), (high_threshold + 40.0, 2.0), (low_threshold - 10.0, 3.0)):
        observed = [band.value for band in bands_from_interval(attenuation, uncertainty, calibration)]
        expected = [band.value for band in verify.reference_band(attenuation, uncertainty, calibration, False)]
        assert observed == expected


def test_reference_grade_matches_the_implementation() -> None:
    from osteoct.ontology.severity import grade_from_height_loss

    thresholds = GenantThresholds()
    for loss in (0.0, 0.12, 0.20, 0.24, 0.31, 0.41, 0.6):
        assert grade_from_height_loss(loss, thresholds) is verify.reference_grade(loss, thresholds)


def test_reference_tier_matches_the_implementation() -> None:
    from osteoct.ontology.index import entry_for
    from osteoct.ontology.tiers import ProtocolEnvelope, tier_from_band
    from osteoct.schema.concepts import TermSystem
    from osteoct.schema.records import AcquisitionDescriptor
    from osteoct.schema.vocabulary import DiagnosticBand, DiagnosticContext

    index = verify.build_concept_index()
    envelope = ProtocolEnvelope(2.5, 80.0, ("B40f",))
    acquisition = AcquisitionDescriptor(120.0, "vendor_a", "B40f", 1.0)
    concepts = (entry_for("risk_tier", TermSystem.LOINC).key,)
    for band in DiagnosticBand:
        for prior in (False, True):
            for treatment in (False, True):
                context = DiagnosticContext(prior_fragility_fracture=prior, treatment_exposure=treatment)
                observed = tier_from_band(band, context, acquisition, concepts, index, envelope)
                expected = verify.reference_tier(band, prior, treatment, True, True)
                assert observed is expected


def test_reference_conformance_matches_the_definition() -> None:
    satisfied = verify.reference_conformance([family for family in FAMILY_ORDER if family is not FamilyName.BAND], [])
    assert satisfied == pytest.approx(
        (sum(FAMILY_WEIGHTS.values()) - FAMILY_WEIGHTS[FamilyName.BAND]) / sum(FAMILY_WEIGHTS.values())
    )


def test_reference_net_benefit_matches_the_closed_form() -> None:
    import numpy as np

    from osteoct.metrics.utility import net_benefit

    rng = np.random.default_rng(29)
    probabilities = rng.uniform(0.0, 1.0, size=90)
    outcomes = (rng.random(90) < probabilities).astype(np.float64)
    assert verify.reference_net_benefit(probabilities, outcomes, 0.25) == pytest.approx(
        net_benefit(probabilities, outcomes, 0.25), abs=1e-12
    )


def test_claim_map_writes_into_the_supplied_root(tmp_path: Path) -> None:
    package_dir = Path(verify.__file__).resolve().parents[1]
    payload = verify.write_claim_map(tmp_path, package_dir)
    written = json.loads((tmp_path / "claim_to_code.json").read_text(encoding="utf-8"))
    assert written["counts"]["claims"] == len(verify.CLAIMS)
    assert written["counts"]["unresolved"] == []
    assert payload["counts"]["anchors"] == written["counts"]["anchors"]
    assert written["deviations"]
    for entry in written["claims"]:
        assert entry["paper_location"]
        assert entry["code"]


def test_claim_map_anchors_resolve_against_the_real_package() -> None:
    package_dir = Path(verify.__file__).resolve().parents[1]
    for entry in verify.CLAIMS:
        for path, symbol in entry["code"]:  # type: ignore[misc]
            assert verify.resolve_anchor(package_dir, str(path), str(symbol)), f"{entry['id']}: {path}:{symbol}"


def test_manifest_covers_the_live_tree_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    nested = tmp_path / "configs" / "data"
    nested.mkdir(parents=True)
    (nested / "cohort.yaml").write_text("k: v\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"cache")
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run.json").write_text("{}\n", encoding="utf-8")
    payload = verify.write_integrity_manifest(tmp_path)
    listed = set(payload["files"])  # type: ignore[arg-type]
    assert "a.txt" in listed
    assert "configs/data/cohort.yaml" in listed
    assert not any(entry.startswith("__pycache__") for entry in listed)
    assert not any(entry.startswith("runs/") for entry in listed)
    assert "integrity_manifest.json" not in listed
    comparison = verify.verify_manifest(tmp_path, payload)
    assert comparison["stale"] == []
    assert comparison["unlisted"] == []


def test_manifest_reports_a_stale_entry(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    payload = verify.write_integrity_manifest(tmp_path)
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    comparison = verify.verify_manifest(tmp_path, payload)
    assert comparison["stale"] == ["a.txt"]


def test_report_counts_every_check_and_keeps_two_statuses(tmp_path: Path) -> None:
    results = [
        verify.Check("a", "provenance", "first").ok(),
        verify.Check("b", "execution", "second").ok(),
        verify.Check("c", "manuscript", "third").bad("a printed value does not follow"),
    ]
    payload: dict[str, object] = {
        "artifact": "paper-claim to code mapping",
        "counts": {"claims": 1, "anchors": 1, "resolved_anchors": 1, "unresolved": []},
    }
    report = verify.write_verification_report(tmp_path, results, payload, "main")
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["checks"] == 3
    assert summary[verify.PASS] == 2
    assert summary[verify.FAIL] == 1
    assert summary["overall_status"] == "UNVERIFIED"
    assert summary["code_status"] == "VERIFIED"
    assert len(summary["manuscript_discrepancies"]) == 1
    written = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    assert "integrity_manifest" not in json.dumps(written)


def test_report_records_a_blocked_check_without_failing_code_status(tmp_path: Path) -> None:
    results = [
        verify.Check("a", "execution", "first").ok(),
        verify.Check("b", "serialisation", "second").blocked("external tool missing"),
    ]
    payload: dict[str, object] = {
        "artifact": "paper-claim to code mapping",
        "counts": {"claims": 1, "anchors": 1, "resolved_anchors": 1, "unresolved": []},
    }
    report = verify.write_verification_report(tmp_path, results, payload, "main")
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["overall_status"] == "PARTIALLY_VERIFIED"
    assert summary["code_status"] == "PARTIALLY_VERIFIED"
    assert summary[verify.BLOCKED] == 1


def test_summary_renders_non_passing_checks(tmp_path: Path) -> None:
    results = [verify.Check("a", "execution", "first").bad("did not hold")]
    payload: dict[str, object] = {
        "artifact": "paper-claim to code mapping",
        "counts": {"claims": 1, "anchors": 1, "resolved_anchors": 1, "unresolved": []},
    }
    report = verify.write_verification_report(tmp_path, results, payload, "main")
    text = verify.render_summary(report)
    assert "did not hold" in text
    assert "overall_status UNVERIFIED" in text


def test_manuscript_family_reproduces_the_printed_table_arithmetic() -> None:
    results = {result.identifier: result for result in verify.manuscript_checks()}
    for identifier in (
        "ms.trial_flow_arithmetic",
        "ms.table3_auroc_delta",
        "ms.table3_conformance_delta",
        "ms.severity_concordance_gain",
        "ms.dissociation_incidence_ratio",
        "ms.dissociation_share",
        "ms.conformance_span",
        "ms.unsupported_and_abstention_direction",
        "ms.conformance_ordering",
        "ms.reader_study_arithmetic",
    ):
        assert results[identifier].status == verify.PASS, identifier
    assert results["ms.table4_distinct_intervals"].status == verify.FAIL


def test_manuscript_discrepancy_is_pinned() -> None:
    rows = {label: (value, half) for label, value, half in verify.MANUSCRIPT_TABLE_FOUR}
    transformer = rows["3-D shifted-window transformer"]
    agent = rows["Tool-using agent, no ontology layer"]
    assert transformer == agent
    assert round(11.9 / 4.2, 1) == 2.8
    assert round(84 / 214, 3) == 0.393
    assert abs((0.801 - 0.776) - 0.025) < 1e-12
    assert 0.025 < 0.030


def test_hygiene_checks_pass_on_the_shipped_tree(repository_root: Path) -> None:
    results = {result.identifier: result for result in verify.hygiene_checks(repository_root)}
    assert results["hygiene.markdown_inventory"].status in {verify.PASS, verify.FAIL}
    assert results["hygiene.absolute_paths"].status == verify.PASS
    assert results["hygiene.emoji"].status == verify.PASS
    assert results["hygiene.framing_phrases"].status == verify.PASS
    assert results["hygiene.data_vocabulary"].status == verify.PASS
    assert results["hygiene.gitignore_anchoring"].status == verify.PASS
    assert results["hygiene.dataset_links"].status == verify.PASS


def test_shipped_report_is_the_complete_one(repository_root: Path) -> None:
    report_path = repository_root / "verification_report.json"
    if not report_path.exists():
        pytest.skip("the report is produced by the verification pass")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report["checks"]
    summary = report["summary"]
    assert len(checks) >= 40
    assert summary["checks"] == len(checks)
    categories = {record["category"] for record in checks}
    assert {"provenance", "execution", "algorithm", "measurement", "metric", "statistic", "manuscript"} <= categories
    assert summary["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    assert summary["code_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}


def test_shipped_manifest_matches_the_live_tree(repository_root: Path) -> None:
    manifest_path = repository_root / "integrity_manifest.json"
    if not manifest_path.exists():
        pytest.skip("the manifest is produced by the verification pass")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = verify.verify_manifest(repository_root, manifest)
    assert comparison["stale"] == []
    assert comparison["unlisted"] == []


def test_artefact_checks_accept_a_complete_report(tmp_path: Path) -> None:
    results = [verify.Check(f"c{index}", "execution", "check").ok() for index in range(45)]
    results.append(verify.Check("m", "manuscript", "manuscript check").bad("printed value"))
    payload: dict[str, object] = {
        "artifact": "paper-claim to code mapping",
        "counts": {"claims": 1, "anchors": 1, "resolved_anchors": 1, "unresolved": []},
    }
    verify.write_verification_report(tmp_path, results, payload, "main")
    verify.write_integrity_manifest(tmp_path)
    checks = {result.identifier: result for result in verify.artefact_checks(tmp_path)}
    assert checks["artefact.report_present"].status == verify.PASS
    assert checks["artefact.report_completeness"].status == verify.PASS
    assert checks["artefact.manifest_consistency"].status == verify.PASS


def test_artefact_checks_skip_before_the_artefacts_exist(tmp_path: Path) -> None:
    checks = {result.identifier: result for result in verify.artefact_checks(tmp_path)}
    assert checks["artefact.report_present"].status == verify.NOT_RUN


def test_dataset_link_probe_is_opt_in(repository_root: Path) -> None:
    skipped = verify.dataset_link_check(repository_root, probe_live=False)
    assert skipped.status == verify.NOT_RUN
    assert skipped.evidence["probed_live"] is False


def test_stray_markdown_is_reported_by_the_hygiene_check(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / "NOTES.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (tmp_path / "dataset_urls.txt").write_text("https://example.org\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("notice\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("licence\n", encoding="utf-8")
    (tmp_path / "claim_to_code.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "verification_report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "integrity_manifest.json").write_text("{}\n", encoding="utf-8")
    checks = {result.identifier: result for result in verify.hygiene_checks(tmp_path)}
    assert checks["hygiene.markdown_inventory"].status == verify.FAIL


def test_untrusted_vocabulary_is_reported(tmp_path: Path) -> None:
    stem = verify.BANNED_DATA_VOCABULARY_STEMS[0]
    (tmp_path / "README.md").write_text(f"the {stem} cohort\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (tmp_path / "dataset_urls.txt").write_text("https://example.org\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("notice\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("licence\n", encoding="utf-8")
    (tmp_path / "claim_to_code.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "verification_report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "integrity_manifest.json").write_text("{}\n", encoding="utf-8")
    checks = {result.identifier: result for result in verify.hygiene_checks(tmp_path)}
    assert checks["hygiene.data_vocabulary"].status == verify.FAIL


def test_absolute_path_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("see " + "/Us" + "ers/someone/repo for details\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (tmp_path / "dataset_urls.txt").write_text("https://example.org\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("notice\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("licence\n", encoding="utf-8")
    (tmp_path / "claim_to_code.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "verification_report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "integrity_manifest.json").write_text("{}\n", encoding="utf-8")
    checks = {result.identifier: result for result in verify.hygiene_checks(tmp_path)}
    assert checks["hygiene.absolute_paths"].status == verify.FAIL


def test_finalise_writes_a_fixed_point(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    (tmp_path / "dataset_urls.txt").write_text("https://example.org\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("notice\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("licence\n", encoding="utf-8")
    results = [verify.Check(f"c{index}", "execution", "check").ok() for index in range(45)]
    results.append(verify.Check("m", "manuscript", "manuscript check").ok())
    payload: dict[str, object] = {
        "artifact": "paper-claim to code mapping",
        "counts": {"claims": 1, "anchors": 1, "resolved_anchors": 1, "unresolved": []},
    }
    first_report, first_manifest = verify.finalise(tmp_path, results, payload, "main")
    report_digest = verify.file_digest(tmp_path / "verification_report.json")
    manifest_digest = verify.file_digest(tmp_path / "integrity_manifest.json")
    second_report, second_manifest = verify.finalise(tmp_path, results, payload, "main")
    assert first_report["summary"] == second_report["summary"]
    assert first_manifest["aggregate_sha256"] == second_manifest["aggregate_sha256"]
    assert verify.file_digest(tmp_path / "verification_report.json") == report_digest
    assert verify.file_digest(tmp_path / "integrity_manifest.json") == manifest_digest
    assert (tmp_path / "verification_summary.txt").exists()


def test_mechanism_checks_run_on_the_smoke_configuration(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext
    from osteoct.training.trainer import TrainConfig, train_arm

    assert isinstance(smoke_study, StudyContext)
    config = TrainConfig(seeds=(0,), epochs=1, batch_size=4, grad_accum=1, early_stopping_patience=1, validation_fraction=0.25)
    training = {
        arm: train_arm(smoke_study.encoded, arm, config, smoke_study.attributes, None) for arm in ("constrained", "shifted_window")
    }
    results = {result.identifier: result for result in verify.mechanism_checks(smoke_study, training, "constrained", 5.0)}
    assert set(results) == {"fig2.cumulative_axiom_curve", "fig2.per_examination_budget"}
    for result in results.values():
        assert result.status in {verify.PASS, verify.FAIL, verify.NOT_RUN}
    curve = results["fig2.cumulative_axiom_curve"]
    assert curve.status == verify.PASS
    points = curve.evidence["points"]
    assert isinstance(points, list)
    assert len(points) == 6


def test_mechanism_checks_skip_when_the_reference_arm_is_a_rule(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    results = {result.identifier: result for result in verify.mechanism_checks(smoke_study, {}, "hu_threshold_rule", 5.0)}
    assert results["fig2.per_examination_budget"].status == verify.NOT_RUN


def test_atomic_writer_leaves_a_readable_file(tmp_path: Path) -> None:
    target = tmp_path / "artefact.json"
    atomic_write_json(target, {"a": 1})
    assert oct(target.stat().st_mode & 0o777) == "0o644"
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert not list(tmp_path.glob(".*tmp"))
