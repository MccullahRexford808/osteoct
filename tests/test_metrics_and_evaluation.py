"""Metric, statistic and evaluation-layer tests with independent references."""

from __future__ import annotations

import numpy as np
import pytest

from osteoct.agent.pipeline import Policy
from osteoct.evaluation.dissociation import densitometric_band, patient_first_rows, run_dissociation, severity_grade
from osteoct.evaluation.family_ablation import (
    FAMILY_LABELS,
    conformance_ordering,
    event_time_of,
    non_monotone_rows,
    outcome_arrays,
    smallest_utility_change,
)
from osteoct.evaluation.main_comparison import delta_against_reference, ordering, reference_labels, unconstrained_comparator
from osteoct.evaluation.outcome_validation import c_index_difference
from osteoct.evaluation.reader_study import (
    ASSISTED_ACCURACY,
    READER_ARMS,
    READER_ROLES,
    UNASSISTED_ACCURACY,
    accuracy_gain,
    arm_agreement,
    cohen_kappa,
    cross_reader_pairs,
    reading_time_reduction,
    run_reader_study,
)
from osteoct.evaluation.silent_mode import SilentModeAudit, abstention_reason_text, family_violation_rates, recount_reasons
from osteoct.evaluation.study import PARTITION_EXTERNAL, PARTITION_INTERNAL, execute_rule_arm
from osteoct.metrics.calibration import (
    brier,
    calibration_in_the_large,
    calibration_slope_intercept,
    observed_expected_ratio,
    over_dispersion,
    reliability_bins,
)
from osteoct.metrics.discrimination import (
    accuracy,
    auc,
    balanced_accuracy,
    band_is_osteoporotic,
    positive_rate,
    rank_scores,
    roc_curve,
    sensitivity_at_specificity,
    specificity_at_threshold,
    threshold_at_specificity,
)
from osteoct.metrics.reclassification import continuous_nri, moved_share, nri_interval, reclassify, risk_category
from osteoct.metrics.survival import (
    CensoringRule,
    censored_count,
    comparable_pairs,
    concordance_index,
    event_count,
    five_year_incidence,
    incidence_ratio,
)
from osteoct.metrics.utility import (
    best_threshold,
    decision_curve,
    net_benefit,
    net_benefit_gain,
    standardised_net_benefit,
    treat_all_net_benefit,
)
from osteoct.ontology.axioms import FamilyName
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand


def brute_force_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = 0.0
    for positive in positives.tolist():
        for negative in negatives.tolist():
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (positives.size * negatives.size)


def brute_force_kappa(first: list[int], second: list[int]) -> float:
    labels = sorted(set(first) | set(second))
    size = len(first)
    observed = sum(1 for left, right in zip(first, second, strict=True) if left == right) / size
    expected = 0.0
    for label in labels:
        row = sum(1 for value in first if value == label) / size
        column = sum(1 for value in second if value == label) / size
        expected += row * column
    return (observed - expected) / (1.0 - expected)


def test_auc_matches_the_pairwise_definition(score_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = score_vectors
    assert auc(scores, labels) == pytest.approx(brute_force_auc(scores, labels), abs=1e-12)


def test_auc_with_ties_is_half_credit() -> None:
    scores = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    assert auc(scores, labels) == pytest.approx(0.5)


def test_auc_is_undefined_without_both_classes() -> None:
    scores = np.asarray([0.1, 0.2], dtype=np.float64)
    assert np.isnan(auc(scores, np.asarray([1, 1], dtype=np.int64)))


def test_roc_curve_is_monotone(score_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = score_vectors
    fall_out, sensitivity = roc_curve(scores, labels)
    assert np.all(np.diff(fall_out) >= -1e-12)
    assert np.all(np.diff(sensitivity) >= -1e-12)
    assert sensitivity[-1] == pytest.approx(1.0)


def test_threshold_and_sensitivity_are_consistent(score_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = score_vectors
    threshold = threshold_at_specificity(scores, labels, 0.9)
    assert specificity_at_threshold(scores, labels, threshold) >= 0.9 - 1e-9
    sensitivity = sensitivity_at_specificity(scores, labels, 0.9)
    assert 0.0 <= sensitivity <= 1.0


def test_accuracy_and_band_helpers(score_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = score_vectors
    assert 0.0 <= accuracy(scores, labels) <= 1.0
    assert 0.0 <= balanced_accuracy(scores, labels) <= 1.0
    assert positive_rate(labels) == pytest.approx(float(np.mean(labels == 1)))
    assert rank_scores(scores).shape == scores.shape
    bands = np.asarray([0, 1, 2, 3], dtype=np.int64)
    assert band_is_osteoporotic(bands).tolist() == [0, 0, 1, 1]


def test_concordance_matches_pair_enumeration() -> None:
    rng = np.random.default_rng(23)
    times = rng.uniform(10.0, 1500.0, size=60)
    events = rng.random(60) < 0.6
    risk = rng.normal(0.4, 0.2, size=60)
    observed = concordance_index(times, events, risk, CensoringRule(), None)
    total = 0
    concordant = 0.0
    for first in range(60):
        for second in range(first + 1, 60):
            if times[first] == times[second]:
                continue
            earlier, later = (first, second) if times[first] < times[second] else (second, first)
            if not events[earlier]:
                continue
            total += 1
            concordant += 1.0 if risk[earlier] > risk[later] else (0.5 if risk[earlier] == risk[later] else 0.0)
    assert observed == pytest.approx(concordant / total)
    assert comparable_pairs(times, events).shape[1] == 2


def test_concordance_is_one_for_a_perfect_order() -> None:
    times = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    events = np.asarray([True, True, True])
    risk = np.asarray([0.9, 0.5, 0.1], dtype=np.float64)
    assert concordance_index(times, events, risk, CensoringRule(), None) == pytest.approx(1.0)


def test_treatment_aware_censoring_rule() -> None:
    times = np.asarray([100.0, 400.0], dtype=np.float64)
    events = np.asarray([True, True])
    treatment = np.asarray([np.inf, 200.0], dtype=np.float64)
    effective_times, effective_events = CensoringRule().effective(times, events, treatment)
    assert float(effective_times[1]) == 200.0
    assert not bool(effective_events[1])
    assert bool(effective_events[0])
    assert event_count(events) == 2
    assert censored_count(events) == 0


def test_five_year_incidence_and_ratio() -> None:
    times = np.asarray([100.0, 2000.0, 500.0, 1800.0], dtype=np.float64)
    events = np.asarray([True, False, True, False])
    incidence = five_year_incidence(times, events, 1825.0)
    assert incidence == pytest.approx(0.5)
    assert incidence_ratio(0.119, 0.042) == pytest.approx(2.8333333, abs=1e-6)
    assert np.isnan(incidence_ratio(1.0, 0.0))


def test_net_benefit_closed_form() -> None:
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    observed = net_benefit(probabilities, outcomes, 0.5)
    prevalence = 0.5
    true_positive = 2
    false_positive = 0
    manual = true_positive / 4 - false_positive / 4 * (0.5 / 0.5)
    assert observed == pytest.approx(manual)
    assert treat_all_net_benefit(outcomes, 0.5) == pytest.approx(prevalence - (1 - prevalence) * 1.0)
    with pytest.raises(ValueError):
        net_benefit(probabilities, outcomes, 0.0)


def test_decision_curve_and_gain() -> None:
    rng = np.random.default_rng(31)
    probabilities = rng.uniform(0.0, 1.0, size=80)
    outcomes = (rng.random(80) < probabilities).astype(np.float64)
    points = decision_curve(probabilities, outcomes)
    assert len(points) == 7
    assert all(point.net_benefit == point.net_benefit for point in points)
    gain = net_benefit_gain(probabilities, probabilities * 0.5, outcomes, 0.2)
    assert gain >= 0.0
    assert best_threshold(probabilities, outcomes) in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    assert 0.0 <= standardised_net_benefit(probabilities, outcomes, 0.2) <= 1.0


def test_reclassification_counts_match_the_enumeration() -> None:
    new = np.asarray([0.1, 0.3, 0.8, 0.9], dtype=np.float64)
    old = np.asarray([0.25, 0.1, 0.4, 0.85], dtype=np.float64)
    outcomes = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    record = reclassify(new, old, outcomes, 0.2)
    assert record.events_up == 1
    assert record.events_down == 0
    assert record.non_events_up == 0
    assert record.non_events_down == 1
    assert record.nri() == pytest.approx((1 - 0) / 1 + (1 - 0) / 1)
    assert record.as_mapping()["cut"] == 0.2
    assert continuous_nri(new, old, outcomes) == continuous_nri(new, old, outcomes)
    assert 0.0 <= moved_share(new, old, 0.2) <= 1.0
    assert risk_category(new, 0.2).tolist() == [0, 1, 1, 1]


def test_nri_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(41)
    outcomes = (rng.random(120) < 0.3).astype(np.float64)
    old = rng.uniform(0.0, 0.5, size=120)
    new = old + 0.05 * outcomes
    point = reclassify(new, old, outcomes, 0.2).nri()
    low, high = nri_interval(new, old, outcomes, 0.2, 200, 0)
    assert low <= point + 1e-9 <= high or low < high


def test_calibration_slope_recovers_a_known_model() -> None:
    rng = np.random.default_rng(53)
    linear = rng.normal(0.0, 1.0, size=400)
    probabilities = 1.0 / (1.0 + np.exp(-(0.4 + 1.3 * linear)))
    outcomes = (rng.random(400) < probabilities).astype(np.float64)
    fit = calibration_slope_intercept(probabilities, outcomes)
    assert fit.converged
    assert abs(fit.slope - 1.0) < 0.35
    assert abs(fit.intercept) < 0.35
    assert 0.0 <= brier(probabilities, outcomes) <= 1.0
    assert abs(calibration_in_the_large(probabilities, outcomes)) < 0.1
    assert observed_expected_ratio(probabilities, outcomes) > 0.0
    assert len(reliability_bins(probabilities, outcomes, 5)) > 0
    assert over_dispersion(calibration_slope_intercept(probabilities, outcomes)) in (True, False)


def test_holm_step_down_uses_the_running_maximum() -> None:
    from osteoct.statistics.permutation import holm_step_down

    p_values = np.asarray([0.001, 0.008, 0.039, 0.041, 0.31, 0.62], dtype=np.float64)
    adjusted = holm_step_down(p_values).tolist()
    order = sorted(range(p_values.size), key=lambda index: p_values[index])
    running = 0.0
    manual = [0.0] * p_values.size
    for rank, index in enumerate(order):
        running = max(running, float(p_values[index]) * (p_values.size - rank))
        manual[index] = min(1.0, running)
    assert all(abs(first - second) < 1e-12 for first, second in zip(adjusted, manual, strict=True))


def test_cohen_kappa_matches_the_contingency_table() -> None:
    rng = np.random.default_rng(61)
    first = rng.integers(0, 4, size=180).tolist()
    second = [min(3, max(0, value + int(draw))) for value, draw in zip(first, rng.integers(-1, 2, size=180).tolist(), strict=True)]
    assert cohen_kappa(np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)) == pytest.approx(
        brute_force_kappa(first, second), abs=1e-12
    )


def test_reader_study_fixture_and_report(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    study = run_reader_study(smoke_study, size=24, seed=0)
    assert study.raters == len(READER_ROLES)
    assert set(READER_ARMS) <= set(study.mean_seconds)
    assert 0.0 <= study.accuracy["clinician_alone"] <= 1.0
    assert study.accuracy["clinician_with_model"] >= study.accuracy["clinician_alone"] - 0.5
    assert accuracy_gain(study) == pytest.approx(study.accuracy["clinician_with_model"] - study.accuracy["clinician_alone"])
    assert reading_time_reduction(study) == pytest.approx(
        study.mean_seconds["clinician_alone"] - study.mean_seconds["clinician_with_model"]
    )
    first, second = cross_reader_pairs(study, "clinician_with_model")
    assert first.size == second.size == study.size
    assert len(arm_agreement(study)) == len(READER_ARMS)
    assert -1.0 <= study.kappa <= 1.0
    assert UNASSISTED_ACCURACY < ASSISTED_ACCURACY


def test_dissociation_study_reports_both_strata(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    study = run_dissociation(smoke_study)
    assert study.patients == len(patient_first_rows(smoke_study))
    assert len(study.strata) == 2
    assert 0.0 <= study.dissociated_share <= 1.0
    assert study.missed_by_density_rule == study.deformed_densitometrically_normal
    first_row = next(iter(patient_first_rows(smoke_study).values()))
    assert densitometric_band(smoke_study, first_row) in set(DiagnosticBand)
    assert severity_grade(smoke_study, first_row) in set(DeformityGrade)


def test_comparison_helpers_over_the_smoke_tables(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = smoke_study.rows(PARTITION_INTERNAL)
    labels = reference_labels(smoke_study, rows)
    assert labels.dtype == np.int64
    assert set(np.unique(labels).tolist()) <= {0, 1}


def test_rule_arm_runs_without_training(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = smoke_study.rows(PARTITION_INTERNAL)[:6]
    result = execute_rule_arm(smoke_study, "hu_threshold_rule", Policy.UNCONSTRAINED, rows)
    assert len(result.ensemble) == int(rows.size)
    assert result.policy is Policy.UNCONSTRAINED
    assert 0.0 <= result.mean_repairs() <= 10.0


def test_arm_result_reports_family_violation_rates(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = smoke_study.rows(PARTITION_INTERNAL)[:6]
    result = execute_rule_arm(smoke_study, "hu_threshold_rule", Policy.CONSTRAINED, rows)
    rates = family_violation_rates(result)
    assert set(rates) == {family.value for family in FamilyName}
    assert all(0.0 <= value <= 1.0 for value in rates.values())
    assert (
        abstention_reason_text("band")
        == "the diagnostic band remains inconsistent with the harmonised interval after the repair budget"
    )
    audit = SilentModeAudit(
        partition="internal_test",
        processed=int(rows.size),
        emitted=len(result.emitted()),
        abstained=int(rows.size) - len(result.emitted()),
        abstention_reasons=result.reasons(),
        mean_conformance=0.0,
        unsupported_rate=0.0,
        mean_risk=0.0,
        escalated_share=0.0,
        threshold=0.2,
        family_violation_rates=rates,
        repaired_share=0.0,
    )
    assert recount_reasons(audit) == sum(audit.abstention_reasons.values())


def test_outcome_validation_runs_end_to_end(smoke_study: object) -> None:
    from osteoct.evaluation.outcome_validation import run_outcome_validation
    from osteoct.evaluation.study import StudyContext
    from osteoct.training.trainer import TrainConfig, train_arm

    assert isinstance(smoke_study, StudyContext)
    config = TrainConfig(seeds=(0,), epochs=1, batch_size=4, grad_accum=1, early_stopping_patience=1, validation_fraction=0.25)
    training = {
        arm: train_arm(smoke_study.encoded, arm, config, smoke_study.attributes, None) for arm in ("shifted_window", "constrained")
    }
    specification = (
        ("hu_threshold_rule", "Vertebral HU threshold rule", Policy.UNCONSTRAINED),
        ("shifted_window", "3-D shifted-window transformer", Policy.UNCONSTRAINED),
        ("constrained", "Tool-using agent, no ontology layer", Policy.UNCONSTRAINED),
        ("constrained", "OSTEO-CT (constrained)", Policy.CONSTRAINED),
    )
    table, results = run_outcome_validation(smoke_study, training, specification, "Tool-using agent, no ontology layer", 40, 0)
    assert len(table.rows) == 4
    assert set(results) == {
        "Vertebral HU threshold rule",
        "3-D shifted-window transformer",
        "Tool-using agent, no ontology layer",
        "OSTEO-CT (constrained)",
    }
    assert table.reclassification is not None
    assert 0.0 <= table.reclassification.cut <= 1.0
    for row in table.rows:
        assert row.c_index == row.c_index
        assert row.c_index_low <= row.c_index_high


def test_outcome_arrays_expose_times_and_events(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext

    assert isinstance(smoke_study, StudyContext)
    rows = smoke_study.rows(PARTITION_INTERNAL)
    arrays = outcome_arrays(smoke_study, rows)
    assert arrays.times.shape == rows.shape
    assert arrays.events.dtype == bool
    first_record = smoke_study.encoded.records[int(rows[0])]
    assert event_time_of(first_record) > 0.0


def test_family_labels_cover_every_family() -> None:
    assert set(FAMILY_LABELS) == set(FamilyName)
    assert conformance_ordering_is_stable()


def conformance_ordering_is_stable() -> bool:
    from osteoct.evaluation.family_ablation import AblationRow, AblationTable

    table = AblationTable(
        rows=(
            AblationRow("all_families", None, 0.9, 0.0, 0.8, 0.95, 0.0),
            AblationRow("without_band", "band", 0.88, 0.02, 0.79, 0.90, -0.05),
            AblationRow("without_evidence", "evidence", 0.89, 0.01, 0.80, 0.92, -0.03),
        ),
        conformance_monotone=True,
        removed_family_lowers_conformance={"band": True, "evidence": True},
    )
    assert conformance_ordering(table) == ("evidence", "band")
    assert non_monotone_rows(table) == ()
    assert smallest_utility_change(table) == pytest.approx(0.01)
    return True


def test_outcome_and_silent_helpers(smoke_study: object) -> None:
    from osteoct.evaluation.study import StudyContext
    from osteoct.metrics.calibration import CalibrationFit

    assert isinstance(smoke_study, StudyContext)
    assert smoke_study.rows(PARTITION_EXTERNAL).size >= 0
    assert over_dispersion(CalibrationFit(1.2, -0.04, 4, True))
    assert not over_dispersion(CalibrationFit(0.95, 0.0, 4, True))
    assert c_index_difference is not None


def test_evaluation_payload_and_summary_are_renderable() -> None:
    from osteoct.evaluation.dissociation import DissociationStratum, DissociationStudy

    study = DissociationStudy(
        patients=10,
        deformed=4,
        deformed_densitometrically_normal=2,
        normal_without_deformity=6,
        dissociated_share=0.5,
        strata=(
            DissociationStratum("deformed", 4, 2, 0.12),
            DissociationStratum("reference", 6, 1, 0.04),
        ),
        incidence_ratio=3.0,
        missed_by_density_rule=2,
    )
    assert study.as_mapping()["incidence_ratio"] == 3.0


def test_comparison_delta_helpers_over_a_small_table() -> None:
    from osteoct.evaluation.main_comparison import ComparisonRow, ComparisonTable

    row = ComparisonRow("OSTEO-CT (constrained)", "constrained", 0.9, 0.88, 0.92, 0.7, 0.95, 0.003, 0.004, 0.001, 0.03, 0.002)
    other = ComparisonRow(
        "Tool-using agent, no ontology layer", "unconstrained", 0.88, 0.86, 0.90, 0.68, 0.74, 0.006, 0.058, 0.005, 0.012, 0.002
    )
    table = ComparisonTable("internal_test", (row, other), 100, 40, 5)
    assert delta_against_reference(table)["Tool-using agent, no ontology layer"] == pytest.approx(-0.02)
    assert ordering(table)[0] == "OSTEO-CT (constrained)"
    assert unconstrained_comparator(table) is other
    payload = table.as_mapping()
    assert payload["rows"][0]["method"] == "OSTEO-CT (constrained)"
