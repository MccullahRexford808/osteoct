"""Rendering of the evaluation payload and its plain-text summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osteoct.evaluation.dissociation import DissociationStudy
from osteoct.evaluation.family_ablation import AblationTable
from osteoct.evaluation.findings import FindingSet
from osteoct.evaluation.main_comparison import ComparisonTable
from osteoct.evaluation.outcome_validation import OutcomeTable
from osteoct.evaluation.reader_study import ReaderStudy, reading_time_reduction
from osteoct.evaluation.silent_mode import SilentModeAudit
from osteoct.utils.io import atomic_write_json, atomic_write_text


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    experiment: str
    comparison: ComparisonTable
    ablation: AblationTable
    outcomes: OutcomeTable
    readers: ReaderStudy
    silent: SilentModeAudit
    dissociation: DissociationStudy
    findings: FindingSet
    seeds: int
    compute_seconds: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "seeds": self.seeds,
            "table_main_comparison": self.comparison.as_mapping(),
            "table_constraint_ablation": self.ablation.as_mapping(),
            "table_clinical_outcome": self.outcomes.as_mapping(),
            "reading_study": self.readers.as_mapping(),
            "silent_mode_audit": self.silent.as_mapping(),
            "dissociation": self.dissociation.as_mapping(),
            "findings": self.findings.as_mapping(),
        }


def build_payload(bundle: EvaluationBundle) -> dict[str, object]:
    return bundle.as_mapping()


def render_summary(bundle: EvaluationBundle) -> str:
    lines: list[str] = []
    lines.append(f"experiment {bundle.experiment}")
    lines.append(f"seeds {bundle.seeds}")
    lines.append("")
    lines.append("main comparison, internal multicentre test partition")
    lines.append(f"  labelled examinations {bundle.comparison.labelled_examinations}")
    lines.append(f"  patients with fracture follow-up {bundle.comparison.patients_with_followup}")
    lines.append("  method | AUROC (95% CI) | sensitivity at 90% specificity | conformance | unsupported | abstention")
    for comparison_row in bundle.comparison.rows:
        lines.append(
            "  "
            f"{comparison_row.method} | {comparison_row.auroc:.3f} ({comparison_row.auroc_low:.3f}-{comparison_row.auroc_high:.3f}) | "
            f"{comparison_row.sensitivity_at_90:.3f} | {comparison_row.conformance_mean:.3f} +- {comparison_row.conformance_sd:.3f} | "
            f"{comparison_row.unsupported_mean:.3f} +- {comparison_row.unsupported_sd:.3f} | "
            f"{comparison_row.abstention_mean:.3f} +- {comparison_row.abstention_sd:.3f}"
        )
    lines.append("")
    lines.append("constraint-family ablation and component substitution")
    lines.append("  configuration | AUROC | delta AUROC | C-index | conformance | delta conformance")
    for ablation_row in bundle.ablation.rows:
        lines.append(
            "  "
            f"{ablation_row.configuration} | {ablation_row.auroc:.3f} | {ablation_row.auroc_delta:+.3f} | "
            f"{ablation_row.c_index:.3f} | {ablation_row.conformance:.3f} | {ablation_row.conformance_delta:+.3f}"
        )
    lines.append(f"  every family removal lowers conformance: {bundle.ablation.conformance_monotone}")
    lines.append("")
    lines.append("clinical-outcome validation, internal multicentre test partition")
    lines.append("  method | C-index (95% CI) | calibration slope | calibration intercept | NB at 0.20 | NB at 0.30")
    for outcome_row in bundle.outcomes.rows:
        lines.append(
            "  "
            f"{outcome_row.method} | {outcome_row.c_index:.3f} ({outcome_row.c_index_low:.3f}-{outcome_row.c_index_high:.3f}) | "
            f"{outcome_row.calibration.slope:.3f} | {outcome_row.calibration.intercept:+.3f} | "
            f"{outcome_row.net_benefit_20:.3f} | {outcome_row.net_benefit_30:.3f}"
        )
    if bundle.outcomes.reclassification is not None:
        record = bundle.outcomes.reclassification
        lines.append(
            f"  reclassification against the comparator at cut {record.cut:.2f}: "
            f"NRI {record.nri():+.3f} (events {record.events_up} up {record.events_down} down, "
            f"non-events {record.non_events_up} up {record.non_events_down} down)"
        )
    lines.append("")
    lines.append("reading study and deployment audit")
    lines.append(f"  studies {bundle.readers.size}, raters {bundle.readers.raters}")
    lines.append(f"  agreement kappa {bundle.readers.kappa:.3f} ({bundle.readers.kappa_low:.3f}-{bundle.readers.kappa_high:.3f})")
    lines.append(
        "  mean reading seconds " + ", ".join(f"{key} {value:.2f}" for key, value in sorted(bundle.readers.mean_seconds.items()))
    )
    lines.append(f"  reading time reduction {reading_time_reduction(bundle.readers):.3f} seconds")
    lines.append(
        f"  silent mode processed {bundle.silent.processed}, emitted {bundle.silent.emitted}, "
        f"abstained {bundle.silent.abstained}, escalated share {bundle.silent.escalated_share:.4f}"
    )
    lines.append("")
    lines.append("deformity and densitometric dissociation")
    lines.append(
        f"  patients {bundle.dissociation.patients}, deformed {bundle.dissociation.deformed}, "
        f"deformed and densitometrically normal or osteopenic {bundle.dissociation.deformed_densitometrically_normal}"
    )
    lines.append(f"  dissociated share {bundle.dissociation.dissociated_share:.4f}")
    lines.append(f"  incidence ratio {bundle.dissociation.incidence_ratio:.4f}")
    lines.append(f"  missed by a density-only rule {bundle.dissociation.missed_by_density_rule}")
    lines.append("")
    lines.append("named findings")
    for finding in bundle.findings.findings:
        lines.append(f"  {finding.identifier}: holds={finding.holds}")
        lines.append(f"    {finding.statement}")
        for key in sorted(finding.evidence):
            lines.append(f"    {key} = {finding.evidence[key]}")
    lines.append("")
    lines.append(
        "Every cohort-level value above is produced from the schema-compatible cohort fixture shipped with "
        "this release, not from the study cohort, which is not distributed."
    )
    return "\n".join(lines) + "\n"


def write_artefacts(bundle: EvaluationBundle, root: Path) -> tuple[Path, Path]:
    payload = build_payload(bundle)
    json_path = root / "evaluation.json"
    text_path = root / "evaluation_summary.txt"
    summary = render_summary(bundle)
    atomic_write_json(json_path, payload)
    atomic_write_text(text_path, summary)
    return (json_path, text_path)
