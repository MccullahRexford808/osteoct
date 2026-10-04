"""The two named findings, assembled from the executed tables."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.evaluation.dissociation import DissociationStudy
from osteoct.evaluation.family_ablation import AblationTable
from osteoct.evaluation.main_comparison import ComparisonTable
from osteoct.evaluation.outcome_validation import OutcomeTable

FINDING_ONE_ID = "finding_1_unordered_axes"
FINDING_TWO_ID = "finding_2_dissociation"

FINDING_ONE_STATEMENT = (
    "For a decision layer constrained by an ontology of clinical standards, the pair consisting of agreement "
    "with the reference assessment and conformance with the encoded standard admits no monotone order: adding "
    "an axiom can raise conformance while lowering agreement."
)

FINDING_TWO_STATEMENT = (
    "Deformity and densitometric status dissociate, and the dissociation carries prognostic weight: a stratum "
    "of patients graded at or above the moderate deformity floor is densitometrically normal or osteopenic, and "
    "that stratum carries a higher five-year incidence of major osteoporotic fracture than densitometrically "
    "normal patients without deformity."
)


@dataclass(frozen=True, slots=True)
class Finding:
    identifier: str
    statement: str
    evidence: dict[str, object]
    holds: bool

    def as_mapping(self) -> dict[str, object]:
        return {"id": self.identifier, "statement": self.statement, "holds": self.holds, "evidence": self.evidence}


@dataclass(frozen=True, slots=True)
class FindingSet:
    findings: tuple[Finding, ...]

    def as_mapping(self) -> dict[str, object]:
        return {"findings": [finding.as_mapping() for finding in self.findings], "count": len(self.findings)}

    def get(self, identifier: str) -> Finding:
        for finding in self.findings:
            if finding.identifier == identifier:
                return finding
        raise KeyError(f"unknown finding {identifier!r}")


def assemble_findings(
    comparison: ComparisonTable,
    ablation: AblationTable,
    outcomes: OutcomeTable,
    dissociation: DissociationStudy,
    constrained_method: str,
) -> FindingSet:
    base = ablation.row("all_families")
    utility_rises = any(entry.family is not None and entry.c_index > base.c_index for entry in ablation.rows)
    conformance_rises_with_utility_rise = any(
        entry.family is not None and entry.conformance_delta > 0.0 and entry.c_index < base.c_index for entry in ablation.rows
    )
    finding_one = Finding(
        identifier=FINDING_ONE_ID,
        statement=FINDING_ONE_STATEMENT,
        evidence={
            "conformance_monotone_under_removal": ablation.conformance_monotone,
            "utility_increases_without_a_family": utility_rises,
            "agreement_and_conformance_move_in_opposite_directions": conformance_rises_with_utility_rise,
            "reference_auroc": round(base.auroc, 4),
            "reference_c_index": round(base.c_index, 4),
            "reference_conformance": round(base.conformance, 4),
        },
        holds=bool(ablation.conformance_monotone and utility_rises),
    )
    reference_incidence = dissociation.strata[1].five_year_incidence if len(dissociation.strata) > 1 else 0.0
    deformed_incidence = dissociation.strata[0].five_year_incidence if dissociation.strata else 0.0
    finding_two = Finding(
        identifier=FINDING_TWO_ID,
        statement=FINDING_TWO_STATEMENT,
        evidence={
            "patients": dissociation.patients,
            "deformed_patients": dissociation.deformed,
            "deformed_and_densitometrically_normal": dissociation.deformed_densitometrically_normal,
            "dissociated_share": round(dissociation.dissociated_share, 4),
            "incidence_deformed_normal_density": round(deformed_incidence, 4),
            "incidence_normal_without_deformity": round(reference_incidence, 4),
            "incidence_ratio": round(dissociation.incidence_ratio, 4),
            "missed_by_density_only_rule": dissociation.missed_by_density_rule,
        },
        holds=bool(dissociation.deformed_densitometrically_normal > 0 and dissociation.incidence_ratio > 1.0),
    )
    return FindingSet(findings=(finding_one, finding_two))


def axes_disagree(comparison: ComparisonTable, ablation: AblationTable, left: str, right: str) -> bool:
    left_row = comparison.row(left)
    right_row = comparison.row(right)
    conformance_moves = (right_row.conformance_mean - left_row.conformance_mean) * (right_row.auroc - left_row.auroc)
    del ablation
    return bool(conformance_moves < 0.0)


def outcome_corroboration(ablation: AblationTable, outcomes: OutcomeTable) -> dict[str, float]:
    base = ablation.row("all_families")
    return {
        "ablation_c_index": round(base.c_index, 4),
        "table_c_index": round(outcomes.row("OSTEO-CT (constrained)").c_index, 4),
    }
