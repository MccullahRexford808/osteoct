"""The five axiom families, their source standards and their satisfiability checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from osteoct.ontology.bands import BandCalibration, band_consistent, band_from_interval
from osteoct.ontology.index import CODE_TO_ELEMENT
from osteoct.ontology.profile import Expression, atom, existential, role_hierarchy
from osteoct.ontology.severity import GenantThresholds, grade_from_vertebrae
from osteoct.ontology.tiers import ProtocolEnvelope, tier_from_band
from osteoct.schema.concepts import ConceptIndex
from osteoct.schema.records import CandidateDecision, MeasurementBundle
from osteoct.schema.vocabulary import DiagnosticBand


class FamilyName(str, Enum):
    IDENTITY = "id"
    BAND = "band"
    TIER = "tier"
    SEVERITY = "sev"
    EVIDENCE = "ev"


FAMILY_ORDER: tuple[FamilyName, ...] = (
    FamilyName.IDENTITY,
    FamilyName.BAND,
    FamilyName.TIER,
    FamilyName.SEVERITY,
    FamilyName.EVIDENCE,
)

REPAIRABLE_ORDER: tuple[FamilyName, ...] = (
    FamilyName.BAND,
    FamilyName.TIER,
    FamilyName.SEVERITY,
    FamilyName.EVIDENCE,
)


@dataclass(frozen=True, slots=True)
class AxiomFamily:
    name: FamilyName
    source_standard: str
    representative_axiom: str
    mode: str

    def is_hard(self) -> bool:
        return self.mode == "hard"


FAMILY_TABLE: dict[FamilyName, AxiomFamily] = {
    FamilyName.IDENTITY: AxiomFamily(
        FamilyName.IDENTITY,
        "SNOMED CT, RadLex, LOINC, DICOM SR",
        "Every emitted concept resolves to a published identifier in the released index",
        "hard",
    ),
    FamilyName.BAND: AxiomFamily(
        FamilyName.BAND,
        "ISCD official positions / WHO T-score bands",
        "The diagnostic band is consistent with the harmonised attenuation interval and the reference-available density bands",
        "hard",
    ),
    FamilyName.TIER: AxiomFamily(
        FamilyName.TIER,
        "FRAX-based risk semantics",
        "The risk tier follows from the band, the prior-fracture state and the treatment-exposure state",
        "hard",
    ),
    FamilyName.SEVERITY: AxiomFamily(
        FamilyName.SEVERITY,
        "Genant semiquantitative grading",
        "The reported deformity grade follows from the measured vertebral morphology",
        "hard",
    ),
    FamilyName.EVIDENCE: AxiomFamily(
        FamilyName.EVIDENCE,
        "internal audit requirement",
        "Every asserted element is linked to the measurement that produced it",
        "soft",
    ),
}

FAMILY_WEIGHTS: dict[FamilyName, float] = {
    FamilyName.IDENTITY: 0.30,
    FamilyName.BAND: 0.28,
    FamilyName.TIER: 0.20,
    FamilyName.SEVERITY: 0.10,
    FamilyName.EVIDENCE: 0.12,
}

TOTAL_FAMILY_WEIGHT: float = float(sum(FAMILY_WEIGHTS.values()))

ASSERTED_ELEMENTS: dict[str, tuple[str, ...]] = {
    "diagnostic_band": ("m_att_star", "u_att"),
    "risk_tier": ("band", "prior_fracture_state", "treatment_exposure_state", "acquisition_descriptor", "risk_semantics"),
    "deformity_grade": ("morphology",),
}

EVIDENCE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "trabecular_attenuation": ("m_att",),
    "attenuation_uncertainty": ("u_att",),
    "areal_bone_mineral_density": ("reference_density",),
    "volumetric_bone_mineral_density": ("density_indicator",),
    "vertebral_level": ("localisation",),
    "deformity_grade": ("morphology",),
    "diagnostic_band": ("m_att_star", "u_att"),
    "risk_tier": ("band", "prior_fracture_state", "treatment_exposure_state", "acquisition_descriptor", "risk_semantics"),
    "measurement_method": ("acquisition_descriptor",),
    "evidence_link": (),
}

FAMILY_AXIOM_EXPRESSIONS: dict[FamilyName, Expression] = {
    FamilyName.IDENTITY: atom("IdentifiedDecision"),
    FamilyName.BAND: existential("derived_from", atom("HarmonisedInterval")),
    FamilyName.TIER: existential("derived_from", atom("DiagnosticBand")),
    FamilyName.SEVERITY: existential("derived_from", atom("VertebralMorphology")),
    FamilyName.EVIDENCE: role_hierarchy("linked_to", "evidenced_by"),
}


@dataclass(frozen=True, slots=True)
class Violation:
    family: FamilyName
    detail: str

    def label(self) -> str:
        return self.family.value


@dataclass(frozen=True, slots=True)
class AxiomSet:
    index: ConceptIndex
    calibration: BandCalibration
    thresholds: GenantThresholds
    envelope: ProtocolEnvelope
    active: frozenset[FamilyName] = frozenset(FAMILY_ORDER)

    def all_families(self) -> tuple[FamilyName, ...]:
        return tuple(family for family in FAMILY_ORDER if family in self.active)

    def hard_families(self) -> tuple[FamilyName, ...]:
        return tuple(family for family in self.all_families() if FAMILY_TABLE[family].is_hard())

    def soft_families(self) -> tuple[FamilyName, ...]:
        return tuple(family for family in self.all_families() if not FAMILY_TABLE[family].is_hard())

    def repairable_families(self) -> tuple[FamilyName, ...]:
        return tuple(family for family in REPAIRABLE_ORDER if family in self.active)

    def without_family(self, family: FamilyName) -> AxiomSet:
        return AxiomSet(self.index, self.calibration, self.thresholds, self.envelope, frozenset(set(self.active) - {family}))

    def with_only(self, families: tuple[FamilyName, ...]) -> AxiomSet:
        return AxiomSet(self.index, self.calibration, self.thresholds, self.envelope, frozenset(families))

    def violations(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        found: list[Violation] = []
        for family in self.all_families():
            found.extend(self._family_violations(family, candidate, bundle))
        return tuple(found)

    def first_violation(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> Violation | None:
        for family in self.all_families():
            found = self._family_violations(family, candidate, bundle)
            if found:
                return found[0]
        return None

    def hard_violations(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        return tuple(violation for violation in self.violations(candidate, bundle) if FAMILY_TABLE[violation.family].is_hard())

    def soft_violations(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        return tuple(violation for violation in self.violations(candidate, bundle) if not FAMILY_TABLE[violation.family].is_hard())

    def satisfies(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> bool:
        return not self.hard_violations(candidate, bundle)

    def _family_violations(
        self, family: FamilyName, candidate: CandidateDecision, bundle: MeasurementBundle
    ) -> tuple[Violation, ...]:
        if family is FamilyName.IDENTITY:
            return self._identity(candidate)
        if family is FamilyName.BAND:
            return self._band(candidate, bundle)
        if family is FamilyName.TIER:
            return self._tier(candidate, bundle)
        if family is FamilyName.SEVERITY:
            return self._severity(candidate, bundle)
        return self._evidence(candidate, bundle)

    def _identity(self, candidate: CandidateDecision) -> tuple[Violation, ...]:
        unresolved = self.unresolved(candidate)
        if not unresolved:
            return ()
        return (Violation(FamilyName.IDENTITY, "unresolved identifiers: " + ", ".join(unresolved)),)

    def unresolved(self, candidate: CandidateDecision) -> tuple[str, ...]:
        table = self.index.alias_table()
        return tuple(token for token in candidate.concepts if token not in table)

    def _band(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        consistent = band_consistent(
            candidate.band,
            bundle.standard_attenuation_hu,
            bundle.uncertainty_hu,
            self.calibration,
            candidate.context,
        )
        if consistent:
            return ()
        return (Violation(FamilyName.BAND, "band inconsistent with the harmonised interval or the reference density"),)

    def _tier(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        expected = tier_from_band(
            candidate.band,
            candidate.context,
            bundle.acquisition,
            candidate.concepts,
            self.index,
            self.envelope,
        )
        if candidate.tier is expected:
            return ()
        return (Violation(FamilyName.TIER, f"tier {candidate.tier.value!r} does not follow from band {candidate.band.value!r}"),)

    def _severity(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        expected = grade_from_vertebrae(tuple(measurement.morphology for measurement in bundle.vertebrae), self.thresholds)
        if candidate.severity is expected:
            return ()
        return (Violation(FamilyName.SEVERITY, f"grade {candidate.severity.value!r} does not follow from the measured morphology"),)

    def _evidence(self, candidate: CandidateDecision, bundle: MeasurementBundle) -> tuple[Violation, ...]:
        required = self.required_evidence(candidate)
        missing = tuple(token for token in required if token not in candidate.evidence)
        if not missing:
            return ()
        return (Violation(FamilyName.EVIDENCE, "elements without a measurement link: " + ", ".join(sorted(set(missing)))),)

    def required_evidence(self, candidate: CandidateDecision) -> tuple[str, ...]:
        required: list[str] = []
        for element in self.asserted_elements(candidate):
            if element in ASSERTED_ELEMENTS:
                required.extend(ASSERTED_ELEMENTS[element])
            required.extend(EVIDENCE_REQUIREMENTS.get(element, ()))
        if not candidate.context.reference_density_available():
            required = [token for token in required if token != "reference_density"]
        return tuple(dict.fromkeys(required))

    def asserted_elements(self, candidate: CandidateDecision) -> tuple[str, ...]:
        elements = ["diagnostic_band", "risk_tier", "deformity_grade"]
        table = self.index.alias_table()
        for token in candidate.concepts:
            concept = table.get(token)
            if concept is None:
                continue
            element = CODE_TO_ELEMENT.get(concept.code)
            if element is not None:
                elements.append(element)
        return tuple(dict.fromkeys(elements))


def conformance_score(
    axiom_set: AxiomSet,
    candidate: CandidateDecision,
    bundle: MeasurementBundle,
) -> float:
    violated = {violation.family for violation in axiom_set.violations(candidate, bundle)}
    satisfied = sum(FAMILY_WEIGHTS[family] for family in FAMILY_ORDER if family in axiom_set.active and family not in violated)
    return satisfied / TOTAL_FAMILY_WEIGHT


def supported_code(candidate: CandidateDecision, axiom_set: AxiomSet, bundle: MeasurementBundle) -> bool:
    if axiom_set.unresolved(candidate):
        return False
    if not band_consistent(
        candidate.band,
        bundle.standard_attenuation_hu,
        bundle.uncertainty_hu,
        axiom_set.calibration,
        candidate.context,
    ):
        return False
    required = axiom_set.required_evidence(candidate)
    return all(token in candidate.evidence for token in required)


def conformance_ceiling(axiom_set: AxiomSet) -> float:
    return sum(FAMILY_WEIGHTS[family] for family in axiom_set.all_families()) / TOTAL_FAMILY_WEIGHT


def band_after_interval(axiom_set: AxiomSet, bundle: MeasurementBundle) -> DiagnosticBand:
    return band_from_interval(bundle.standard_attenuation_hu, bundle.uncertainty_hu, axiom_set.calibration)
