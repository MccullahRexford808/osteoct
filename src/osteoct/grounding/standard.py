"""Binding of every measured element to an identifier in the released concept index."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.ontology.index import entry_for
from osteoct.schema.concepts import Concept, ConceptIndex, TermSystem
from osteoct.schema.records import ConceptBinding, VertebraRegion
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, RiskTier

BAND_CONCEPT_CODE: dict[DiagnosticBand, str] = {
    DiagnosticBand.NORMAL: "OSTEOCT-1003",
    DiagnosticBand.OSTEOPENIA: "OSTEOCT-1002",
    DiagnosticBand.OSTEOPOROSIS: "OSTEOCT-1001",
    DiagnosticBand.SEVERE_OSTEOPOROSIS: "OSTEOCT-1004",
}

HISTORY_CONCEPT_CODE: dict[bool, str] = {True: "OSTEOCT-1005", False: "OSTEOCT-1003"}

MEASUREMENT_ELEMENTS: tuple[str, ...] = (
    "vertebral_level",
    "trabecular_attenuation",
    "attenuation_uncertainty",
    "volumetric_bone_mineral_density",
    "measurement_method",
    "evidence_link",
)


@dataclass(frozen=True, slots=True)
class Grounding:
    concepts: tuple[str, ...]
    bindings: tuple[ConceptBinding, ...]

    def concept_keys(self) -> tuple[str, ...]:
        return tuple(binding.concept_key for binding in self.bindings)

    def binding_for(self, element: str) -> tuple[ConceptBinding, ...]:
        return tuple(binding for binding in self.bindings if binding.element == element)

    def extended(self, extra: tuple[ConceptBinding, ...]) -> Grounding:
        merged = list(self.bindings) + list(extra)
        ordered: list[str] = []
        for binding in merged:
            if binding.concept_key not in ordered:
                ordered.append(binding.concept_key)
        return Grounding(concepts=tuple(ordered), bindings=tuple(merged))


def concept_key(system: TermSystem, code: str) -> str:
    return Concept(system=system, code=code, display="").key()


def ground_measurements(
    regions: tuple[VertebraRegion, ...],
    standard_attenuation_hu: float,
    uncertainty_hu: float,
    density: float,
) -> Grounding:
    bindings: list[ConceptBinding] = []
    level_entry = entry_for("vertebral_level", TermSystem.RADLEX)
    for region in regions:
        bindings.append(
            ConceptBinding(
                element="vertebral_level",
                concept_key=concept_key(level_entry.system, level_entry.key),
                measurement=f"localisation:{region.level.value}",
            )
        )
    attenuation_entry = entry_for("trabecular_attenuation", TermSystem.LOINC)
    bindings.append(
        ConceptBinding(
            element="trabecular_attenuation",
            concept_key=concept_key(attenuation_entry.system, attenuation_entry.key),
            measurement=f"m_att_star:{standard_attenuation_hu:.4f}",
        )
    )
    uncertainty_entry = entry_for("attenuation_uncertainty", TermSystem.LOINC)
    bindings.append(
        ConceptBinding(
            element="attenuation_uncertainty",
            concept_key=concept_key(uncertainty_entry.system, uncertainty_entry.key),
            measurement=f"u_att:{uncertainty_hu:.4f}",
        )
    )
    density_entry = entry_for("volumetric_bone_mineral_density", TermSystem.LOINC)
    bindings.append(
        ConceptBinding(
            element="volumetric_bone_mineral_density",
            concept_key=concept_key(density_entry.system, density_entry.key),
            measurement=f"density_indicator:{density:.4f}",
        )
    )
    method_entry = entry_for("measurement_method", TermSystem.LOINC)
    bindings.append(
        ConceptBinding(
            element="measurement_method",
            concept_key=concept_key(method_entry.system, method_entry.key),
            measurement="acquisition_descriptor",
        )
    )
    tracking_entry = entry_for("evidence_link", TermSystem.DICOM_SR)
    bindings.append(
        ConceptBinding(
            element="evidence_link",
            concept_key=concept_key(tracking_entry.system, tracking_entry.key),
            measurement="evidence_chain",
        )
    )
    ordered: list[str] = []
    for binding in bindings:
        if binding.concept_key not in ordered:
            ordered.append(binding.concept_key)
    return Grounding(concepts=tuple(ordered), bindings=tuple(bindings))


def ground_decision(
    base: Grounding,
    band: DiagnosticBand,
    tier: RiskTier,
    severity: DeformityGrade,
    reference_available: bool,
    prior_fragility_fracture: bool,
    treatment_exposure: bool,
    index: ConceptIndex,
) -> Grounding:
    extra: list[ConceptBinding] = [
        ConceptBinding(
            element="diagnostic_band",
            concept_key=concept_key(TermSystem.SNOMED_CT, BAND_CONCEPT_CODE[band]),
            measurement="band_interval",
        ),
        ConceptBinding(
            element="risk_tier",
            concept_key=concept_key(TermSystem.SNOMED_CT, HISTORY_CONCEPT_CODE[prior_fragility_fracture]),
            measurement="prior_fracture_state",
        ),
        ConceptBinding(
            element="measurement_method",
            concept_key=concept_key(TermSystem.LOINC, entry_for("measurement_method", TermSystem.LOINC).key),
            measurement="acquisition_descriptor",
        ),
    ]
    if reference_available:
        areal = entry_for("areal_bone_mineral_density", TermSystem.LOINC)
        extra.append(
            ConceptBinding(
                element="areal_bone_mineral_density",
                concept_key=concept_key(areal.system, areal.key),
                measurement="reference_density",
            )
        )
    if treatment_exposure:
        extra.append(
            ConceptBinding(
                element="risk_tier",
                concept_key=concept_key(TermSystem.SNOMED_CT, "OSTEOCT-1007"),
                measurement="treatment_exposure_state",
            )
        )
    if severity is not DeformityGrade.NONE:
        grade_entry = entry_for("deformity_grade", TermSystem.SNOMED_CT)
        extra.append(
            ConceptBinding(
                element="deformity_grade",
                concept_key=concept_key(grade_entry.system, grade_entry.key),
                measurement=f"grade:{int(severity.value)}",
            )
        )
    conformance_entry = _by_code(index, "OSTEOCT-4007")
    if conformance_entry is not None:
        extra.append(
            ConceptBinding(
                element="evidence_link",
                concept_key=concept_key(conformance_entry.system, conformance_entry.code),
                measurement="conformance_assertion",
            )
        )
    return base.extended(tuple(extra))


def ground_to_standard(
    regions: tuple[VertebraRegion, ...],
    band: DiagnosticBand,
    tier: RiskTier,
    severity: DeformityGrade,
    standard_attenuation_hu: float,
    uncertainty_hu: float,
    density: float,
    index: ConceptIndex,
    prior_fragility_fracture: bool = False,
    treatment_exposure: bool = False,
    reference_available: bool = False,
) -> Grounding:
    base = ground_measurements(regions, standard_attenuation_hu, uncertainty_hu, density)
    return ground_decision(
        base,
        band,
        tier,
        severity,
        reference_available,
        prior_fragility_fracture,
        treatment_exposure,
        index,
    )


def _by_code(index: ConceptIndex, code: str) -> Concept | None:
    for concept in index.concepts:
        if concept.code == code:
            return concept
    return None


def resolved_concepts(grounding: Grounding, index: ConceptIndex) -> tuple[str, ...]:
    table = index.alias_table()
    return tuple(binding.concept_key for binding in grounding.bindings if binding.concept_key in table)
