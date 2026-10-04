"""The released concept index over the four source systems named in the artefact."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.concepts import Concept, ConceptIndex, TermSystem

RECORD_ELEMENTS: tuple[str, ...] = (
    "trabecular_attenuation",
    "attenuation_uncertainty",
    "areal_bone_mineral_density",
    "volumetric_bone_mineral_density",
    "vertebral_level",
    "deformity_grade",
    "diagnostic_band",
    "risk_tier",
    "measurement_method",
    "evidence_link",
)


@dataclass(frozen=True, slots=True)
class IndexEntry:
    key: str
    system: TermSystem
    term: str
    element: str
    unit: str | None = None


INDEX_ENTRIES: tuple[IndexEntry, ...] = (
    IndexEntry("OSTEOCT-1001", TermSystem.SNOMED_CT, "Osteoporosis", "diagnostic_band"),
    IndexEntry("OSTEOCT-1002", TermSystem.SNOMED_CT, "Osteopenia", "diagnostic_band"),
    IndexEntry("OSTEOCT-1003", TermSystem.SNOMED_CT, "Normal bone density", "diagnostic_band"),
    IndexEntry("OSTEOCT-1004", TermSystem.SNOMED_CT, "Severe osteoporosis", "diagnostic_band"),
    IndexEntry("OSTEOCT-1005", TermSystem.SNOMED_CT, "Prior fragility fracture", "risk_tier"),
    IndexEntry("OSTEOCT-1006", TermSystem.SNOMED_CT, "Vertebral compression fracture", "deformity_grade"),
    IndexEntry("OSTEOCT-1007", TermSystem.SNOMED_CT, "Antiresorptive therapy", "risk_tier"),
    IndexEntry("OSTEOCT-2001", TermSystem.RADLEX, "Vertebral body", "vertebral_level"),
    IndexEntry("OSTEOCT-2002", TermSystem.RADLEX, "Trabecular bone", "trabecular_attenuation"),
    IndexEntry("OSTEOCT-2003", TermSystem.RADLEX, "Sagittal reformation", "measurement_method"),
    IndexEntry("OSTEOCT-2004", TermSystem.RADLEX, "Anterior vertebral height", "deformity_grade", "mm"),
    IndexEntry("OSTEOCT-2005", TermSystem.RADLEX, "Middle vertebral height", "deformity_grade", "mm"),
    IndexEntry("OSTEOCT-2006", TermSystem.RADLEX, "Posterior vertebral height", "deformity_grade", "mm"),
    IndexEntry("OSTEOCT-2007", TermSystem.RADLEX, "Wedge deformity", "deformity_grade"),
    IndexEntry("OSTEOCT-2008", TermSystem.RADLEX, "Biconcave deformity", "deformity_grade"),
    IndexEntry("OSTEOCT-3001", TermSystem.LOINC, "Bone mineral density", "areal_bone_mineral_density", "g/cm2"),
    IndexEntry(
        "OSTEOCT-3002", TermSystem.LOINC, "Volumetric bone mineral density estimate", "volumetric_bone_mineral_density", "mg/cm3"
    ),
    IndexEntry("OSTEOCT-3003", TermSystem.LOINC, "Attenuation in Hounsfield units", "trabecular_attenuation", "HU"),
    IndexEntry("OSTEOCT-3004", TermSystem.LOINC, "Attenuation uncertainty half-width", "attenuation_uncertainty", "HU"),
    IndexEntry("OSTEOCT-3005", TermSystem.LOINC, "T-score", "diagnostic_band"),
    IndexEntry("OSTEOCT-3006", TermSystem.LOINC, "Ten-year major osteoporotic fracture probability", "risk_tier", "%"),
    IndexEntry("OSTEOCT-3007", TermSystem.LOINC, "Tube voltage", "measurement_method", "kV"),
    IndexEntry("OSTEOCT-3008", TermSystem.LOINC, "Slice thickness", "measurement_method", "mm"),
    IndexEntry("OSTEOCT-4001", TermSystem.DICOM_SR, "Measurement group", "evidence_link"),
    IndexEntry("OSTEOCT-4002", TermSystem.DICOM_SR, "Numeric measurement", "trabecular_attenuation"),
    IndexEntry("OSTEOCT-4003", TermSystem.DICOM_SR, "Code sequence", "evidence_link"),
    IndexEntry("OSTEOCT-4004", TermSystem.DICOM_SR, "Tracking identifier", "evidence_link"),
    IndexEntry("OSTEOCT-4005", TermSystem.DICOM_SR, "Vertebral level index", "vertebral_level"),
    IndexEntry("OSTEOCT-4006", TermSystem.DICOM_SR, "Observer context", "measurement_method"),
    IndexEntry("OSTEOCT-4007", TermSystem.DICOM_SR, "Conformance assertion", "evidence_link"),
)

INDEX_BY_SYSTEM: dict[TermSystem, tuple[IndexEntry, ...]] = {
    system: tuple(entry for entry in INDEX_ENTRIES if entry.system is system) for system in TermSystem
}

ELEMENT_MAP: dict[str, tuple[IndexEntry, ...]] = {
    element: tuple(entry for entry in INDEX_ENTRIES if entry.element == element) for element in RECORD_ELEMENTS
}

CODE_TO_ELEMENT: dict[str, str] = {entry.key: entry.element for entry in INDEX_ENTRIES}

TERM_SYSTEM_BY_CODE: dict[str, TermSystem] = {entry.key: entry.system for entry in INDEX_ENTRIES}


def build_concept_index(entries: tuple[IndexEntry, ...] = INDEX_ENTRIES) -> ConceptIndex:
    return ConceptIndex.of(Concept(system=entry.system, code=entry.key, display=entry.term, unit=entry.unit) for entry in entries)


def entry_for(element: str, system: TermSystem | None = None) -> IndexEntry:
    candidates = ELEMENT_MAP[element]
    if system is None:
        return candidates[0]
    for entry in candidates:
        if entry.system is system:
            return entry
    raise KeyError(f"no index entry for element {element!r} in system {system.value!r}")
