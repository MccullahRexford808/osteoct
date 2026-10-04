"""DICOM Structured Reporting serialisation of an admitted decision record."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from osteoct.schema.records import ConceptBinding, DecisionRecord, MeasurementBundle
from osteoct.schema.vocabulary import DeformityGrade

CONTAINS = "CONTAINS"
HAS_CONCEPT_MODIFIER = "HAS CONCEPT MOD"
INFERRED_FROM = "INFERRED FROM"

ROOT_CONCEPT = ("DICOM SR", "OSTEOCT-4001", "Measurement group")
CONFORMANCE_CONCEPT = ("DICOM SR", "OSTEOCT-4007", "Conformance assertion")


@dataclass(frozen=True, slots=True)
class SrItem:
    relationship: str
    value_type: str
    concept: tuple[str, str, str]
    value: str
    children: tuple[SrItem, ...] = ()

    def is_leaf(self) -> bool:
        return not self.children


def numeric_item(relationship: str, concept: tuple[str, str, str], value: float, unit: str) -> SrItem:
    return SrItem(relationship, "NUM", concept, f"{value:.4f} {unit}")


def code_item(relationship: str, concept: tuple[str, str, str], value: str) -> SrItem:
    return SrItem(relationship, "CODE", concept, value)


def text_item(relationship: str, concept: tuple[str, str, str], value: str) -> SrItem:
    return SrItem(relationship, "TEXT", concept, value)


def container_item(concept: tuple[str, str, str], children: tuple[SrItem, ...]) -> SrItem:
    return SrItem(CONTAINS, "CONTAINER", concept, "", children)


def build_sr_tree(record: DecisionRecord, bundle: MeasurementBundle) -> SrItem:
    children: list[SrItem] = [
        text_item(HAS_CONCEPT_MODIFIER, ("DICOM SR", "OSTEOCT-4004", "Tracking identifier"), record.record_id),
        code_item(CONTAINS, ("DICOM SR", "OSTEOCT-4003", "Code sequence"), record.band.value),
        code_item(CONTAINS, ("DICOM SR", "OSTEOCT-4006", "Observer context"), record.tier.value),
        numeric_item(CONTAINS, ("LOINC", "OSTEOCT-3003", "Attenuation in Hounsfield units"), bundle.standard_attenuation_hu, "HU"),
        numeric_item(CONTAINS, ("LOINC", "OSTEOCT-3004", "Attenuation uncertainty half-width"), bundle.uncertainty_hu, "HU"),
        numeric_item(
            CONTAINS, ("LOINC", "OSTEOCT-3002", "Volumetric bone mineral density estimate"), bundle.density_indicator, "mg/cm3"
        ),
        text_item(CONTAINS, ("DICOM SR", "OSTEOCT-4006", "Observer context"), bundle.acquisition.signature()),
        numeric_item(CONTAINS, ("DICOM SR", "OSTEOCT-4007", "Conformance assertion"), record.conformance, "ratio"),
    ]
    if record.severity is not DeformityGrade.NONE:
        children.append(
            text_item(
                CONTAINS, ("SNOMED CT", "OSTEOCT-1006", "Vertebral compression fracture"), f"grade {int(record.severity.value)}"
            )
        )
    for measurement in bundle.vertebrae:
        children.append(
            text_item(
                CONTAINS,
                ("DICOM SR", "OSTEOCT-4005", "Vertebral level index"),
                f"{measurement.region.level.value} {measurement.mean_attenuation_hu:.2f} HU",
            )
        )
    for binding in record.bindings:
        children.append(_binding_item(binding))
    return container_item(ROOT_CONCEPT, tuple(children))


def _binding_item(binding: ConceptBinding) -> SrItem:
    system, _, code = binding.concept_key.partition(":")
    return text_item(
        INFERRED_FROM,
        ("DICOM SR", "OSTEOCT-4004", "Tracking identifier"),
        f"{system}|{code}|{binding.element}|{binding.measurement}",
    )


def serialise_sr(item: SrItem, depth: int = 0) -> str:
    system, code, display = item.concept
    pad = "  " * depth
    header = f"{pad}{item.relationship} {item.value_type} ({system}, {code}, {display})"
    if item.value:
        header = f"{header} = {item.value}"
    lines = [header]
    for child in item.children:
        lines.append(serialise_sr(child, depth + 1))
    return "\n".join(lines)


def serialise_record(record: DecisionRecord, bundle: MeasurementBundle) -> str:
    return serialise_sr(build_sr_tree(record, bundle))


def sr_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def leaf_count(item: SrItem) -> int:
    if item.is_leaf():
        return 1
    return sum(leaf_count(child) for child in item.children)


def concept_sequence(item: SrItem) -> tuple[str, ...]:
    ordered: list[str] = []
    system, code, _ = item.concept
    ordered.append(f"{system}:{code}")
    for child in item.children:
        for key in concept_sequence(child):
            if key not in ordered:
                ordered.append(key)
    return tuple(ordered)


def emit_part10(item: SrItem, out_path: Path) -> bool:
    try:
        import pydicom  # type: ignore[import-not-found]
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        return False
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = Dataset()
    dataset.file_meta = meta
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.ContentDate = "20000101"
    dataset.ContentTime = "000000"
    dataset.ValueType = item.value_type
    dataset.TextValue = serialise_sr(item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(out_path), dataset, write_like_original=False)
    return True
