"""Scanner families and the acquisition descriptor k attached to every measurement."""

from __future__ import annotations

from dataclasses import dataclass

from osteoct.schema.records import AcquisitionDescriptor


@dataclass(frozen=True, slots=True)
class ScannerFamily:
    name: str
    nominal_voltage_kvp: float
    attenuation_scale: float
    attenuation_offset_hu: float
    noise_sigma_hu: float
    kernels: tuple[str, ...]


SCANNER_FAMILIES: tuple[ScannerFamily, ...] = (
    ScannerFamily("vendor_a", 120.0, 1.0, 0.0, 14.0, ("B30f", "B40f", "B50f")),
    ScannerFamily("vendor_b", 120.0, 1.035, 4.5, 15.5, ("STANDARD", "BONE", "DETAIL")),
    ScannerFamily("vendor_c", 100.0, 0.958, -6.0, 17.0, ("FC30", "FC50", "I30f")),
    ScannerFamily("vendor_d", 140.0, 1.072, 9.0, 13.0, ("Br40d", "Br60d", "Qr40d")),
)

FAMILY_BY_NAME: dict[str, ScannerFamily] = {family.name: family for family in SCANNER_FAMILIES}

REFERENCE_VOLTAGE_KVP = 120.0


def scanner_family(name: str) -> ScannerFamily:
    if name not in FAMILY_BY_NAME:
        raise KeyError(f"unknown scanner family {name!r}")
    return FAMILY_BY_NAME[name]


def acquisition_descriptor(
    tube_voltage_kvp: float,
    scanner_family_name: str,
    reconstruction_kernel: str,
    slice_thickness_mm: float,
) -> AcquisitionDescriptor:
    family = scanner_family(scanner_family_name)
    if not family.kernels:
        raise ValueError(f"scanner family {scanner_family_name!r} declares no reconstruction kernel")
    return AcquisitionDescriptor(
        tube_voltage_kvp=float(tube_voltage_kvp),
        scanner_family=family.name,
        reconstruction_kernel=reconstruction_kernel,
        slice_thickness_mm=float(slice_thickness_mm),
    )


def kernel_offset_hu(family: ScannerFamily, kernel: str) -> float:
    if kernel not in family.kernels:
        return 0.0
    return 1.5 * float(family.kernels.index(kernel))


def voltage_delta_kvp(acquisition: AcquisitionDescriptor) -> float:
    return acquisition.tube_voltage_kvp - REFERENCE_VOLTAGE_KVP


def nominal_noise_hu(acquisition: AcquisitionDescriptor) -> float:
    return scanner_family(acquisition.scanner_family).noise_sigma_hu


def family_of(acquisition: AcquisitionDescriptor) -> ScannerFamily:
    return scanner_family(acquisition.scanner_family)
