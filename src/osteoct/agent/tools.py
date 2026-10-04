"""The ten tools the agent exposes, with their declared inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from osteoct.acquisition.attenuation import AttenuationParams, NoAssessableVoxels, examination_attenuation
from osteoct.acquisition.density import DensityModel, density_indicator
from osteoct.acquisition.harmonise import HarmonisationModel, HarmonisedAttenuation, harmonise
from osteoct.decision.operator import OperatorOutcome, constrain, unconstrained
from osteoct.decision.repair import RepairTrace
from osteoct.grounding.standard import Grounding, ground_decision, ground_measurements
from osteoct.localisation.morphology import MorphometryParams
from osteoct.localisation.vertebra import LocalisationParams, localise
from osteoct.ontology.axioms import AxiomSet
from osteoct.schema.concepts import ConceptIndex
from osteoct.schema.records import (
    AcquisitionDescriptor,
    CandidateDecision,
    DecisionRecord,
    MeasurementBundle,
    MorphologyFeatures,
    VertebraMeasurement,
    VertebraRegion,
)
from osteoct.schema.vocabulary import DeformityGrade, DiagnosticBand, DiagnosticContext, RiskTier
from osteoct.serialisation.dcm_sr import serialise_record
from osteoct.texture.descriptors import TextureParams, fit_stability, retained_texture
from osteoct.texture.stability import StabilityVerdict


class ToolName(str, Enum):
    LOCALISE = "LOCALISE"
    ATTENUATION = "ATTENUATION"
    HARMONISE = "HARMONISE"
    TEXTURE = "TEXTURE"
    STABILITYFILTER = "STABILITYFILTER"
    DENSITYESTIMATE = "DENSITYESTIMATE"
    GROUNDTOSTANDARD = "GROUNDTOSTANDARD"
    SCORE = "SCORE"
    CONSTRAIN = "CONSTRAIN"
    SERIALISE = "SERIALISE"


TOOL_ORDER: tuple[ToolName, ...] = tuple(ToolName)

TOOL_INPUTS: dict[ToolName, tuple[str, ...]] = {
    ToolName.LOCALISE: ("volume",),
    ToolName.ATTENUATION: ("volume", "regions"),
    ToolName.HARMONISE: ("mean", "sd", "voxels", "acquisition"),
    ToolName.TEXTURE: ("volume", "regions"),
    ToolName.STABILITYFILTER: ("texture_vector",),
    ToolName.DENSITYESTIMATE: ("standard_attenuation", "uncertainty", "texture", "acquisition"),
    ToolName.GROUNDTOSTANDARD: ("regions", "standard_attenuation", "uncertainty", "density"),
    ToolName.SCORE: ("density_indicator", "standard_attenuation", "uncertainty", "texture", "concepts", "context"),
    ToolName.CONSTRAIN: ("candidate", "bundle", "axioms", "budget"),
    ToolName.SERIALISE: ("decision", "bundle"),
}


class PipelineError(Exception):
    pass


@dataclass(slots=True)
class ToolRegistry:
    calls: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def record(self, tool: ToolName) -> None:
        self.calls[tool.value] = self.calls.get(tool.value, 0) + 1

    def record_error(self, tool: ToolName, message: str) -> None:
        key = f"{tool.value}:{message}"
        self.errors[key] = self.errors.get(key, 0) + 1

    def inventory(self) -> tuple[str, ...]:
        return tuple(tool.value for tool in TOOL_ORDER)

    def total(self) -> int:
        return sum(self.calls.values())

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.calls.items()))

    def error_paths(self) -> dict[str, int]:
        return dict(sorted(self.errors.items()))


@dataclass(frozen=True, slots=True)
class Toolbox:
    localisation: LocalisationParams
    attenuation: AttenuationParams
    morphometry: MorphometryParams
    texture: TextureParams
    harmonisation: HarmonisationModel
    density: DensityModel
    index: ConceptIndex
    stability: StabilityVerdict | None = None

    def localise(
        self,
        volume: npt.NDArray[np.float32],
        registry: ToolRegistry,
    ) -> tuple[tuple[VertebraRegion, ...], tuple[MorphologyFeatures, ...]]:
        registry.record(ToolName.LOCALISE)
        regions, morphologies = localise(volume, self.localisation, self.attenuation, self.morphometry)
        if not regions:
            registry.record_error(ToolName.LOCALISE, "no_assessable_vertebra")
        return (regions, morphologies)

    def measure_attenuation(
        self,
        volume: npt.NDArray[np.float32],
        regions: tuple[VertebraRegion, ...],
        morphologies: tuple[MorphologyFeatures, ...],
        registry: ToolRegistry,
    ) -> tuple[tuple[VertebraMeasurement, ...], float, float, int]:
        registry.record(ToolName.ATTENUATION)
        return examination_attenuation(volume, regions, morphologies, self.attenuation)

    def harmonise(
        self,
        mean_hu: float,
        sd_hu: float,
        voxels: int,
        acquisition: AcquisitionDescriptor,
        registry: ToolRegistry,
    ) -> HarmonisedAttenuation:
        registry.record(ToolName.HARMONISE)
        return harmonise(mean_hu, sd_hu, voxels, acquisition, self.harmonisation)

    def extract_texture(
        self,
        volume: npt.NDArray[np.float32],
        regions: tuple[VertebraRegion, ...],
        registry: ToolRegistry,
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        registry.record(ToolName.TEXTURE)
        blocks = [_region_block(volume, region) for region in regions]
        if not blocks:
            registry.record_error(ToolName.TEXTURE, "no_region_block")
            return ((), ())
        usable = [block for block in blocks if min(block.shape) >= 4]
        if not usable:
            registry.record_error(ToolName.TEXTURE, "region_below_descriptor_window")
            return ((), ())
        if self.stability is None:
            from osteoct.texture.descriptors import descriptor_names, descriptor_vector

            vector = descriptor_vector(usable[0], self.texture)
            return (tuple(float(value) for value in vector), descriptor_names())
        return retained_texture(usable[0], self.stability, self.texture)

    def stability_filter(
        self,
        volume: npt.NDArray[np.float32],
        regions: tuple[VertebraRegion, ...],
        registry: ToolRegistry,
    ) -> StabilityVerdict:
        registry.record(ToolName.STABILITYFILTER)
        blocks = [_region_block(volume, region) for region in regions]
        usable = tuple(block for block in blocks if min(block.shape) >= 4)
        if not usable:
            registry.record_error(ToolName.STABILITYFILTER, "no_usable_block")
            return StabilityVerdict((), (), ())
        return fit_stability(usable, usable, self.texture)

    def stability_applied(
        self,
        vector: tuple[float, ...],
        names: tuple[str, ...],
        registry: ToolRegistry,
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        registry.record(ToolName.STABILITYFILTER)
        if self.stability is None:
            return (vector, names)
        values = dict(zip(names, vector, strict=False))
        retained = tuple(name for name in names if name in self.stability.retained)
        if not retained:
            registry.record_error(ToolName.STABILITYFILTER, "empty_retained_set")
            return (vector, names)
        return (tuple(values[name] for name in retained), retained)

    def density_estimate(
        self,
        standard_attenuation_hu: float,
        uncertainty_hu: float,
        texture_vector: tuple[float, ...],
        texture_names: tuple[str, ...],
        acquisition: AcquisitionDescriptor,
        registry: ToolRegistry,
    ) -> float:
        registry.record(ToolName.DENSITYESTIMATE)
        return density_indicator(
            standard_attenuation_hu,
            uncertainty_hu,
            texture_vector,
            texture_names,
            acquisition,
            self.density,
        )

    def ground_measurements(
        self,
        regions: tuple[VertebraRegion, ...],
        standard_attenuation_hu: float,
        uncertainty_hu: float,
        density: float,
        registry: ToolRegistry,
    ) -> Grounding:
        registry.record(ToolName.GROUNDTOSTANDARD)
        return ground_measurements(regions, standard_attenuation_hu, uncertainty_hu, density)

    def ground_decision(
        self,
        base: Grounding,
        band: DiagnosticBand,
        tier: RiskTier,
        severity: DeformityGrade,
        context: DiagnosticContext,
        registry: ToolRegistry,
    ) -> Grounding:
        return ground_decision(
            base,
            band,
            tier,
            severity,
            context.reference_density_available(),
            context.prior_fragility_fracture,
            context.treatment_exposure,
            self.index,
        )

    def operate(
        self,
        candidate: CandidateDecision,
        bundle: MeasurementBundle,
        axiom_set: AxiomSet,
        budget: int,
        registry: ToolRegistry,
        enforced: bool = True,
    ) -> OperatorOutcome:
        registry.record(ToolName.CONSTRAIN)
        if enforced:
            outcome = constrain(candidate, bundle, axiom_set, budget)
            if not outcome.emitted():
                registry.record_error(ToolName.CONSTRAIN, outcome.reason())
            return outcome
        admitted = unconstrained(candidate, bundle, axiom_set)
        return OperatorOutcome(admitted=admitted, abstention=None, trace=_null_trace(candidate))

    def serialise(self, record: DecisionRecord, bundle: MeasurementBundle, registry: ToolRegistry) -> str:
        registry.record(ToolName.SERIALISE)
        return serialise_record(record, bundle)


def _region_block(volume: npt.NDArray[np.float32], region: VertebraRegion) -> npt.NDArray[np.float64]:
    centre = region.centre_index
    extent = region.extent_index
    selector = tuple(  # type: ignore[assignment]
        slice(max(0, centre[axis] - extent[axis] // 2), max(0, centre[axis] - extent[axis] // 2) + extent[axis])
        for axis in range(3)
    )
    return np.asarray(volume[selector], dtype=np.float64)


def _null_trace(candidate: CandidateDecision) -> RepairTrace:
    return RepairTrace(candidate=candidate, steps=(), terminal_violation=None, resolved_identifiers=False, budget_exhausted=False)


def tool_report(registry: ToolRegistry) -> dict[str, object]:
    return {
        "inventory": list(registry.inventory()),
        "calls": registry.summary(),
        "total_calls": registry.total(),
        "error_paths": registry.error_paths(),
    }


def has_assessable_vertebra(regions: tuple[VertebraRegion, ...]) -> bool:
    return bool(regions)


def is_no_vertebra_error(exc: BaseException) -> bool:
    return isinstance(exc, NoAssessableVoxels)
