"""Architecture arms and the planner substitutions reported in the ablation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from osteoct.models.blocks import ENCODER_TABLE, CompactVolumeEncoder3D, ConvEncoder3D, ShiftedWindowEncoder3D
from osteoct.models.scorer import OsteoScorer, ScorerSpec


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    encoder: str | None
    use_descriptors: bool
    use_texture: bool
    use_concepts: bool
    use_embedding: bool
    planner: str

    def labelled(self) -> str:
        return self.name.replace("_", " ")


ARM_SPECS: dict[str, ArmSpec] = {
    "constrained": ArmSpec("constrained", "shifted_window", True, True, True, True, "learned"),
    "shifted_window": ArmSpec("shifted_window", "shifted_window", True, True, True, True, "learned"),
    "convolutional": ArmSpec("convolutional", "3d_conv", True, True, True, True, "learned"),
    "foundation_encoder": ArmSpec("foundation_encoder", "foundation_contract", True, True, True, True, "learned"),
    "tool_free_scorer": ArmSpec("tool_free_scorer", None, True, True, True, False, "none"),
    "text_only_planner": ArmSpec("text_only_planner", None, False, False, True, False, "text_only"),
}

OBSERVED_ARMS: tuple[str, ...] = (
    "convolutional",
    "shifted_window",
    "foundation_encoder",
    "tool_free_scorer",
)

SUBSTITUTED_ARMS: tuple[str, ...] = ("text_only_planner", "tool_free_scorer")


@dataclass(frozen=True, slots=True)
class ArmBundle:
    name: str
    scorer: OsteoScorer
    encoder: nn.Module | None
    embedding_dim: int
    spec: ArmSpec


def encoder_for(name: str | None, embed_dim: int) -> tuple[nn.Module | None, int]:
    if name is None:
        return (None, 0)
    if name == "3d_conv":
        return (ConvEncoder3D(in_channels=1, embed_dim=embed_dim, depth=3), embed_dim)
    if name == "shifted_window":
        return (ShiftedWindowEncoder3D(in_channels=1, dim=embed_dim, depth=2, window=4, heads=2), embed_dim)
    if name == "foundation_contract":
        return (CompactVolumeEncoder3D(in_channels=1, dim=embed_dim, depth=2, heads=2), embed_dim)
    raise KeyError(f"unknown encoder {name!r}")


def build_arm(
    name: str,
    descriptor_dim: int = 3,
    texture_dim: int = 25,
    concept_dim: int = 30,
    embed_dim: int = 16,
    hidden: int = 64,
    dropout: float = 0.1,
) -> ArmBundle:
    if name not in ARM_SPECS:
        raise KeyError(f"unknown arm {name!r}")
    spec = ARM_SPECS[name]
    encoder, embedding_dim = encoder_for(spec.encoder, embed_dim)
    scorer_spec = ScorerSpec(
        descriptor_dim=descriptor_dim,
        texture_dim=texture_dim,
        concept_dim=concept_dim,
        embedding_dim=embedding_dim,
        hidden=hidden,
        dropout=dropout,
        use_descriptors=spec.use_descriptors,
        use_texture=spec.use_texture,
        use_concepts=spec.use_concepts,
        use_embedding=spec.use_embedding,
    )
    return ArmBundle(name=name, scorer=OsteoScorer(scorer_spec), encoder=encoder, embedding_dim=embedding_dim, spec=spec)


def arm_names() -> tuple[str, ...]:
    return tuple(ARM_SPECS)


def encoder_names() -> tuple[str, ...]:
    return tuple(ENCODER_TABLE)


def is_constrained_arm(name: str) -> bool:
    return ARM_SPECS[name].planner == "learned" and name != "tool_free_scorer"


def reference_arm() -> str:
    return "constrained"
