"""Evidence-token bookkeeping for the trail-extension family."""

from __future__ import annotations

from osteoct.schema.records import MeasurementBundle
from osteoct.schema.vocabulary import DiagnosticContext

BASE_TOKENS: tuple[str, ...] = (
    "localisation",
    "m_att",
    "u_att",
    "m_att_star",
    "phi",
    "density_indicator",
    "acquisition_descriptor",
    "morphology",
)

HISTORY_TOKENS: tuple[str, ...] = (
    "band",
    "prior_fracture_state",
    "treatment_exposure_state",
    "risk_semantics",
)

CHAIN_TERMINALS: tuple[str, ...] = ("evidence_chain", "conformance_assertion")


def measurement_tokens(bundle: MeasurementBundle, context: DiagnosticContext) -> tuple[str, ...]:
    tokens = list(BASE_TOKENS)
    for measurement in bundle.vertebrae:
        tokens.append(f"level:{measurement.region.level.value}")
    if context.reference_density_available():
        tokens.append("reference_density")
    return tuple(dict.fromkeys(tokens))


def history_tokens(context: DiagnosticContext) -> tuple[str, ...]:
    prior_fracture, treatment_exposure = context.as_flags()
    tokens = list(HISTORY_TOKENS)
    if prior_fracture:
        tokens.append("prior_fracture_state")
    if treatment_exposure:
        tokens.append("treatment_exposure_state")
    return tuple(dict.fromkeys(tokens))


def attach(tokens: tuple[str, ...], additional: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tokens + additional))


def close_chain(tokens: tuple[str, ...], bundle: MeasurementBundle) -> tuple[str, ...]:
    closed = list(tokens)
    for token in CHAIN_TERMINALS:
        if token not in closed:
            closed.append(token)
    for measurement in bundle.vertebrae:
        token = f"traced:{measurement.region.level.value}"
        if token not in closed:
            closed.append(token)
    return tuple(closed)


def chain_complete(tokens: tuple[str, ...]) -> bool:
    return all(terminal in tokens for terminal in CHAIN_TERMINALS)


def chain_depth(tokens: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if token in BASE_TOKENS or token.startswith("level:") or token.startswith("traced:"))
