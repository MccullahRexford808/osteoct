"""OWL 2 EL profile constraint used by every axiom in the artefact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExpressionKind(str, Enum):
    ATOM = "atom"
    CONJUNCTION = "conjunction"
    EXISTENTIAL = "existential"
    ROLE_HIERARCHY = "role_hierarchy"
    NEGATION = "negation"
    UNIVERSAL = "universal"
    CARDINALITY = "cardinality"
    DISJUNCTION = "disjunction"


EL_KINDS: frozenset[ExpressionKind] = frozenset(
    {ExpressionKind.ATOM, ExpressionKind.CONJUNCTION, ExpressionKind.EXISTENTIAL, ExpressionKind.ROLE_HIERARCHY}
)

NON_EL_KINDS: frozenset[ExpressionKind] = frozenset(
    {ExpressionKind.NEGATION, ExpressionKind.UNIVERSAL, ExpressionKind.CARDINALITY, ExpressionKind.DISJUNCTION}
)


class ProfileViolation(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Expression:
    kind: ExpressionKind
    name: str = ""
    operands: tuple[Expression, ...] = ()
    role: str = ""
    filler: Expression | None = None

    def children(self) -> tuple[Expression, ...]:
        nested = list(self.operands)
        if self.filler is not None:
            nested.append(self.filler)
        return tuple(nested)

    def walk(self) -> tuple[Expression, ...]:
        collected: list[Expression] = [self]
        for child in self.children():
            collected.extend(child.walk())
        return tuple(collected)


def atom(name: str) -> Expression:
    return Expression(ExpressionKind.ATOM, name=name)


def conjunction(*operands: Expression) -> Expression:
    return Expression(ExpressionKind.CONJUNCTION, operands=operands)


def existential(role: str, filler: Expression) -> Expression:
    return Expression(ExpressionKind.EXISTENTIAL, role=role, filler=filler)


def role_hierarchy(super_role: str, sub_role: str) -> Expression:
    return Expression(ExpressionKind.ROLE_HIERARCHY, role=super_role, name=sub_role)


def negation(operand: Expression) -> Expression:
    return Expression(ExpressionKind.NEGATION, operands=(operand,))


def universal(role: str, filler: Expression) -> Expression:
    return Expression(ExpressionKind.UNIVERSAL, role=role, filler=filler)


def cardinality(role: str, bound: int) -> Expression:
    return Expression(ExpressionKind.CARDINALITY, role=role, name=str(bound))


def disjunction(*operands: Expression) -> Expression:
    return Expression(ExpressionKind.DISJUNCTION, operands=operands)


def profile_errors(expression: Expression) -> tuple[str, ...]:
    return tuple(
        f"{node.kind.value} at {node.role or node.name or 'root'}" for node in expression.walk() if node.kind in NON_EL_KINDS
    )


def is_el(expression: Expression) -> bool:
    return not profile_errors(expression)


def require_el(expression: Expression) -> None:
    errors = profile_errors(expression)
    if errors:
        joined = "; ".join(errors)
        raise ProfileViolation(f"expression leaves the OWL 2 EL profile: {joined}")


def el_subsumes(
    expression: Expression,
    target: str,
    definitions: dict[str, Expression],
    role_hierarchy_pairs: tuple[tuple[str, str], ...] = (),
) -> bool:
    for node in expression.walk():
        if node.kind is ExpressionKind.ATOM and node.name == target:
            return True
    supers = _role_closure(role_hierarchy_pairs)
    for node in expression.walk():
        if node.kind is not ExpressionKind.EXISTENTIAL or node.filler is None:
            continue
        for filler_node in node.filler.walk():
            if filler_node.kind is not ExpressionKind.ATOM:
                continue
            definition = definitions.get(filler_node.name)
            if definition is None:
                continue
            if el_subsumes(definition, target, definitions, role_hierarchy_pairs):
                return True
        if target in supers.get(node.role, frozenset()):
            return True
    return False


def _role_closure(pairs: tuple[tuple[str, str], ...]) -> dict[str, frozenset[str]]:
    closure: dict[str, set[str]] = {}
    for super_role, sub_role in pairs:
        closure.setdefault(sub_role, set()).add(super_role)
    changed = True
    while changed:
        changed = False
        for supers in list(closure.values()):
            for super_role in tuple(supers):
                for inherited in closure.get(super_role, set()):
                    if inherited not in supers:
                        supers.add(inherited)
                        changed = True
    return {role: frozenset(supers) for role, supers in closure.items()}


def el_consistent(definitions: dict[str, Expression]) -> bool:
    for name, definition in definitions.items():
        if not is_el(definition):
            return False
        if el_subsumes(definition, name, definitions):
            continue
    pairs = tuple(
        (node.role, node.name)
        for definition in definitions.values()
        for node in definition.walk()
        if node.kind is ExpressionKind.ROLE_HIERARCHY
    )
    for name, definition in definitions.items():
        if el_subsumes(definition, name, definitions, pairs):
            continue
        return False
    return True
