"""Published-identifier bindings used by the concept layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class TermSystem(str, Enum):
    SNOMED_CT = "SNOMED CT"
    RADLEX = "RadLex"
    LOINC = "LOINC"
    DICOM_SR = "DICOM SR"


@dataclass(frozen=True, slots=True)
class Concept:
    system: TermSystem
    code: str
    display: str
    unit: str | None = None

    def key(self) -> str:
        return f"{self.system.value}:{self.code}"


@dataclass(frozen=True, slots=True)
class ConceptIndex:
    concepts: tuple[Concept, ...]

    @classmethod
    def of(cls, concepts: Iterable[Concept]) -> ConceptIndex:
        unique: dict[str, Concept] = {}
        for concept in concepts:
            unique.setdefault(concept.key(), concept)
        return cls(tuple(unique.values()))

    def keys(self) -> frozenset[str]:
        return frozenset(concept.key() for concept in self.concepts)

    def alias_table(self) -> dict[str, Concept]:
        table: dict[str, Concept] = {}
        for concept in self.concepts:
            table[concept.key()] = concept
            table[concept.code] = concept
            table[concept.display.lower()] = concept
        return table

    def contains(self, token: str) -> bool:
        return token in self.alias_table()

    def resolve(self, token: str) -> Concept | None:
        return self.alias_table().get(token)

    def by_system(self, system: TermSystem) -> tuple[Concept, ...]:
        return tuple(concept for concept in self.concepts if concept.system is system)
