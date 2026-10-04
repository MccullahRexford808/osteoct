"""Configuration loading: group defaults, deep merge and dotted CLI overrides."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GROUPS: tuple[str, ...] = ("ontology", "acquisition", "cohort", "pipeline", "model", "train")


class ConfigError(Exception):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing configuration file: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root must be a mapping: {path.name}")
    return payload


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = config
    for part in parts[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


def parse_override(token: str) -> tuple[str, Any]:
    if "=" not in token:
        raise ConfigError(f"override must be key=value, received {token!r}")
    key, raw = token.split("=", 1)
    return (key.strip(), _coerce(raw.strip()))


def _coerce(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        return yaml.safe_load(raw)
    return raw


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    root: Path
    payload: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        cursor: Any = self.payload
        for part in dotted.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required key {dotted!r} in experiment {self.name!r}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"section {name!r} must be a mapping")
        return value

    def with_overrides(self, overrides: Sequence[str]) -> ExperimentConfig:
        updated = copy.deepcopy(self.payload)
        for token in overrides:
            key, value = parse_override(token)
            set_dotted(updated, key, value)
        return ExperimentConfig(name=self.name, root=self.root, payload=updated)


def _defaults_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    defaults = payload.get("defaults")
    if defaults is None:
        return resolved
    if not isinstance(defaults, list):
        raise ConfigError("defaults must be a list of group entries")
    for entry in defaults:
        if isinstance(entry, str):
            if ":" not in entry:
                raise ConfigError(f"default entry must be group: name, received {entry!r}")
            group, name = entry.split(":", 1)
            group = group.strip()
            name = name.strip()
        elif isinstance(entry, Mapping) and len(entry) == 1:
            group, name = next(iter(entry.items()))
            group = str(group).strip()
            name = str(name).strip()
        else:
            raise ConfigError(f"unreadable default entry: {entry!r}")
        if group not in DEFAULT_GROUPS:
            raise ConfigError(f"unknown configuration group {group!r}")
        resolved[group] = name
    return resolved


def load_experiment(root: Path, name: str, overrides: Sequence[str] = ()) -> ExperimentConfig:
    experiment_path = root / "configs" / "experiment" / f"{name}.yaml"
    payload = load_yaml(experiment_path)
    merged: dict[str, Any] = {}
    selections = _defaults_of(payload)
    for group, group_name in selections.items():
        group_payload = load_yaml(root / "configs" / group / f"{group_name}.yaml")
        merged[group] = group_payload
    body = {key: value for key, value in payload.items() if key != "defaults"}
    merged = deep_merge(merged, body)
    merged["experiment_name"] = name
    config = ExperimentConfig(name=name, root=root, payload=merged)
    if overrides:
        return config.with_overrides(overrides)
    return config


def discovered_experiments(root: Path) -> tuple[str, ...]:
    directory = root / "configs" / "experiment"
    if not directory.exists():
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.yaml")))


def flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            output.update(flatten(value, f"{dotted}."))
        else:
            output[dotted] = value
    return output
