"""Atomic artefact writes, digests and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SKIP_DIRECTORIES: frozenset[str] = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git", ".hypothesis"})


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        temp_path.chmod(0o644)
        temp_path.replace(path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(path: Path, payload: Any, indent: int = 1) -> None:
    serialised = json.dumps(payload, indent=indent, sort_keys=False, ensure_ascii=False)
    atomic_write_text(path, serialised + "\n")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        temp_path.chmod(0o644)
        temp_path.replace(path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        yield path


def relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def directory_digest(root: Path, excluded: Iterable[str] = ()) -> tuple[int, str, dict[str, str]]:
    skip = {name for name in excluded}
    entries: dict[str, str] = {}
    for path in iter_files(root):
        relative = relative_posix(root, path)
        if relative in skip:
            continue
        entries[relative] = sha256_file(path)
    aggregate = hashlib.sha256()
    for relative in sorted(entries):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(entries[relative].encode("utf-8"))
        aggregate.update(b"\n")
    return (len(entries), aggregate.hexdigest(), entries)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def counts_by_suffix(root: Path) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for path in iter_files(root):
        suffix = path.suffix or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
    return counts
