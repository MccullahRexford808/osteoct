"""Seed handling: one entry point, stored with every checkpoint."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_determinism(threads: int = 1) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if bool(torch.backends.cudnn.is_available()):  # type: ignore[no-untyped-call]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> np.random.Generator:
    configure_determinism()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def seed_list(base: int, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("at least one seed is required")
    generator = np.random.default_rng(base)
    return tuple(int(value) for value in generator.integers(0, 2**31 - 1, size=count).tolist())


def derive_seed(base: int, stream: str) -> int:
    digest = abs(hash((base, stream)))
    return int(digest % (2**31 - 1))


def rng_for(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, stream))


def seed_report(seeds: tuple[int, ...]) -> dict[str, object]:
    return {"count": len(seeds), "seeds": list(seeds), "base": seeds[0] if seeds else None}
