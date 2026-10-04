"""Atomic checkpoints whose digest covers the tensor payload, not the container."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    seed: int
    step: int
    epoch: int
    arm: str
    run: str
    state: dict[str, dict[str, Tensor]]
    metric: float
    extra: dict[str, str]

    def as_mapping(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "step": self.step,
            "epoch": self.epoch,
            "arm": self.arm,
            "run": self.run,
            "metric": self.metric,
            "extra": dict(sorted(self.extra.items())),
        }


def content_digest(state: Mapping[str, Mapping[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for group in sorted(state):
        digest.update(group.encode("utf-8"))
        digest.update(b"\0")
        for key in sorted(state[group]):
            digest.update(key.encode("utf-8"))
            tensor = state[group][key].detach().to(torch.float64).contiguous()
            digest.update(tensor.numpy().tobytes())
            digest.update(b"\n")
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: CheckpointPayload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save({"meta": payload.as_mapping(), "state": payload.state}, handle)
        temp_path.chmod(0o644)
        temp_path.replace(path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return content_digest(payload.state)


def load_checkpoint(path: Path) -> CheckpointPayload:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    return CheckpointPayload(
        seed=int(meta["seed"]),
        step=int(meta["step"]),
        epoch=int(meta["epoch"]),
        arm=str(meta["arm"]),
        run=str(meta["run"]),
        state=payload["state"],
        metric=float(meta["metric"]),
        extra={str(key): str(value) for key, value in meta.get("extra", {}).items()},
    )


def checkpoint_metadata(seed: int, step: int, epoch: int, arm: str, run: str, metric: float, digest: str) -> str:
    return json.dumps(
        {"seed": seed, "step": step, "epoch": epoch, "arm": arm, "run": run, "metric": metric, "content": digest},
        sort_keys=True,
    )


def state_matches(first: Mapping[str, Mapping[str, Tensor]], second: Mapping[str, Mapping[str, Tensor]]) -> bool:
    return content_digest(first) == content_digest(second)
