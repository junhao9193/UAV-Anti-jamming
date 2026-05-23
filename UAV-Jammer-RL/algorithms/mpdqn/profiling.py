from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def profile_section(owner: Any, name: str):
    profiler = getattr(owner, "profiler", None)
    if profiler is None:
        return nullcontext()
    return profiler.section(f"train_step.{name}", sync_cuda=True)


def set_profiler(owner: Any, profiler: Any) -> None:
    owner.profiler = profiler
    for agent in getattr(owner, "agents", []):
        agent.profiler = profiler


def should_log_loss(owner: Any) -> bool:
    interval = int(getattr(owner, "loss_log_interval", 1))
    if interval <= 0:
        return False
    return int(getattr(owner, "learn_steps", 0)) % interval == 0
