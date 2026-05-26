"""Metric aggregation helpers for baseline-compatible training logs."""

from __future__ import annotations

import numpy as np


def success_rate_from_suc(total_suc: float, *, total_links: float) -> float:
    """Map baseline suc values in {-3,+1} to [0,1]."""
    links = max(float(total_links), 1.0)
    avg_suc_per_link = float(total_suc) / links
    return float(np.clip((avg_suc_per_link + 3.0) / 4.0, 0.0, 1.0))


def aggregate_baseline_metrics(
    *,
    energy: np.ndarray,
    jump: np.ndarray,
    suc: np.ndarray,
    steps_done: int,
    n_envs: int,
    n_agents: int,
    n_des: int,
) -> dict[str, float]:
    total_links = float(max(1, int(steps_done)) * int(n_envs) * int(n_agents) * int(n_des))
    total_energy = float(np.sum(np.asarray(energy, dtype=np.float32)))
    total_jump = float(np.sum(np.asarray(jump, dtype=np.float32)))
    total_suc = float(np.sum(np.asarray(suc, dtype=np.float32)))
    return {
        "energy": total_energy / total_links,
        "jump": total_jump / total_links,
        "success_rate": success_rate_from_suc(total_suc, total_links=total_links),
    }


__all__ = ["aggregate_baseline_metrics", "success_rate_from_suc"]
