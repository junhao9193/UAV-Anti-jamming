"""Small training schedules shared by Stage 5 runners."""

from __future__ import annotations


def epsilon_by_episode(*, episode: int, epsilon_start: float, epsilon_min: float, epsilon_decay: float) -> float:
    """Baseline multiplicative epsilon schedule."""
    return float(max(float(epsilon_min), float(epsilon_start) * (float(epsilon_decay) ** int(episode))))


__all__ = ["epsilon_by_episode"]
