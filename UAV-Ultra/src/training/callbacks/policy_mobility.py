"""Zero-delta policy mobility action adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import specs
from src.training.callbacks.base import TrainingCallback


class PolicyMobilityCallback(TrainingCallback):
    name = "policy_mobility"

    def __init__(self, *, env_cfg: Any):
        self.env_cfg = env_cfg
        self.base_param_dim = int(specs.total_param_dim(env_cfg))
        self.full_param_dim = int(specs.per_ch_param_dim(env_cfg))

    def _adapt_joint_actions(self, joint_actions: list[tuple[int, Any]]) -> list[tuple[int, np.ndarray]]:
        out: list[tuple[int, np.ndarray]] = []
        for discrete, params in joint_actions:
            arr = np.asarray(params, dtype=np.float32).reshape(-1)
            if self.env_cfg.uav_mobility_control == "policy":
                if arr.size == self.base_param_dim:
                    arr = np.concatenate([arr, np.zeros(3, dtype=np.float32)], axis=0)
                elif arr.size != self.full_param_dim:
                    raise ValueError(
                        f"policy_mobility expected param dim {self.base_param_dim} or "
                        f"{self.full_param_dim}, got {arr.size}"
                    )
            out.append((int(discrete), arr.astype(np.float32, copy=False)))
        return out

    def on_action_selected(self, actions: Any) -> Any:
        if self.env_cfg.uav_mobility_control != "policy":
            return actions
        if not isinstance(actions, list):
            return actions
        if not actions:
            return actions
        if isinstance(actions[0], tuple):
            return self._adapt_joint_actions(actions)
        return [self._adapt_joint_actions(list(joint)) for joint in actions]


__all__ = ["PolicyMobilityCallback"]
