"""VDN 评估期策略：每 agent greedy。与 IQL evaluator 同构。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.schema import EnvConfig, VDNConfig


class VDNEvalPolicy:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: Optional[VDNConfig],
        trainer,
    ):
        if trainer is None:
            raise ValueError("VDNEvalPolicy requires a trainer (with .agents)")
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.trainer = trainer
        self.n_agents = int(env_cfg.n_ch)

    def select_actions(self, states: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        out: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            action_discrete, action_params = self.trainer.agents[i].select_action(
                states[i], epsilon=0.0
            )
            out.append((int(action_discrete), np.asarray(action_params, dtype=np.float32)))
        return out


__all__ = ["VDNEvalPolicy"]
