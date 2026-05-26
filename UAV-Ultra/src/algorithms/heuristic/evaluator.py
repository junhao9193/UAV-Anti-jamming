"""``HeuristicEvalPolicy``：每 agent 一份规则策略 + 各自的本地观测。

签名遵循 ``build_evaluator(name="heuristic", *, env_cfg, ...)``：trainer 永远为 None。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.algorithms.heuristic.policies import (
    HeuristicDims,
    build_heuristic_policy,
)
from src.config import specs
from src.config.schema import EnvConfig


class HeuristicEvalPolicy:
    """N 个 agent 各持一份规则 policy，对齐 baseline per-agent seed 分流。"""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg=None,
        trainer=None,
        policy_name: str = "greedy_sensing",
        seed: int = 0,
        power_mode: str = "quality_adaptive",
    ):
        del algo_cfg, trainer  # heuristic 无 trainer / algo_cfg
        self.env_cfg = env_cfg
        self.n_agents = int(env_cfg.n_ch)
        self.policy_name = str(policy_name)
        self.seed = int(seed)

        self.dims = HeuristicDims(
            n_channel=int(env_cfg.n_channel),
            n_des=int(specs.n_des(env_cfg)),
            n_actions=int(specs.action_dim(env_cfg)),
            param_dim=int(specs.param_dim_per_action(env_cfg)),
        )
        self.policies = [
            build_heuristic_policy(
                self.policy_name,
                self.dims,
                seed=self.seed + 1009 * i,
                power_mode=str(power_mode),
            )
            for i in range(self.n_agents)
        ]
        self.policy = self.policies[0]

    def select_actions(self, states: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        out: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            discrete, params = self.policies[i].select_action(states[i])
            out.append((int(discrete), np.asarray(params, dtype=np.float32)))
        return out


__all__ = ["HeuristicEvalPolicy"]
