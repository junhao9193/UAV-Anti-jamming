"""MAPPO 评估期策略：deterministic act（argmax discrete + Beta mean）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.schema import EnvConfig, MAPPOConfig


class MAPPOEvalPolicy:
    """每 agent 用 ``trainer.act(deterministic=True)`` 选动作。

    返回 ``[(discrete, params_flat), ...]`` 与 MP-DQN 族 evaluator 同构 —— 但 MAPPO
    的 ``params_flat`` 仅在 chosen action 槽位填入连续值，其它槽位为 0。
    """

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: Optional[MAPPOConfig],
        trainer,
        deterministic: bool = True,
    ):
        if trainer is None:
            raise ValueError("MAPPOEvalPolicy requires a trainer (with .actor/.critic)")
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.trainer = trainer
        self.deterministic = bool(deterministic)
        self.n_agents = int(env_cfg.n_ch)
        self.n_actions = int(trainer.n_actions)
        self.cont_dim = int(trainer.cont_dim)

    def select_actions(
        self, states: List[np.ndarray], global_state: Optional[np.ndarray] = None
    ) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        if global_state is None:
            # 默认：concat 各 agent 自己的 obs 作为 global_state
            global_state = np.concatenate(
                [np.asarray(s, dtype=np.float32).reshape(-1) for s in states]
            )
        out: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            result = self.trainer.act(
                obs=states[i],
                global_state=global_state,
                agent_id=i,
                deterministic=self.deterministic,
            )
            params_flat = np.zeros(
                (self.n_actions * self.cont_dim,), dtype=np.float32
            )
            start = int(result.action_discrete) * self.cont_dim
            params_flat[start : start + self.cont_dim] = result.action_cont.astype(np.float32)
            out.append((int(result.action_discrete), params_flat))
        return out


__all__ = ["MAPPOEvalPolicy"]
