"""QMIX 评估期策略：每 agent greedy。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.schema import EnvConfig, QMIXConfig


class QMIXEvalPolicy:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: Optional[QMIXConfig],
        trainer,
    ):
        if trainer is None:
            raise ValueError("QMIXEvalPolicy requires a trainer (with .agents)")
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.trainer = trainer
        self.n_agents = int(env_cfg.n_ch)

    def select_actions(
        self,
        states: List[np.ndarray],
        sensing_histories: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Greedy 选动作。``sensing_histories`` 为 JP-aware 模型的每 agent 感知历史
        ``(n_agents, H, C)``；None 时退回各 agent 的默认 history（仅 JP-off 安全）。

        Stage 10 修复：评估期必须把真实滚动的 sensing history 透传给 ``select_action``，
        否则 JP 特征退化成 ``_default_history``，eval SR 系统性低估。
        """
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        out: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            agent = self.trainer.agents[i]
            # JP-aware agent 才接受 sensing_history；plain MPDQNAgent 的 select_action
            # 不带该 kwarg，传了会 TypeError。用 jammer_predictor 探测区分。
            if hasattr(agent, "jammer_predictor"):
                hist_i = None if sensing_histories is None else sensing_histories[i]
                action_discrete, action_params = agent.select_action(
                    states[i], epsilon=0.0, sensing_history=hist_i
                )
            else:
                action_discrete, action_params = agent.select_action(states[i], epsilon=0.0)
            out.append((int(action_discrete), np.asarray(action_params, dtype=np.float32)))
        return out


__all__ = ["QMIXEvalPolicy"]
