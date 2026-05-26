"""QMIX 侧的 world_model adapter（plan locked decision #8：放 qmix 侧避免环形 import）。

与 baseline ``algorithms/world_model/qmix_adapters.py`` 1:1，但 import 方向反转：
依赖 ``QMIXTrainer`` 与 ``algorithms.world_model``，**不**反向 import qmix。

暴露：
- ``MPDQNQMIXDims`` 数据类
- ``QMIXValueTeacher``：target Q_tot(s, u) 教师 + greedy joint action u*(s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.algorithms.qmix.trainer import QMIXTrainer


@dataclass(frozen=True)
class MPDQNQMIXDims:
    n_agents: int
    agent_state_dim: int
    n_actions: int
    param_dim: int


class QMIXValueTeacher:
    """QMIX target critic 教师，供 RSSM 价值一致性 rollout 使用。

    冻结 target 网络参数；允许梯度回流到**输入 state**（用于 L_VC 塑形世界模型动力学）。
    """

    def __init__(self, trainer: QMIXTrainer, dims: MPDQNQMIXDims):
        self.trainer = trainer
        self.dims = dims
        # 冻结 target critic 参数；允许 input state 端梯度
        for agent in self.trainer.agents:
            for p in agent.target_q_net.parameters():
                p.requires_grad_(False)
            agent.target_q_net.eval()
        for p in self.trainer.target_mixer.parameters():
            p.requires_grad_(False)
        self.trainer.target_mixer.eval()

    def _reshape_state(self, global_state: torch.Tensor) -> torch.Tensor:
        """``(B, N*S) -> (B, N, S)``。"""
        if global_state.ndim != 2:
            raise ValueError(f"global_state must be (B, N*S), got {tuple(global_state.shape)}")
        bsz = int(global_state.shape[0])
        return global_state.view(bsz, int(self.dims.n_agents), int(self.dims.agent_state_dim))

    def _maybe_aug(self, agent, s_i: torch.Tensor, *, target: bool) -> torch.Tensor:
        """Stage 8：JP-aware agent 时用 agent.augment_state(..., sensing_history=None) 兜底。

        WM 想象 state 无 sensing_history，agent._default_history 会用 state 末尾 n_channel
        切片 repeat 兜底（baseline `trainer_jammer_prediction.py:192` 同款）。
        """
        augment_fn = getattr(agent, "augment_state", None)
        if augment_fn is None:
            return s_i
        aug, _, _ = augment_fn(s_i, None, target=target)
        return aug

    @torch.no_grad()
    def greedy_action(self, global_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decentralized greedy joint action via **online** networks (Double DQN style)。

        Returns:
            action_discrete: ``(B, N)`` int64
            action_params_flat: ``(B, N, A*P)`` float32
        """
        state = self._reshape_state(global_state)
        bsz = int(state.shape[0])

        a_list = []
        params_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            aug_i = self._maybe_aug(self.trainer.agents[i], s_i, target=False)
            params_all = self.trainer.agents[i].actor(aug_i)  # (B, A, P)
            q_eval = self.trainer.agents[i].q_net(aug_i, params_all)  # (B, A)
            a_i = torch.argmax(q_eval, dim=1)  # (B,)
            a_list.append(a_i)
            params_list.append(params_all.reshape(bsz, -1))

        action_discrete = torch.stack(a_list, dim=1).to(torch.long)
        action_params_flat = torch.stack(params_list, dim=1).to(torch.float32)
        return action_discrete, action_params_flat

    def q_tot_target(
        self,
        global_state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params_flat: torch.Tensor,
    ) -> torch.Tensor:
        """``target_q_net`` + ``target_mixer`` → ``Q_tot``。教师参数冻结，state 端可回流梯度。"""
        state = self._reshape_state(global_state)
        bsz = int(state.shape[0])

        q_sa_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            aug_i = self._maybe_aug(self.trainer.agents[i], s_i, target=True)
            a_i = action_discrete[:, i].view(-1, 1)
            params_i = action_params_flat[:, i, :].view(
                bsz, int(self.dims.n_actions), int(self.dims.param_dim)
            )
            q_all_i = self.trainer.agents[i].target_q_net(aug_i, params_i)
            q_sa_i = q_all_i.gather(1, a_i)  # (B, 1)
            q_sa_list.append(q_sa_i)

        agent_qs = torch.cat(q_sa_list, dim=1)  # (B, N)
        q_tot = self.trainer.target_mixer(agent_qs, global_state)  # (B, 1)
        return q_tot.to(torch.float32)


__all__ = ["MPDQNQMIXDims", "QMIXValueTeacher"]
