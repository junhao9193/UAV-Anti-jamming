"""Stage 8 JP-aware MPDQN agent.

baseline 对齐：``algorithms/mpdqn/qmix/trainer_jammer_prediction.py:131-280``。

关键约束（plan §2）：

- ``self.state_dim = raw_state_dim``（不变；与 plain MPDQNAgent 公开契约一致）。
- ``actor / q_net / target_actor / target_q_net`` 内部第一层 ``Linear`` 的
  ``in_features = state_dim + n_channel``（由 JP-aware 子类内部消费）。
- ``augment_state(state, sensing_history=None, *, target=False)`` 返回
  ``(augmented, logits, probs)``：``feature = probs.detach() * feature_scale`` 是核心，
  Q/actor loss 通过 feature 路径**不**回传 predictor；BCE aux loss 是 predictor 唯一梯度源。
- ``target_jammer_predictor.requires_grad`` 全 False（与 target_actor / target_q_net 同处理）。
- ``select_action_batch(state, epsilon, sensing_history=None)`` 在 ``sensing_history`` 为 None
  时走 ``_default_history(state)`` 兜底（baseline-compatible），**不** raise。
"""

from __future__ import annotations

import copy
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import amp as torch_amp

from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent
from src.algorithms.common.networks.jammer_predictor import (
    JammerAwareMPDQNActor,
    JammerAwareMPDQNQNetwork,
    JammerPredictionHead,
)


class JammerAwareMPDQNAgent(MPDQNAgent):
    """MP-DQN agent with auxiliary jammer predictor.

    BCE aux loss 训 predictor；其 sigmoid 概率（``.detach()``）拼接到 ``state`` 末尾作为
    actor / Q-net 输入。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        *,
        n_channel: int,
        history_len: int,
        jammer_hidden_dim: int = 64,
        use_jammer_feature: bool = True,
        lr_jammer: Optional[float] = None,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        target_update_interval: int = 200,
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
        device: Optional[str] = None,
    ):
        # 父类先建出 plain MPDQNActor/QNetwork；下面立刻替换为 JP-aware 版本。
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            param_dim=param_dim,
            batch_size=batch_size,
            gamma=gamma,
            lr_actor=lr_actor,
            lr_q=lr_q,
            target_update_interval=target_update_interval,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        self.n_channel = int(n_channel)
        self.jammer_history_len = int(history_len)
        self.use_jammer_feature = bool(use_jammer_feature)
        self.feature_scale = 0.0   # warmup 从 0→1 由 callback 设置

        # 替换 actor / q_net 为 JP-aware 版本（state_dim 仍 raw；内部第一层 +n_channel）。
        self.actor = JammerAwareMPDQNActor(
            self.state_dim, self.n_actions, self.param_dim, self.n_channel,
        ).to(self.device)
        self.q_net = JammerAwareMPDQNQNetwork(
            self.state_dim, self.n_actions, self.param_dim, self.n_channel,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_q_net.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=float(lr_q))

        # JP head + target + opt
        self.jammer_predictor = JammerPredictionHead(
            history_len=self.jammer_history_len,
            n_channel=self.n_channel,
            hidden_dim=int(jammer_hidden_dim),
        ).to(self.device)
        self.target_jammer_predictor = copy.deepcopy(self.jammer_predictor).to(self.device)
        for p in self.target_jammer_predictor.parameters():
            p.requires_grad_(False)
        jp_lr = float(lr_q) if lr_jammer is None else float(lr_jammer)
        self.jammer_predictor_opt = torch.optim.Adam(self.jammer_predictor.parameters(), lr=jp_lr)

    # ------------------------------------------------------------------
    # JP utilities
    # ------------------------------------------------------------------
    def set_feature_scale(self, value: float) -> None:
        self.feature_scale = float(value)

    def _default_history(self, state: torch.Tensor) -> torch.Tensor:
        """baseline `trainer_jammer_prediction.py:192` 兜底：state 末尾 n_channel 切片 repeat history_len 次。"""
        sensing = state[:, -self.n_channel:].unsqueeze(1)                # (B, 1, C)
        return sensing.repeat(1, self.jammer_history_len, 1).contiguous()  # (B, H, C)

    def augment_state(
        self,
        state: torch.Tensor,
        sensing_history: Optional[torch.Tensor] = None,
        *,
        target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对齐 baseline `trainer_jammer_prediction.py:196-213`。

        梯度边界：
        - ``feature = probs.detach() * feature_scale`` —— Q/actor loss 不回传到 predictor。
        - ``target=True`` 时用 ``target_jammer_predictor``（requires_grad 全 False）。
        """
        if sensing_history is None:
            sensing_history = self._default_history(state)
        predictor = self.target_jammer_predictor if bool(target) else self.jammer_predictor
        logits = predictor(sensing_history)
        probs = torch.sigmoid(logits)
        if self.use_jammer_feature:
            feature = probs.detach() * float(self.feature_scale)
        else:
            feature = torch.zeros_like(probs)
        augmented = torch.cat([state, feature], dim=-1)
        return augmented, logits, probs

    # ------------------------------------------------------------------
    # action selection（覆盖父类；状态先 augment 再喂 actor/q_net）
    # ------------------------------------------------------------------
    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
        sensing_history: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray]:
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)
        hist_t: Optional[torch.Tensor] = None
        if sensing_history is not None:
            hist_t = torch.as_tensor(sensing_history, dtype=torch.float32, device=self.device)
            hist_t = hist_t.reshape(1, self.jammer_history_len, self.n_channel)
        with torch.no_grad():
            augmented, _, _ = self.augment_state(state_t, hist_t, target=False)
            params_all = self.actor(augmented)
            q_values = self.q_net(augmented, params_all)

        if random.random() < float(epsilon):
            action_discrete = random.randrange(self.n_actions)
        else:
            action_discrete = int(torch.argmax(q_values, dim=1).item())

        params_flat = params_all.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action_discrete, params_flat

    def select_action_batch(
        self,
        states: np.ndarray,
        epsilon: float,
        sensing_history: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must be (B,{self.state_dim}), got shape={states.shape}")
        state_t = torch.from_numpy(states).to(self.device)
        hist_t: Optional[torch.Tensor] = None
        if sensing_history is not None:
            hist_t = torch.as_tensor(sensing_history, dtype=torch.float32, device=self.device)
            hist_t = hist_t.reshape(int(states.shape[0]), self.jammer_history_len, self.n_channel)
        with torch.no_grad():
            augmented, _, _ = self.augment_state(state_t, hist_t, target=False)
            params_all = self.actor(augmented)
            q_values = self.q_net(augmented, params_all)

        greedy = torch.argmax(q_values, dim=1).detach().cpu().numpy().astype(np.int32)
        batch_size = int(states.shape[0])
        if float(epsilon) <= 0.0:
            action_discrete = greedy
        elif float(epsilon) >= 1.0:
            action_discrete = np.random.randint(0, self.n_actions, size=batch_size, dtype=np.int32)
        else:
            rnd = np.random.randint(0, self.n_actions, size=batch_size, dtype=np.int32)
            explore = np.random.random(size=batch_size) < float(epsilon)
            action_discrete = np.where(explore, rnd, greedy).astype(np.int32)

        params_flat = params_all.detach().cpu().numpy().reshape(batch_size, -1).astype(np.float32)
        return action_discrete, params_flat


__all__ = ["JammerAwareMPDQNAgent"]
