"""QMIX trainer：``ValueDecompTrainerBase`` 的可学习 mixer 具化。

与 baseline ``algorithms/mpdqn/qmix/trainer_greedy_actor.py`` **标准 train_step** 等价
（plan locked #2：``train_step_value_expansion`` 由 Stage 5 callback 实现；
Stage 8 起 ``jammer_prediction`` callback 触发 JP-aware agent + buffer）。

Stage 8：override `_build_agent` / `_build_replay_buffer`，当 `algo_cfg.callbacks` 含
`"jammer_prediction"` 时构造 `JammerAwareMPDQNAgent` + `JointReplayBuffer(track_jammer=True)`。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.algorithms.common.agents.jammer_aware_mpdqn_agent import JammerAwareMPDQNAgent
from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent
from src.algorithms.common.buffers.joint_replay import JointReplayBuffer
from src.algorithms.common.networks.mixers import QMIXMixer
from src.algorithms.common.value_decomp import ValueDecompTrainerBase
from src.config import specs
from src.config.schema import EnvConfig, QMIXConfig


class QMIXTrainer(ValueDecompTrainerBase):
    """QMIX（hypernet 单调 mixer）+ MP-DQN agents + joint replay。"""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: QMIXConfig,
        device: Optional[str] = None,
    ):
        n_agents = int(env_cfg.n_ch)
        state_dim = int(specs.state_dim(env_cfg))
        n_actions = int(specs.action_dim(env_cfg))
        param_dim = int(specs.param_dim_per_action(env_cfg))
        global_state_dim = n_agents * state_dim

        if device is None:
            chosen = algo_cfg.device
            resolved_device = "cuda" if (chosen == "auto" and torch.cuda.is_available()) else (
                "cpu" if chosen == "auto" else chosen
            )
        else:
            resolved_device = device

        # loader 已确保 lr_mixer 不为 None（None → lr_q 落定）
        self._mixing_hidden_dim = int(algo_cfg.mixing_hidden_dim)
        self._hypernet_hidden_dim = int(algo_cfg.hypernet_hidden_dim)

        # ★ Stage 8：先存 cfg + JP enabled flag，再 super.__init__（hook 内部读取）
        self._env_cfg = env_cfg
        self._algo_cfg = algo_cfg
        self._jp_enabled_flag = "jammer_prediction" in list(getattr(algo_cfg, "callbacks", []))

        super().__init__(
            n_agents=n_agents,
            state_dim=state_dim,
            n_actions=n_actions,
            param_dim=param_dim,
            global_state_dim=global_state_dim,
            buffer_capacity=int(algo_cfg.buffer_capacity),
            batch_size=int(algo_cfg.batch_size),
            gamma=float(algo_cfg.gamma),
            lr_actor=float(algo_cfg.lr_actor),
            lr_q=float(algo_cfg.lr_q),
            lr_mixer=float(algo_cfg.lr_mixer),
            target_update_interval=int(algo_cfg.target_update_interval),
            use_amp=bool(algo_cfg.use_amp),
            max_grad_norm=float(algo_cfg.max_grad_norm),
            value_target_clip=float(algo_cfg.value_target_clip),
            device=resolved_device,
        )

    def _build_mixer(self) -> nn.Module:
        return QMIXMixer(
            n_agents=self.n_agents,
            global_state_dim=self.global_state_dim,
            mixing_hidden_dim=self._mixing_hidden_dim,
            hypernet_hidden_dim=self._hypernet_hidden_dim,
        )

    # ----------------------- Stage 8 JP-aware factory overrides -----------------------

    def _build_agent(self, idx: int, *, lr_actor: float, lr_q: float) -> MPDQNAgent:
        if self._jp_enabled_flag:
            return JammerAwareMPDQNAgent(
                state_dim=self.state_dim,                    # ★ raw（baseline 同款）
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                n_channel=int(self._env_cfg.n_channel),
                history_len=int(self._algo_cfg.jammer_history_len),
                jammer_hidden_dim=int(self._algo_cfg.jammer_pred_hidden_dim),
                use_jammer_feature=bool(self._algo_cfg.use_jammer_feature),
                lr_jammer=self._algo_cfg.lr_jammer,
                batch_size=self.batch_size,
                gamma=self.gamma,
                lr_actor=lr_actor,
                lr_q=lr_q,
                target_update_interval=self.target_update_interval,
                use_amp=self.use_amp,
                max_grad_norm=self.max_grad_norm,
                device=str(self.device),
            )
        return super()._build_agent(idx, lr_actor=lr_actor, lr_q=lr_q)

    def _build_replay_buffer(self, *, capacity: int) -> JointReplayBuffer:
        return JointReplayBuffer(
            capacity=capacity,
            per_agent_reward=self._per_agent_reward(),
            track_jammer=self._jp_enabled_flag,
        )


__all__ = ["QMIXTrainer"]
