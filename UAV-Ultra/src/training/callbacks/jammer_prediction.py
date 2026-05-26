"""Stage 8 jammer prediction callback：从 surface-only 升级为活动 callback。

职责（plan §5）：
1. 维护 per-env sensing ring buffer（reset / roll）。
2. 从 info 提 jammer_target（baseline (n_envs, C) layout）。
3. 暴露 ``on_aux_loss(trainer, batch, context)`` 返回 weighted BCE tensor。
4. warmup ramp：``reset_jp_state(states, episode=...)`` 内（select 之前）一次性设置每个
   agent 的 ``feature_scale``。

**callback 不**负责 predictor optimizer step / target sync / backward — 这些由
``ValueDecompTrainerBase._critic_step / _target_sync`` 主流程统一处理。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.training.callbacks.base import TrainHookContext, TrainingCallback


def jammer_target_from_info(info: dict, *, n_channel: int) -> np.ndarray:
    """Return a multi-hot current-jammer target, preferring the explicit field."""
    if "jammer_channels_current_multi_hot" in info:
        target = np.asarray(info["jammer_channels_current_multi_hot"], dtype=np.float32).reshape(-1)
        if target.size != int(n_channel):
            raise ValueError(
                f"jammer_channels_current_multi_hot size {target.size} != n_channel {n_channel}"
            )
        return target

    if "jammer_channels_current" not in info:
        raise KeyError("info lacks jammer_channels_current_multi_hot and jammer_channels_current")
    channels = np.asarray(info["jammer_channels_current"], dtype=np.int64).reshape(-1)
    target = np.zeros((int(n_channel),), dtype=np.float32)
    target[channels] = 1.0
    return target


class JammerPredictionCallback(TrainingCallback):
    """Stage 8 JP active callback。

    Stage 5/7 仅作 target surface；Stage 8 起作真正的 JP head 训练驱动：
    - reset_jp_state(states, episode) 复位 history + 设 feature_scale（select 之前）
    - on_transition_batch(states, next_states, infos) 滚动 history + 提取 jammer target
    - on_aux_loss(trainer, batch, ctx) 计算 BCE aux loss 返回给 trainer 主 backward
    """

    name = "jammer_prediction"

    def __init__(self, *, env_cfg: Any, algo_cfg: Any):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.history_len = int(getattr(algo_cfg, "jammer_history_len", 4))
        self.aux_weight = float(getattr(algo_cfg, "jammer_aux_weight", 0.1))
        self.warmup_episodes = int(getattr(algo_cfg, "jammer_warmup_episodes", 200))
        self.use_feature = bool(getattr(algo_cfg, "use_jammer_feature", True))
        # ring buffer：runner 通过 CallbackManager helper 读
        self.current_sensing_histories: Optional[np.ndarray] = None  # (n_envs, n_agents, H, C)
        self.next_sensing_histories: Optional[np.ndarray] = None
        self.current_jammer_targets: Optional[np.ndarray] = None  # (n_envs, C)  baseline layout
        # 兼容 Stage 5/7 旧字段（surface-only 时缓存的 last targets）
        self.last_targets: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        super().attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=n_envs)
        if not any(hasattr(a, "jammer_predictor") for a in trainer.agents):
            raise RuntimeError(
                "jammer_prediction callback requires a JP-aware trainer "
                "(agents must have jammer_predictor / target_jammer_predictor); "
                "QMIXTrainer constructs JammerAwareMPDQNAgent automatically when "
                "'jammer_prediction' is in algo_cfg.callbacks."
            )

    def reset_jp_state(self, states: np.ndarray, *, episode: int) -> None:
        """Episode reset：history 复位 + feature_scale 按 episode 设置。

        - history 复位用 ``states`` 末尾 ``n_channel`` 切片重复 ``history_len`` 次（baseline
          ``train_qmix_wm_alternating_jammer_prediction.py:360-364``）。
        - feature_scale 在 select **之前** 设置，避免首批 action 用上一 episode 的 scale。
        """
        n_channel = int(self.env_cfg.n_channel)
        sensing = np.asarray(states, dtype=np.float32)[..., -n_channel:]  # (n_envs, n_agents, n_channel)
        self.current_sensing_histories = np.repeat(
            sensing[:, :, None, :], self.history_len, axis=2,
        ).astype(np.float32)
        self.next_sensing_histories = self.current_sensing_histories.copy()
        scale = min(1.0, float(episode) / max(1, self.warmup_episodes))
        scale = scale if self.use_feature else 0.0
        for agent in self.trainer.agents:
            set_scale = getattr(agent, "set_feature_scale", None)
            if set_scale is not None:
                set_scale(scale)

    def on_transition_batch(
        self,
        *,
        states: Any,
        actions: Any,
        action_discrete: Any,
        action_params: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
        infos: Sequence[dict],
    ) -> None:
        """每步 step_wait 之后：滚动 next_sensing_histories + 从 infos 提 jammer target。

        runner 必须在调本钩子**之后、commit_jp_history_swap 之前** 用
        ``CallbackManager.get_jp_buffer_fields()`` 拿字段调 ``store_transition_batch``。
        """
        del actions, action_discrete, action_params, rewards, dones
        if self.current_sensing_histories is None:
            return
        n_channel = int(self.env_cfg.n_channel)
        next_states_arr = np.asarray(next_states, dtype=np.float32)
        new_slice = next_states_arr[..., -n_channel:]  # (n_envs, n_agents, n_channel)
        self.next_sensing_histories = np.concatenate(
            [
                self.current_sensing_histories[:, :, 1:, :],
                new_slice[:, :, None, :],
            ],
            axis=2,
        ).astype(np.float32)

        if len(infos) == 0:
            self.current_jammer_targets = None
            self.last_targets = None
        else:
            self.current_jammer_targets = np.stack(
                [jammer_target_from_info(info, n_channel=n_channel) for info in infos],
                axis=0,
            ).astype(np.float32)  # (n_envs, C)
            self.last_targets = self.current_jammer_targets  # Stage 5/7 兼容字段

    # ------------------------------------------------------------------
    # aux loss（trainer 主 backward 入口）
    # ------------------------------------------------------------------
    def on_aux_loss(
        self,
        trainer: Any,
        batch: dict,
        context: TrainHookContext,
    ) -> Optional[torch.Tensor]:
        """BCE aux loss，broadcast (B, C) target 到每个 agent。

        active JP callback 缺字段直接 raise，**不**静默 no-op。no-op 路径（hook_context=None）
        由 base trainer ``_collect_aux_losses`` 短路兜底。
        """
        del context  # unused
        if batch is None:
            raise ValueError("JammerPredictionCallback.on_aux_loss received batch=None")
        missing = [k for k in ("sensing_history", "jammer_target") if k not in batch]
        if missing:
            raise ValueError(
                f"JammerPredictionCallback active but batch missing keys {missing}; "
                "JP-aware trainer/buffer must provide these fields"
            )
        history = batch["sensing_history"]  # (B, N, H, C)
        target = batch["jammer_target"]      # (B, C) — broadcast to all agents
        n_agents = int(history.shape[1])
        total: Optional[torch.Tensor] = None
        for i, agent in enumerate(trainer.agents):
            h_i = history[:, i]  # (B, H, C)
            logits = agent.jammer_predictor(h_i)  # (B, C)
            loss_i = F.binary_cross_entropy_with_logits(logits, target)
            total = loss_i if total is None else (total + loss_i)
        if total is None:
            return None
        return float(self.aux_weight) * total / float(max(1, n_agents))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "history_len": int(self.history_len),
            "aux_weight": float(self.aux_weight),
            "warmup_episodes": int(self.warmup_episodes),
            "use_feature": bool(self.use_feature),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Stage 6 strict callback reload 会调本方法（基类默认对非空 state 直接 raise，必须 override）。"""
        allowed = {"history_len", "aux_weight", "warmup_episodes", "use_feature"}
        if strict and (set(state) - allowed):
            raise ValueError(
                f"{self.name}: unexpected callback state keys {sorted(set(state) - allowed)}"
            )
        if "history_len" in state:
            self.history_len = int(state["history_len"])
        if "aux_weight" in state:
            self.aux_weight = float(state["aux_weight"])
        if "warmup_episodes" in state:
            self.warmup_episodes = int(state["warmup_episodes"])
        if "use_feature" in state:
            self.use_feature = bool(state["use_feature"])


__all__ = ["JammerPredictionCallback", "jammer_target_from_info"]
