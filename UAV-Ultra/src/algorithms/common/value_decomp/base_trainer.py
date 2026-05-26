"""值分解 trainer 基类（VDN / QMIX / QPLEX 共享）。

抽自 baseline `mpdqn/qmix/trainer_greedy_actor.py:13-292` 与 `mpdqn/qplex/trainer.py:13-299`
共享段。详见 plan §「API contracts → Trainer Batch Contract」。

Hook 边界（plan locked decision #7 + 修订）：
- ``_build_mixer(self) -> nn.Module``：返回 mixer 实例。
- ``_mix(self, agent_qs, global_state, **extras) -> Tensor[(B, 1)]``。
- ``_target_mix(self, next_agent_qs, next_global_state, **extras) -> Tensor[(B, 1)]``。
- ``_collect_critic_extras(self, state, action_params) -> dict``：critic 前向额外项。
- ``_collect_target_extras(self, next_state) -> dict``：target 前向额外项。
- ``_collect_actor_extras(self, state, params_pred, q_pred_all) -> dict``：actor surrogate 额外项。
- ``_mixer_parameters(self) -> Iterable[Tensor]``：VDN 返回 ()。

更新顺序：``zero_grad(q + mixer) → forward critic → backward → step(q + mixer)
→ zero_grad(actor) + freeze critic → forward actor surrogate → backward → step(actor)
→ unfreeze critic → target_sync 每 N 步一次``。
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp

from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent
from src.algorithms.common.buffers.joint_replay import JointReplayBuffer
from src.algorithms.common.optim.utils import clip_grad_norm, hard_sync_target


@dataclass(frozen=True)
class TDTargetContext:
    """Optional context consumed by Stage 5 value-expansion callbacks.

    The base trainer intentionally keeps these fields loosely typed to avoid a
    reverse dependency from ``algorithms.common`` into ``algorithms.world_model``.
    """

    state_seq: torch.Tensor | None = None
    action_enc_seq: torch.Tensor | None = None
    world_model: Any | None = None
    value_teacher: Any | None = None
    td_cfg: Any | None = None
    alpha_model: float = 0.0


class ValueDecompTrainerBase(ABC):
    """基类，子类只需实现 5 个 hook + ``_build_mixer``。"""

    def __init__(
        self,
        *,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        global_state_dim: int,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        lr_mixer: Optional[float] = None,
        target_update_interval: int = 200,
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
        value_target_clip: Optional[float] = 1000.0,
        device: Optional[str] = None,
    ):
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.global_state_dim = int(global_state_dim)

        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.scaler = torch_amp.GradScaler("cuda", enabled=self.use_amp)
        self.max_grad_norm = float(max_grad_norm)
        self.value_target_clip = None if value_target_clip is None else float(value_target_clip)

        if lr_mixer is None:
            lr_mixer = float(lr_q)

        # Stage 8：通过 _build_agent / _build_replay_buffer hook 替换 literal class，
        # 让 QMIX JP-aware subclass 可以 override 出 JammerAwareMPDQNAgent + track_jammer=True buffer。
        # IQL/VDN/QPLEX 默认 hook 行为 = Stage 4。
        self.agents: List[MPDQNAgent] = [
            self._build_agent(i, lr_actor=float(lr_actor), lr_q=float(lr_q))
            for i in range(self.n_agents)
        ]

        # 子类提供 mixer 与 target mixer
        self.mixer: nn.Module = self._build_mixer().to(self.device)
        self.target_mixer: nn.Module = copy.deepcopy(self.mixer).to(self.device)
        mixer_params = list(self._mixer_parameters())
        self.mixer_opt: Optional[torch.optim.Optimizer] = (
            torch.optim.Adam(mixer_params, lr=float(lr_mixer)) if mixer_params else None
        )

        self.buffer = self._build_replay_buffer(capacity=int(buffer_capacity))
        self.learn_steps = 0
        self._last_aux_loss_values: Dict[str, torch.Tensor] = {}

    # ----------------------- Stage 8 窄 factory hooks -----------------------

    def _per_agent_reward(self) -> bool:
        """IQLTrainer override 为 True；VDN/QMIX/QPLEX 默认 False。"""
        return False

    def _build_agent(self, idx: int, *, lr_actor: float, lr_q: float) -> MPDQNAgent:
        """Stage 8：子类可 override 返回 JammerAwareMPDQNAgent 等子类。

        默认行为字段级 = Stage 4 plain MPDQNAgent。
        """
        return MPDQNAgent(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            param_dim=self.param_dim,
            batch_size=self.batch_size,
            gamma=self.gamma,
            lr_actor=lr_actor,
            lr_q=lr_q,
            target_update_interval=self.target_update_interval,
            use_amp=self.use_amp,
            max_grad_norm=self.max_grad_norm,
            device=str(self.device),
        )

    def _build_replay_buffer(self, *, capacity: int) -> JointReplayBuffer:
        """Stage 8：子类可 override 加 ``track_jammer=True``。默认 plain。"""
        return JointReplayBuffer(
            capacity=capacity,
            per_agent_reward=self._per_agent_reward(),
        )

    # ----------------------- Hooks（子类实现）-----------------------

    @abstractmethod
    def _build_mixer(self) -> nn.Module:
        """返回 mixer 实例（非 target）。"""

    def _mixer_parameters(self) -> Iterable[torch.nn.Parameter]:
        """VDN 覆盖返回 ()。默认返回 ``self.mixer.parameters()``，
        但注意 ``__init__`` 在 ``self.mixer`` 赋值前会调一次这个 hook —— 子类应
        在覆盖时也兼容这种情况（直接返回 ``()``）。"""
        if not hasattr(self, "mixer"):
            return ()
        return self.mixer.parameters()

    def _mix(self, agent_qs: torch.Tensor, global_state: torch.Tensor, **extras: Any) -> torch.Tensor:
        """默认走 ``self.mixer(agent_qs, global_state)``。QPLEX 覆盖以传入 ``max_agent_qs``。"""
        return self.mixer(agent_qs, global_state)

    def _target_mix(
        self, next_agent_qs: torch.Tensor, next_global_state: torch.Tensor, **extras: Any
    ) -> torch.Tensor:
        return self.target_mixer(next_agent_qs, next_global_state)

    def _collect_critic_extras(
        self, state: torch.Tensor, action_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """默认空，QPLEX 覆盖返回 max_agent_qs。"""
        return {}

    def _collect_target_extras(self, next_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """默认空，QPLEX 覆盖（在 ``torch.no_grad()`` 内调用），返回 max_agent_qs。"""
        return {}

    def _collect_actor_extras(
        self,
        state: torch.Tensor,
        params_pred_list: List[torch.Tensor],
        q_pred_all_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """默认空，QPLEX 覆盖返回 max_agent_qs (detached)。"""
        return {}

    # ----------------------- 公共流程 -----------------------

    def _clip_value_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_target_clip is None or self.value_target_clip <= 0.0:
            return x
        return torch.clamp(x, min=-self.value_target_clip, max=self.value_target_clip)

    def _compute_td_target(
        self,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_q_tot: torch.Tensor,
        *,
        target_context: TDTargetContext | None = None,
    ) -> torch.Tensor:
        td = reward + (1.0 - done) * self.gamma * next_q_tot
        if (
            target_context is not None
            and float(target_context.alpha_model) > 0.0
            and target_context.world_model is not None
            and target_context.value_teacher is not None
            and target_context.state_seq is not None
            and target_context.action_enc_seq is not None
            and target_context.td_cfg is not None
        ):
            from src.algorithms.world_model.action_encoding import encode_joint_action_exec
            from src.algorithms.world_model.value_expansion import rollout_td_lambda_return
            from src.config import specs

            teacher = target_context.value_teacher
            env_cfg = getattr(teacher, "env_cfg", None)
            if env_cfg is None:
                raise ValueError("TDTargetContext.value_teacher must expose env_cfg for action encoding")

            def _policy_fn(global_state: torch.Tensor):
                action_discrete, action_params = teacher.greedy_action(global_state)
                action_enc = encode_joint_action_exec(
                    action_discrete,
                    action_params,
                    n_agents=int(env_cfg.n_ch),
                    n_channel=int(env_cfg.n_channel),
                    n_des=int(specs.n_des(env_cfg)),
                    n_actions=int(specs.action_dim(env_cfg)),
                    param_dim=int(specs.param_dim_per_action(env_cfg)),
                    power_min_dbm=float(env_cfg.uav_power_min),
                    power_max_dbm=float(env_cfg.uav_power_max),
                )
                return action_enc, action_discrete, action_params

            model_td, _ = rollout_td_lambda_return(
                wm=target_context.world_model,
                state_seq=target_context.state_seq,
                action_seq=target_context.action_enc_seq,
                policy_fn=_policy_fn,
                q_tot_target_fn=teacher.q_tot_target,
                cfg=target_context.td_cfg,
            )
            alpha = float(target_context.alpha_model)
            td = (1.0 - alpha) * td + alpha * model_td.to(dtype=td.dtype, device=td.device)
        return self._clip_value_target(td)

    def _autocast(self):
        return torch_amp.autocast("cuda", enabled=self.use_amp)

    # ----------------------- Stage 8 JP helpers -----------------------

    def _maybe_augment_state(
        self,
        agent: MPDQNAgent,
        state: torch.Tensor,
        sensing_history: Optional[torch.Tensor],
        *,
        target: bool,
    ) -> torch.Tensor:
        """Augment state via agent.augment_state when available; identity otherwise.

        - JP-aware agent: ``aug, _, _ = agent.augment_state(state, sensing_history, target=target)``
        - plain MPDQNAgent: 直接返回 raw ``state``（Stage 4 行为不退化）。
        """
        augment_fn = getattr(agent, "augment_state", None)
        if augment_fn is None:
            return state
        aug, _, _ = augment_fn(state, sensing_history, target=target)
        return aug

    def _collect_aux_losses(
        self,
        batch: Dict[str, torch.Tensor],
        context: Any,
    ) -> torch.Tensor:
        """Sum of all callback aux losses（Stage 8 on_aux_loss hook 入口）。

        - ``context is None`` 或无 aux fns：返回 zeros（plain Stage 4/5/6/7 行为字段级一致）。
        - active JP callback 缺 batch 字段会 raise（在 callback 内部，不在这里吞错误）。
        """
        self._last_aux_loss_values = {}
        if context is None or not getattr(self, "_aux_loss_fns", None):
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for fn in self._aux_loss_fns:
            t = fn(self, batch, context)
            if t is not None:
                total = total + t
                cb = getattr(fn, "__self__", None)
                cb_name = str(getattr(cb, "name", "aux"))
                key = "loss_jammer" if cb_name == "jammer_prediction" else f"loss_{cb_name}"
                prev = self._last_aux_loss_values.get(key)
                self._last_aux_loss_values[key] = t.detach() if prev is None else prev + t.detach()
        return total

    def select_actions(
        self, states: List[np.ndarray], epsilon: float
    ) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        out: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            action_discrete, action_params = self.agents[i].select_action(states[i], epsilon)
            out.append((int(action_discrete), np.asarray(action_params, dtype=np.float32)))
        return out

    def store_transition(
        self,
        states: List[np.ndarray],
        actions: List[Tuple[int, np.ndarray]],
        rewards: np.ndarray,
        next_states: List[np.ndarray],
        done: bool = False,
        *,
        sensing_history: Optional[np.ndarray] = None,
        next_sensing_history: Optional[np.ndarray] = None,
        jammer_target: Optional[np.ndarray] = None,
    ) -> None:
        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)
        action_params_arr = np.stack(
            [np.asarray(a[1], dtype=np.float32).reshape(-1) for a in actions], axis=0
        )
        reward_global = float(np.mean(np.asarray(rewards, dtype=np.float32)))

        track_jammer = bool(getattr(self.buffer, "track_jammer", False))
        jp_kwargs: Dict[str, Any] = {}
        if track_jammer:
            missing = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "JP-aware store_transition requires sensing_history, "
                    f"next_sensing_history and jammer_target; missing: {missing}. "
                    "The training runner uses store_transition_batch to provide them."
                )
            jp_kwargs = {
                "sensing_history": sensing_history,
                "next_sensing_history": next_sensing_history,
                "jammer_target": jammer_target,
            }
        else:
            present = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is not None
            ]
            if present:
                raise ValueError(f"plain store_transition does not accept JP fields: {present}")

        self.buffer.add(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_global,
            next_state=next_state_arr,
            done=bool(done),
            **jp_kwargs,
        )

    def store_transition_batch(
        self,
        *,
        states: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        sensing_history: Optional[np.ndarray] = None,
        next_sensing_history: Optional[np.ndarray] = None,
        jammer_target: Optional[np.ndarray] = None,
    ) -> None:
        """Store a batch of joint transitions into the trainer-owned replay buffer.

        Shapes follow ``JointReplayBuffer.sample`` without the sample dimension removed:
        ``states`` and ``next_states`` are ``(B, N, S)``, ``action_discrete`` is
        ``(B, N)``, ``action_params`` is ``(B, N, A*P)``, and ``dones`` is ``(B,)``.
        ``rewards`` may be ``(B,)`` global rewards or ``(B, N)`` per-agent rewards,
        in which case the mean reward is stored for value-decomposition trainers.

        Stage 8 JP 字段（buffer.track_jammer=True 时必传，否则必为 None）：
        - ``sensing_history``:      (B, N, H, C)
        - ``next_sensing_history``: (B, N, H, C)
        - ``jammer_target``:        (B, C)
        """
        states = np.asarray(states, dtype=np.float32)
        action_discrete = np.asarray(action_discrete, dtype=np.int64)
        action_params = np.asarray(action_params, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if states.ndim != 3 or states.shape[1:] != (self.n_agents, self.state_dim):
            raise ValueError(
                f"states must be (B,{self.n_agents},{self.state_dim}), got {states.shape}"
            )
        batch_size = states.shape[0]
        if next_states.shape != states.shape:
            raise ValueError(f"next_states must match states shape {states.shape}, got {next_states.shape}")
        if action_discrete.shape != (batch_size, self.n_agents):
            raise ValueError(
                f"action_discrete must be ({batch_size},{self.n_agents}), got {action_discrete.shape}"
            )
        if action_params.shape != (batch_size, self.n_agents, self.n_actions * self.param_dim):
            raise ValueError(
                "action_params must be "
                f"({batch_size},{self.n_agents},{self.n_actions * self.param_dim}), "
                f"got {action_params.shape}"
            )
        if dones.shape != (batch_size,):
            raise ValueError(f"dones must be ({batch_size},), got {dones.shape}")

        if rewards.shape == (batch_size, self.n_agents):
            reward_global = rewards.mean(axis=1)
        elif rewards.shape == (batch_size,):
            reward_global = rewards
        else:
            raise ValueError(
                f"rewards must be ({batch_size},) or ({batch_size},{self.n_agents}), got {rewards.shape}"
            )

        # Stage 8 JP 字段双向 strict 校验
        track_jammer = bool(getattr(self.buffer, "track_jammer", False))
        if track_jammer:
            missing = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"JP-aware buffer requires {missing}; pass them to store_transition_batch"
                )
            sensing_history = np.asarray(sensing_history, dtype=np.float32)
            next_sensing_history = np.asarray(next_sensing_history, dtype=np.float32)
            jammer_target = np.asarray(jammer_target, dtype=np.float32)
            if sensing_history.shape[0] != batch_size or next_sensing_history.shape[0] != batch_size:
                raise ValueError(
                    f"sensing_history batch dim must be {batch_size}, "
                    f"got {sensing_history.shape} / {next_sensing_history.shape}"
                )
            if jammer_target.shape[0] != batch_size:
                raise ValueError(
                    f"jammer_target batch dim must be {batch_size}, got {jammer_target.shape}"
                )
        else:
            present = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is not None
            ]
            if present:
                raise ValueError(
                    f"plain buffer does not accept JP fields, got: {present}"
                )

        for i in range(batch_size):
            kwargs: Dict[str, Any] = dict(
                state=states[i],
                action_discrete=action_discrete[i],
                action_params=action_params[i],
                reward=float(reward_global[i]),
                next_state=next_states[i],
                done=bool(dones[i]),
            )
            if track_jammer:
                kwargs["sensing_history"] = sensing_history[i]
                kwargs["next_sensing_history"] = next_sensing_history[i]
                kwargs["jammer_target"] = jammer_target[i]
            self.buffer.add(**kwargs)

    def _critic_step(
        self,
        *,
        state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        global_state: torch.Tensor,
        next_global_state: torch.Tensor,
        target_context: TDTargetContext | None = None,
        sensing_history: Optional[torch.Tensor] = None,
        next_sensing_history: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        hook_context: Any = None,
    ) -> torch.Tensor:
        if self.mixer_opt is not None:
            self.mixer_opt.zero_grad(set_to_none=True)
        for agent in self.agents:
            agent.q_opt.zero_grad(set_to_none=True)
            jp_opt = getattr(agent, "jammer_predictor_opt", None)
            if jp_opt is not None:
                jp_opt.zero_grad(set_to_none=True)

        with self._autocast():
            q_sa_list: List[torch.Tensor] = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                a_i = action_discrete[:, i].view(-1, 1)
                params_i = action_params[:, i, :].view(-1, self.n_actions, self.param_dim)
                h_i = None if sensing_history is None else sensing_history[:, i]
                aug_i = self._maybe_augment_state(self.agents[i], s_i, h_i, target=False)
                q_all_i = self.agents[i].q_net(aug_i, params_i)
                q_sa_list.append(q_all_i.gather(1, a_i))

            agent_qs = torch.cat(q_sa_list, dim=1)  # (B, N)
            critic_extras = self._collect_critic_extras(state, action_params)
            q_tot = self._mix(agent_qs, global_state, **critic_extras)

            with torch.no_grad():
                next_q_list: List[torch.Tensor] = []
                for i in range(self.n_agents):
                    ns_i = next_state[:, i, :]
                    nh_i = None if next_sensing_history is None else next_sensing_history[:, i]
                    # Online actor for double-DQN: 用 (online) jammer_predictor 增广
                    aug_ns_online_i = self._maybe_augment_state(self.agents[i], ns_i, nh_i, target=False)
                    next_params_eval = self.agents[i].actor(aug_ns_online_i)
                    next_q_eval = self.agents[i].q_net(aug_ns_online_i, next_params_eval)
                    next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                    # Target net path: 用 target_jammer_predictor 增广
                    aug_ns_target_i = self._maybe_augment_state(self.agents[i], ns_i, nh_i, target=True)
                    next_params_target = self.agents[i].target_actor(aug_ns_target_i)
                    next_q_target_all = self.agents[i].target_q_net(aug_ns_target_i, next_params_target)
                    next_q_target = next_q_target_all.gather(1, next_action)
                    next_q_list.append(next_q_target)

                next_agent_qs = torch.cat(next_q_list, dim=1)
                target_extras = self._collect_target_extras(next_state)
                next_q_tot = self._target_mix(next_agent_qs, next_global_state, **target_extras)
                td_target = self._compute_td_target(
                    reward,
                    done,
                    next_q_tot,
                    target_context=target_context,
                )

            loss_td = F.smooth_l1_loss(q_tot, td_target)
            # Stage 8：aux loss 与 critic loss 同 batch + 同 backward。
            aux_loss = self._collect_aux_losses(batch if batch is not None else {}, hook_context)
            loss_q = loss_td + aux_loss

        if not torch.isfinite(loss_q):
            return loss_q

        if self.use_amp:
            self.scaler.scale(loss_q).backward()
            if self.max_grad_norm > 0.0:
                if self.mixer_opt is not None:
                    self.scaler.unscale_(self.mixer_opt)
                for agent in self.agents:
                    self.scaler.unscale_(agent.q_opt)
                    jp_opt = getattr(agent, "jammer_predictor_opt", None)
                    if jp_opt is not None:
                        self.scaler.unscale_(jp_opt)
                if self.mixer_opt is not None:
                    clip_grad_norm(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    clip_grad_norm(agent.q_net.parameters(), self.max_grad_norm)
                    jp_pred = getattr(agent, "jammer_predictor", None)
                    if jp_pred is not None:
                        clip_grad_norm(jp_pred.parameters(), self.max_grad_norm)
            if self.mixer_opt is not None:
                self.scaler.step(self.mixer_opt)
            for agent in self.agents:
                self.scaler.step(agent.q_opt)
                jp_opt = getattr(agent, "jammer_predictor_opt", None)
                if jp_opt is not None:
                    self.scaler.step(jp_opt)
            # 注意：scaler.update() 不在 critic_step；保留 actor_step 末尾统一 update（与 baseline JP trainer 同 cadence）。
        else:
            loss_q.backward()
            if self.max_grad_norm > 0.0:
                if self.mixer_opt is not None:
                    clip_grad_norm(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    clip_grad_norm(agent.q_net.parameters(), self.max_grad_norm)
                    jp_pred = getattr(agent, "jammer_predictor", None)
                    if jp_pred is not None:
                        clip_grad_norm(jp_pred.parameters(), self.max_grad_norm)
            if self.mixer_opt is not None:
                self.mixer_opt.step()
            for agent in self.agents:
                agent.q_opt.step()
                jp_opt = getattr(agent, "jammer_predictor_opt", None)
                if jp_opt is not None:
                    jp_opt.step()

        return loss_q

    def _actor_step(
        self,
        *,
        state: torch.Tensor,
        global_state: torch.Tensor,
        sensing_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

        try:
            for agent in self.agents:
                agent.actor_opt.zero_grad(set_to_none=True)

            with self._autocast():
                params_pred_list: List[torch.Tensor] = []
                q_pred_all_list: List[torch.Tensor] = []
                q_actor_list: List[torch.Tensor] = []
                for i in range(self.n_agents):
                    s_i = state[:, i, :]
                    h_i = None if sensing_history is None else sensing_history[:, i]
                    aug_i = self._maybe_augment_state(self.agents[i], s_i, h_i, target=False)
                    params_pred = self.agents[i].actor(aug_i)
                    q_pred = self.agents[i].q_net(aug_i, params_pred)
                    params_pred_list.append(params_pred)
                    q_pred_all_list.append(q_pred)
                    q_actor_list.append(q_pred.mean(dim=1, keepdim=True))

                agent_qs_actor = torch.cat(q_actor_list, dim=1)
                actor_extras = self._collect_actor_extras(state, params_pred_list, q_pred_all_list)
                q_tot_actor = self._mix(agent_qs_actor, global_state, **actor_extras)
                loss_actor_total = -q_tot_actor.mean()

            if not torch.isfinite(loss_actor_total):
                if self.use_amp:
                    self.scaler.update()
                return loss_actor_total

            if self.use_amp:
                self.scaler.scale(loss_actor_total).backward()
                if self.max_grad_norm > 0.0:
                    for agent in self.agents:
                        self.scaler.unscale_(agent.actor_opt)
                        clip_grad_norm(agent.actor.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    self.scaler.step(agent.actor_opt)
            else:
                loss_actor_total.backward()
                if self.max_grad_norm > 0.0:
                    for agent in self.agents:
                        clip_grad_norm(agent.actor.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    agent.actor_opt.step()

            if self.use_amp:
                self.scaler.update()
            return loss_actor_total
        finally:
            for agent in self.agents:
                for p in agent.q_net.parameters():
                    p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True

    def _target_sync(self) -> None:
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            for agent in self.agents:
                hard_sync_target(agent.target_actor, agent.actor)
                hard_sync_target(agent.target_q_net, agent.q_net)
                jp_pred = getattr(agent, "jammer_predictor", None)
                jp_target = getattr(agent, "target_jammer_predictor", None)
                if jp_pred is not None and jp_target is not None:
                    hard_sync_target(jp_target, jp_pred)
            hard_sync_target(self.target_mixer, self.mixer)

    def train_step_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        target_context: TDTargetContext | None = None,
        *,
        hook_context: Any = None,
    ) -> Optional[Dict[str, float]]:
        state = batch["state"]
        action_discrete = batch["action_discrete"].long()
        action_params = batch["action_params"]
        reward = batch["reward"].view(-1, 1)
        next_state = batch["next_state"]
        done = batch["done"].view(-1, 1)
        sensing_history = batch.get("sensing_history")
        next_sensing_history = batch.get("next_sensing_history")

        global_state = state.reshape(state.shape[0], -1)
        next_global_state = next_state.reshape(next_state.shape[0], -1)

        loss_q = self._critic_step(
            state=state,
            action_discrete=action_discrete,
            action_params=action_params,
            reward=reward,
            next_state=next_state,
            done=done,
            global_state=global_state,
            next_global_state=next_global_state,
            target_context=target_context,
            sensing_history=sensing_history,
            next_sensing_history=next_sensing_history,
            batch=batch,
            hook_context=hook_context,
        )
        if not torch.isfinite(loss_q):
            result = {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": 1.0}
            result.update(self._last_aux_loss_result())
            return result

        loss_actor_total = self._actor_step(
            state=state,
            global_state=global_state,
            sensing_history=sensing_history,
        )
        if not torch.isfinite(loss_actor_total):
            result = {
                "loss_q": float(loss_q.item()),
                "loss_actor": float("nan"),
                "skipped": 1.0,
            }
            result.update(self._last_aux_loss_result())
            return result

        self._target_sync()

        result = {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor_total.item()),
        }
        result.update(self._last_aux_loss_result())
        return result

    def _last_aux_loss_result(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in self._last_aux_loss_values.items():
            out[key] = float(value.item())
        return out

    def train_step(self, *, hook_context: Any = None) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        batch_np = self.buffer.sample(self.batch_size)
        batch: Dict[str, torch.Tensor] = {
            "state": torch.from_numpy(batch_np["state"]).to(self.device),
            "action_discrete": torch.from_numpy(batch_np["action_discrete"]).long().to(self.device),
            "action_params": torch.from_numpy(batch_np["action_params"]).to(self.device),
            "reward": torch.from_numpy(batch_np["reward"]).to(self.device).view(-1, 1),
            "next_state": torch.from_numpy(batch_np["next_state"]).to(self.device),
            "done": torch.from_numpy(batch_np["done"]).to(self.device).view(-1, 1),
        }
        # Stage 8：JP buffer 字段透传
        for key in ("sensing_history", "next_sensing_history", "jammer_target"):
            if key in batch_np:
                batch[key] = torch.from_numpy(batch_np[key]).to(self.device)
        return self.train_step_from_batch(batch, target_context=None, hook_context=hook_context)


__all__ = ["TDTargetContext", "ValueDecompTrainerBase"]
