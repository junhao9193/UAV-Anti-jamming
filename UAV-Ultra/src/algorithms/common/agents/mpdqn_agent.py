"""MPDQN agent（baseline `algorithms/mpdqn/agent.py` 迁移；剥离内部 buffer）。

Stage 4 plan locked decision #5：``MPDQNAgent`` **只持网络 + optimizer + target**，
不携 `self.buffer`，所有 replay 由 trainer 持有（IQL `JointReplayBuffer(per_agent_reward=True)`
/ VDN/QMIX/QPLEX `JointReplayBuffer(per_agent_reward=False)`）。

仅保留 baseline 的 ``train_step_from_tensors(s, a, r, ns, done)`` 接口供 IQL 走 N 次。
``store / sample / train_step``（从 buffer 采样的版本）**不迁**。
"""

from __future__ import annotations

import copy
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp as torch_amp

from src.algorithms.common.networks.mpdqn import MPDQNActor, MPDQNQNetwork


class MPDQNAgent:
    """单 agent 视图：actor + q_net + target_actor + target_q_net + 两个 optimizer。

    所属 trainer 决定如何调度 N 个 agent + replay buffer。本类不持 buffer。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        target_update_interval: int = 200,
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
        device: Optional[str] = None,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.total_param_dim = self.n_actions * self.param_dim

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

        self.actor = MPDQNActor(self.state_dim, self.n_actions, self.param_dim).to(self.device)
        self.q_net = MPDQNQNetwork(self.state_dim, self.n_actions, self.param_dim).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=float(lr_q))

        # plan locked #5：本类**不**持 replay。baseline 中的 `self.buffer` 已剥离。
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        """单条 state → (discrete_action, params_flat)。"""
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            params_all = self.actor(state_t)  # (1, n_actions, param_dim)
            q_values = self.q_net(state_t, params_all)  # (1, n_actions)

        if random.random() < float(epsilon):
            action_discrete = random.randrange(self.n_actions)
        else:
            action_discrete = int(torch.argmax(q_values, dim=1).item())

        params_flat = params_all.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action_discrete, params_flat

    def select_action_batch(
        self, states: np.ndarray, epsilon: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量动作选择，与 baseline 等价。"""
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must be (B,{self.state_dim}), got shape={states.shape}")

        state_t = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            params_all = self.actor(state_t)
            q_values = self.q_net(state_t, params_all)

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

    def train_step_from_tensors(
        self,
        *,
        state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> Optional[dict]:
        """单 agent 一步训练，与 baseline ``train_step_from_tensors`` 等价。

        IQL trainer 持 N 个 ``MPDQNAgent``，对每个 agent 调本函数。形状契约：
        - state:           (B, state_dim)
        - action_discrete: (B, 1)  int64
        - action_params:   (B, n_actions, param_dim)
        - reward:          (B, 1)
        - next_state:      (B, state_dim)
        - done:            (B, 1)
        """
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must be (B,{self.state_dim}), got {tuple(state.shape)}")
        if next_state.ndim != 2 or next_state.shape[1] != self.state_dim:
            raise ValueError(f"next_state must be (B,{self.state_dim}), got {tuple(next_state.shape)}")
        if action_discrete.ndim != 2 or action_discrete.shape[1] != 1:
            raise ValueError(f"action_discrete must be (B,1), got {tuple(action_discrete.shape)}")
        if (
            action_params.ndim != 3
            or action_params.shape[1] != self.n_actions
            or action_params.shape[2] != self.param_dim
        ):
            raise ValueError(
                f"action_params must be (B,{self.n_actions},{self.param_dim}), got {tuple(action_params.shape)}"
            )
        if reward.ndim != 2 or reward.shape[1] != 1:
            raise ValueError(f"reward must be (B,1), got {tuple(reward.shape)}")
        if done.ndim != 2 or done.shape[1] != 1:
            raise ValueError(f"done must be (B,1), got {tuple(done.shape)}")

        def _autocast():
            return torch_amp.autocast("cuda", enabled=self.use_amp)

        # --- Q update (Double DQN + MP-DQN multi-pass) ---
        self.q_opt.zero_grad(set_to_none=True)
        with _autocast():
            q_all = self.q_net(state, action_params)
            q_sa = q_all.gather(1, action_discrete)

            with torch.no_grad():
                next_params_eval = self.actor(next_state)
                next_q_eval = self.q_net(next_state, next_params_eval)
                next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                next_params_target = self.target_actor(next_state)
                next_q_target_all = self.target_q_net(next_state, next_params_target)
                next_q_target = next_q_target_all.gather(1, next_action)

                td_target = reward + (1.0 - done) * self.gamma * next_q_target

            loss_q = F.mse_loss(q_sa, td_target)

        if not torch.isfinite(loss_q):
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_q).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.q_opt)
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.scaler.step(self.q_opt)
        else:
            loss_q.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_opt.step()

        # --- Actor update ---
        for p in self.q_net.parameters():
            p.requires_grad = False

        self.actor_opt.zero_grad(set_to_none=True)
        with _autocast():
            params_pred = self.actor(state)
            q_pred = self.q_net(state, params_pred)
            loss_actor = -q_pred.mean()

        if not torch.isfinite(loss_actor):
            for p in self.q_net.parameters():
                p.requires_grad = True
            if self.use_amp:
                self.scaler.update()
            return {"loss_q": float(loss_q.item()), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_actor).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.actor_opt)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            loss_actor.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

        for p in self.q_net.parameters():
            p.requires_grad = True

        # --- Target updates ---
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor.item()),
        }


__all__ = ["MPDQNAgent"]
