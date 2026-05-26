"""IQL trainer：N 个独立 ``MPDQNAgent`` + 共享 ``JointReplayBuffer(per_agent_reward=True)``。

与 baseline ``algorithms/mpdqn/iql/trainer.py:10-145`` 等价：
- 每个 agent 仅用自己本地数据 ``(o_i, a_i, r_i, o'_i, done)`` 更新。
- 无 mixer，无 global state，无团队 reward。
- 共用 joint buffer 是流水线优化（一次 store / 一次 sample，按 agent 切片）。

签名遵循 Stage 4 ``build_trainer`` 契约：``IQLTrainer(*, env_cfg, algo_cfg, device=None, ...)``。
``epsilon_*`` 字段由 Stage 5 Runner 控制 ``select_actions(epsilon=...)``，本类不存。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent
from src.algorithms.common.buffers.joint_replay import JointReplayBuffer
from src.config import specs
from src.config.schema import EnvConfig, IQLConfig


class IQLTrainer:
    """Independent Q-Learning trainer (MP-DQN agents, joint buffer)."""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: IQLConfig,
        device: Optional[str] = None,
    ):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg

        self.n_agents = int(env_cfg.n_ch)
        self.state_dim = int(specs.state_dim(env_cfg))
        self.n_actions = int(specs.action_dim(env_cfg))
        self.param_dim = int(specs.param_dim_per_action(env_cfg))

        if device is None:
            chosen = algo_cfg.device
            if chosen == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(chosen)
        else:
            self.device = torch.device(device)

        self.batch_size = int(algo_cfg.batch_size)

        self.agents: List[MPDQNAgent] = [
            MPDQNAgent(
                state_dim=self.state_dim,
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                batch_size=self.batch_size,
                gamma=float(algo_cfg.gamma),
                lr_actor=float(algo_cfg.lr_actor),
                lr_q=float(algo_cfg.lr_q),
                target_update_interval=int(algo_cfg.target_update_interval),
                use_amp=bool(algo_cfg.use_amp),
                max_grad_norm=float(algo_cfg.max_grad_norm),
                device=str(self.device),
            )
            for _ in range(self.n_agents)
        ]

        self.buffer = JointReplayBuffer(
            capacity=int(algo_cfg.buffer_capacity),
            per_agent_reward=True,
        )

    # ----------------------- 动作选择 -----------------------

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

    # ----------------------- store transitions -----------------------

    def store_transition(
        self,
        states: List[np.ndarray],
        actions: List[Tuple[int, np.ndarray]],
        rewards: np.ndarray,
        next_states: List[np.ndarray],
        done: bool = False,
    ) -> None:
        if len(states) != self.n_agents or len(next_states) != self.n_agents or len(actions) != self.n_agents:
            raise ValueError("states/actions/next_states must have length == n_agents")

        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)
        action_params_arr = np.stack(
            [np.asarray(a[1], dtype=np.float32).reshape(-1) for a in actions], axis=0
        )
        reward_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if reward_arr.shape[0] != self.n_agents:
            raise ValueError(
                f"rewards must have shape ({self.n_agents},), got {reward_arr.shape}"
            )

        self.buffer.add(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_arr,
            next_state=next_state_arr,
            done=bool(done),
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
    ) -> None:
        """Bulk (B, N, ...) writer for vec-env workflows; rewards must be ``(B, N)``."""
        states = np.asarray(states, dtype=np.float32)
        action_discrete = np.asarray(action_discrete, dtype=np.int64)
        action_params = np.asarray(action_params, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if states.ndim != 3 or states.shape[1:] != (self.n_agents, self.state_dim):
            raise ValueError(f"states must be (B,{self.n_agents},{self.state_dim}), got {states.shape}")
        batch_size = states.shape[0]
        if next_states.shape != states.shape:
            raise ValueError(f"next_states must match states shape, got {next_states.shape}")
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
        if rewards.shape != (batch_size, self.n_agents):
            raise ValueError(
                f"IQL rewards must be ({batch_size},{self.n_agents}), got {rewards.shape}"
            )
        if dones.shape != (batch_size,):
            raise ValueError(f"dones must be ({batch_size},), got {dones.shape}")

        for i in range(batch_size):
            self.buffer.add(
                state=states[i],
                action_discrete=action_discrete[i],
                action_params=action_params[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=bool(dones[i]),
            )

    # ----------------------- 训练步 -----------------------

    def train_step(self, *, hook_context: object = None) -> Optional[dict]:
        # Stage 8：IQL 不参与 aux loss（hook_context ignored）；接受 kwarg 仅为统一 callback dispatch 接口。
        del hook_context
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(batch["state"]).to(self.device)
        action_discrete = torch.from_numpy(batch["action_discrete"]).long().to(self.device)
        action_params = torch.from_numpy(batch["action_params"]).to(self.device)
        reward = torch.from_numpy(batch["reward"]).to(self.device)  # (B, N)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device)
        done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)

        loss_q_list: List[float] = []
        loss_a_list: List[float] = []
        skipped = 0

        for i in range(self.n_agents):
            s_i = state[:, i, :]
            a_i = action_discrete[:, i].view(-1, 1)
            params_i = action_params[:, i, :].view(-1, self.n_actions, self.param_dim)
            r_i = reward[:, i].view(-1, 1)
            ns_i = next_state[:, i, :]

            out = self.agents[i].train_step_from_tensors(
                state=s_i,
                action_discrete=a_i,
                action_params=params_i,
                reward=r_i,
                next_state=ns_i,
                done=done,
            )
            if out is None:
                continue
            if int(out.get("skipped", 0)) > 0:
                skipped += 1
                continue
            loss_q_list.append(float(out["loss_q"]))
            loss_a_list.append(float(out["loss_actor"]))

        if not loss_q_list:
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": skipped}

        return {
            "loss_q": float(np.mean(loss_q_list)),
            "loss_actor": float(np.mean(loss_a_list)),
            "skipped": skipped,
        }


__all__ = ["IQLTrainer"]
