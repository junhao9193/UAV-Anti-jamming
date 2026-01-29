from __future__ import division

from typing import List, Optional, Tuple

import numpy as np
import torch

from algorithms.mpdqn.agent import MPDQNAgent
from algorithms.mpdqn.joint_replay_buffer_iql import MPDQNJointIQLReplayBuffer


class MPDQNJointIQLTrainer:
    """
    IQL training with a *joint* replay buffer (pipeline optimization only).

    Key point: each agent is still updated using ONLY its own local data:
      (o_i, a_i, r_i, o'_i), no team reward, no global state, no mixer.

    Compared to classic IQL with per-agent replay buffers, this trainer:
    - stores one transition per env step (instead of N stores)
    - samples one joint batch, moves it to GPU once, then slices per-agent
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        target_update_interval: int = 200,
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
        device: Optional[str] = None,
    ):
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.agents = [
            MPDQNAgent(
                state_dim=self.state_dim,
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                buffer_capacity=1,  # unused (joint buffer is used)
                batch_size=int(batch_size),
                gamma=float(gamma),
                lr_actor=float(lr_actor),
                lr_q=float(lr_q),
                target_update_interval=int(target_update_interval),
                use_amp=bool(use_amp),
                max_grad_norm=float(max_grad_norm),
                device=str(self.device),
            )
            for _ in range(self.n_agents)
        ]

        self.batch_size = int(batch_size)
        self.buffer = MPDQNJointIQLReplayBuffer(capacity=int(buffer_capacity))

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

        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)  # (N,S)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)  # (N,S)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)  # (N,)
        action_params_arr = np.stack([np.asarray(a[1], dtype=np.float32).reshape(-1) for a in actions], axis=0)  # (N,A*P)
        reward_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)  # (N,)
        if reward_arr.shape[0] != self.n_agents:
            raise ValueError(f"rewards must have shape ({self.n_agents},), got {reward_arr.shape}")

        self.buffer.add(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_arr,
            next_state=next_state_arr,
            done=bool(done),
        )

    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        state = torch.from_numpy(batch["state"]).to(self.device)  # (B,N,S)
        action_discrete = torch.from_numpy(batch["action_discrete"]).long().to(self.device)  # (B,N)
        action_params = torch.from_numpy(batch["action_params"]).to(self.device)  # (B,N,A*P)
        reward = torch.from_numpy(batch["reward"]).to(self.device)  # (B,N)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device)  # (B,N,S)
        done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)  # (B,1)

        loss_q_list = []
        loss_a_list = []
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


__all__ = ["MPDQNJointIQLTrainer"]

