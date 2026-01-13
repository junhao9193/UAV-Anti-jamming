from __future__ import division

import copy
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mpdqn.model import MPDQNActor, MPDQNQNetwork
from algorithms.mpdqn.replay_buffer import MPDQNReplayBuffer


class MPDQNAgent:
    """
    Independent MP-DQN agent for a parameterized action space:
      action = (discrete_action, continuous_params_for_all_actions)
    where continuous params are normalized to [0, 1] and environment applies physical scaling.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        target_update_interval: int = 200,
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

        self.actor = MPDQNActor(self.state_dim, self.n_actions, self.param_dim).to(self.device)
        self.q_net = MPDQNQNetwork(self.state_dim, self.n_actions, self.param_dim).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=float(lr_q))

        self.buffer = MPDQNReplayBuffer(capacity=int(buffer_capacity))
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            params_all = self.actor(state_t)  # (1, n_actions, param_dim) in [0,1]
            q_values = self.q_net(state_t, params_all)  # (1, n_actions)

        if random.random() < float(epsilon):
            action_discrete = random.randrange(self.n_actions)
        else:
            action_discrete = int(torch.argmax(q_values, dim=1).item())

        params_flat = params_all.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action_discrete, params_flat

    def store(
        self,
        state: np.ndarray,
        action_discrete: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        self.buffer.add(
            state=np.asarray(state, dtype=np.float32),
            action_discrete=int(action_discrete),
            action_params=np.asarray(action_params, dtype=np.float32).reshape(-1),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )

    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        state = torch.from_numpy(batch["state"]).to(self.device)
        action_discrete = torch.from_numpy(batch["action_discrete"]).long().to(self.device).view(-1, 1)
        action_params = torch.from_numpy(batch["action_params"]).to(self.device).view(-1, self.n_actions, self.param_dim)
        reward = torch.from_numpy(batch["reward"]).to(self.device).view(-1, 1)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device)
        done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)

        # --- Q update (Double DQN + MP-DQN multi-pass) ---
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
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # --- Actor update ---
        for p in self.q_net.parameters():
            p.requires_grad = False

        params_pred = self.actor(state)
        q_pred = self.q_net(state, params_pred)
        loss_actor = -q_pred.mean()

        self.actor_opt.zero_grad()
        loss_actor.backward()
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
