from __future__ import division

import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mpdqn.agent import MPDQNAgent
from algorithms.mpdqn.joint_replay_buffer import MPDQNJointReplayBuffer
from algorithms.mpdqn.qmix_mixer import QMIXMixer


class MPDQNQMIXTrainer:
    """
    Cooperative (global) MP-DQN via QMIX:
    - store joint transitions (all agents)
    - use team reward (mean over agents)
    - train per-agent Q networks jointly through a QMIX mixing network using global_state

    Execution is decentralized: each agent selects its own (discrete channel assignment, continuous power params)
    from local observation only.
    """

    def __init__(
        self,
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
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
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

        if lr_mixer is None:
            lr_mixer = float(lr_q)

        self.agents = [
            MPDQNAgent(
                state_dim=self.state_dim,
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                buffer_capacity=1,  # unused (joint buffer is used)
                batch_size=self.batch_size,
                gamma=self.gamma,
                lr_actor=lr_actor,
                lr_q=lr_q,
                target_update_interval=self.target_update_interval,
                device=str(self.device),
            )
            for _ in range(self.n_agents)
        ]

        self.mixer = QMIXMixer(
            n_agents=self.n_agents,
            global_state_dim=self.global_state_dim,
            mixing_hidden_dim=mixing_hidden_dim,
            hypernet_hidden_dim=hypernet_hidden_dim,
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.mixer_opt = torch.optim.Adam(self.mixer.parameters(), lr=float(lr_mixer))

        self.buffer = MPDQNJointReplayBuffer(capacity=int(buffer_capacity))
        self.learn_steps = 0

    def select_actions(self, states: List[np.ndarray], epsilon: float) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")

        actions: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            action_discrete, action_params = self.agents[i].select_action(states[i], epsilon)
            actions.append((int(action_discrete), np.asarray(action_params, dtype=np.float32)))
        return actions

    def store_transition(
        self,
        states: List[np.ndarray],
        actions: List[Tuple[int, np.ndarray]],
        rewards: np.ndarray,
        next_states: List[np.ndarray],
        done: bool = False,
    ) -> None:
        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)  # (N,S)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)  # (N,S)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)  # (N,)
        action_params_arr = np.stack([np.asarray(a[1], dtype=np.float32).reshape(-1) for a in actions], axis=0)
        reward_global = float(np.mean(np.asarray(rewards, dtype=np.float32)))

        self.buffer.add(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_global,
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
        reward = torch.from_numpy(batch["reward"]).to(self.device).view(-1, 1)  # (B,1)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device)  # (B,N,S)
        done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)  # (B,1)

        global_state = state.reshape(state.shape[0], -1)  # (B, N*S)
        next_global_state = next_state.reshape(next_state.shape[0], -1)  # (B, N*S)

        # --- Q update (Double DQN per-agent + QMIX mixing) ---
        q_sa_list = []
        for i in range(self.n_agents):
            s_i = state[:, i, :]
            a_i = action_discrete[:, i].view(-1, 1)
            params_i = action_params[:, i, :].view(-1, self.n_actions, self.param_dim)
            q_all_i = self.agents[i].q_net(s_i, params_i)
            q_sa_i = q_all_i.gather(1, a_i)  # (B,1)
            q_sa_list.append(q_sa_i)

        agent_qs = torch.cat(q_sa_list, dim=1)  # (B,N)
        q_tot = self.mixer(agent_qs, global_state)  # (B,1)

        with torch.no_grad():
            next_q_list = []
            for i in range(self.n_agents):
                ns_i = next_state[:, i, :]
                next_params_eval = self.agents[i].actor(ns_i)
                next_q_eval = self.agents[i].q_net(ns_i, next_params_eval)
                next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                next_params_target = self.agents[i].target_actor(ns_i)
                next_q_target_all = self.agents[i].target_q_net(ns_i, next_params_target)
                next_q_target = next_q_target_all.gather(1, next_action)
                next_q_list.append(next_q_target)

            next_agent_qs = torch.cat(next_q_list, dim=1)  # (B,N)
            next_q_tot = self.target_mixer(next_agent_qs, next_global_state)  # (B,1)
            td_target = reward + (1.0 - done) * self.gamma * next_q_tot

        loss_q = F.mse_loss(q_tot, td_target)

        self.mixer_opt.zero_grad()
        for agent in self.agents:
            agent.q_opt.zero_grad()
        loss_q.backward()
        self.mixer_opt.step()
        for agent in self.agents:
            agent.q_opt.step()

        # --- Actor updates (keep original MP-DQN objective: maximize mean Q over all discrete actions) ---
        actor_losses = []
        for i in range(self.n_agents):
            agent = self.agents[i]
            for p in agent.q_net.parameters():
                p.requires_grad = False

            s_i = state[:, i, :]
            params_pred = agent.actor(s_i)
            q_pred = agent.q_net(s_i, params_pred)
            loss_actor = -q_pred.mean()

            agent.actor_opt.zero_grad()
            loss_actor.backward()
            agent.actor_opt.step()

            for p in agent.q_net.parameters():
                p.requires_grad = True

            actor_losses.append(float(loss_actor.item()))

        # --- Target updates ---
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            for agent in self.agents:
                agent.target_actor.load_state_dict(agent.actor.state_dict())
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(np.mean(actor_losses)) if actor_losses else 0.0,
        }


__all__ = ["MPDQNQMIXTrainer"]

