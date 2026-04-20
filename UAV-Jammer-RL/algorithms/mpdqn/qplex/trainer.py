from __future__ import division

import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mpdqn.agent import MPDQNAgent
from algorithms.mpdqn.qmix.joint_replay_buffer import MPDQNJointReplayBuffer
from algorithms.mpdqn.qplex.mixer import QPLEXMixer

try:
    from torch import amp as torch_amp
except Exception:  # pragma: no cover - older torch
    torch_amp = None


class MPDQNQPLEXTrainer:
    """QPLEX-style cooperative MP-DQN trainer for hybrid discrete-continuous actions."""

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
        n_heads: int = 4,
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
        self.mixing_hidden_dim = int(mixing_hidden_dim)
        self.hypernet_hidden_dim = int(hypernet_hidden_dim)
        self.n_heads = int(n_heads)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        if torch_amp is not None:
            self.scaler = torch_amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_grad_norm = float(max_grad_norm)
        self.value_target_clip = None if value_target_clip is None else float(value_target_clip)

        if lr_mixer is None:
            lr_mixer = float(lr_q)

        self.agents = [
            MPDQNAgent(
                state_dim=self.state_dim,
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                buffer_capacity=1,
                batch_size=self.batch_size,
                gamma=self.gamma,
                lr_actor=lr_actor,
                lr_q=lr_q,
                target_update_interval=self.target_update_interval,
                use_amp=self.use_amp,
                max_grad_norm=self.max_grad_norm,
                device=str(self.device),
            )
            for _ in range(self.n_agents)
        ]

        self.mixer = QPLEXMixer(
            n_agents=self.n_agents,
            global_state_dim=self.global_state_dim,
            mixing_hidden_dim=self.mixing_hidden_dim,
            hypernet_hidden_dim=self.hypernet_hidden_dim,
            n_heads=self.n_heads,
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.mixer_opt = torch.optim.Adam(self.mixer.parameters(), lr=float(lr_mixer))

        self.buffer = MPDQNJointReplayBuffer(capacity=int(buffer_capacity))
        self.learn_steps = 0

    def _clip_value_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_target_clip is None or self.value_target_clip <= 0.0:
            return x
        return torch.clamp(x, min=-self.value_target_clip, max=self.value_target_clip)

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
        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)
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
        state = torch.from_numpy(batch["state"]).to(self.device)
        action_discrete = torch.from_numpy(batch["action_discrete"]).long().to(self.device)
        action_params = torch.from_numpy(batch["action_params"]).to(self.device)
        reward = torch.from_numpy(batch["reward"]).to(self.device).view(-1, 1)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device)
        done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)

        global_state = state.reshape(state.shape[0], -1)
        next_global_state = next_state.reshape(next_state.shape[0], -1)

        def _autocast():
            return (
                torch_amp.autocast("cuda", enabled=self.use_amp)
                if torch_amp is not None
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )

        self.mixer_opt.zero_grad(set_to_none=True)
        for agent in self.agents:
            agent.q_opt.zero_grad(set_to_none=True)

        with _autocast():
            q_sa_list = []
            q_max_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                a_i = action_discrete[:, i].view(-1, 1)
                params_i = action_params[:, i, :].view(-1, self.n_actions, self.param_dim)
                q_all_i = self.agents[i].q_net(s_i, params_i)
                q_sa_i = q_all_i.gather(1, a_i)
                q_sa_list.append(q_sa_i)

                with torch.no_grad():
                    greedy_params_i = self.agents[i].actor(s_i)
                q_greedy_all_i = self.agents[i].q_net(s_i, greedy_params_i)
                q_max_i = q_greedy_all_i.max(dim=1, keepdim=True).values
                q_max_list.append(q_max_i)

            agent_qs = torch.cat(q_sa_list, dim=1)
            max_agent_qs = torch.cat(q_max_list, dim=1)
            q_tot = self.mixer(agent_qs, max_agent_qs, global_state)

            with torch.no_grad():
                next_q_list = []
                next_q_max_list = []
                for i in range(self.n_agents):
                    ns_i = next_state[:, i, :]
                    next_params_eval = self.agents[i].actor(ns_i)
                    next_q_eval = self.agents[i].q_net(ns_i, next_params_eval)
                    next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                    next_params_target = self.agents[i].target_actor(ns_i)
                    next_q_target_all = self.agents[i].target_q_net(ns_i, next_params_target)
                    next_q_target = next_q_target_all.gather(1, next_action)
                    next_q_target_max = next_q_target_all.max(dim=1, keepdim=True).values
                    next_q_list.append(next_q_target)
                    next_q_max_list.append(next_q_target_max)

                next_agent_qs = torch.cat(next_q_list, dim=1)
                next_max_agent_qs = torch.cat(next_q_max_list, dim=1)
                next_q_tot = self.target_mixer(next_agent_qs, next_max_agent_qs, next_global_state)
                td_target = reward + (1.0 - done) * self.gamma * next_q_tot
                td_target = self._clip_value_target(td_target)

            loss_q = F.smooth_l1_loss(q_tot, td_target)

        if not torch.isfinite(loss_q):
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_q).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.mixer_opt)
                for agent in self.agents:
                    self.scaler.unscale_(agent.q_opt)
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), self.max_grad_norm)
            self.scaler.step(self.mixer_opt)
            for agent in self.agents:
                self.scaler.step(agent.q_opt)
        else:
            loss_q.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), self.max_grad_norm)
            self.mixer_opt.step()
            for agent in self.agents:
                agent.q_opt.step()

        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = False
            agent.actor_opt.zero_grad(set_to_none=True)
        for p in self.mixer.parameters():
            p.requires_grad = False

        with _autocast():
            q_actor_list = []
            q_actor_max_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                params_pred = self.agents[i].actor(s_i)
                q_pred = self.agents[i].q_net(s_i, params_pred)
                # Keep the QPLEX advantage branch active during actor updates.
                # `q_policy` follows the same MP-DQN surrogate used by QMIX
                # (maximize the mean utility under actor-produced parameters),
                # while `q_baseline` is a detached greedy utility reference.
                # Do not pass identical tensors here: that would make
                # agent_qs - max_agent_qs == 0 and erase the advantage term.
                q_policy = q_pred.mean(dim=1, keepdim=True)
                q_baseline = q_pred.max(dim=1, keepdim=True).values.detach()
                q_actor_list.append(q_policy)
                q_actor_max_list.append(q_baseline)

            agent_qs_actor = torch.cat(q_actor_list, dim=1)
            max_agent_qs_actor = torch.cat(q_actor_max_list, dim=1)
            q_tot_actor = self.mixer(agent_qs_actor, max_agent_qs_actor, global_state)
            loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            for agent in self.agents:
                for p in agent.q_net.parameters():
                    p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
            if self.use_amp:
                self.scaler.update()
            return {"loss_q": float(loss_q.item()), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_actor_total).backward()
            if self.max_grad_norm > 0.0:
                for agent in self.agents:
                    self.scaler.unscale_(agent.actor_opt)
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
            for agent in self.agents:
                self.scaler.step(agent.actor_opt)
        else:
            loss_actor_total.backward()
            if self.max_grad_norm > 0.0:
                for agent in self.agents:
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
            for agent in self.agents:
                agent.actor_opt.step()

        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = True
        for p in self.mixer.parameters():
            p.requires_grad = True

        if self.use_amp:
            self.scaler.update()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            for agent in self.agents:
                agent.target_actor.load_state_dict(agent.actor.state_dict())
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor_total.item()),
        }


__all__ = ["MPDQNQPLEXTrainer"]
