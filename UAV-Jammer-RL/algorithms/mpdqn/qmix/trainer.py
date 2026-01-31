from __future__ import division

import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mpdqn.agent import MPDQNAgent
from algorithms.mpdqn.qmix.joint_replay_buffer import MPDQNJointReplayBuffer
from algorithms.mpdqn.qmix.mixer import QMIXMixer

try:
    from torch import amp as torch_amp
except Exception:  # pragma: no cover - older torch
    torch_amp = None


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
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
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
        if torch_amp is not None:
            self.scaler = torch_amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_grad_norm = float(max_grad_norm)

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
                use_amp=self.use_amp,
                max_grad_norm=self.max_grad_norm,
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

        def _autocast():
            return (
                torch_amp.autocast("cuda", enabled=self.use_amp)
                if torch_amp is not None
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )

        # --- Q update (Double DQN per-agent + QMIX mixing) ---
        self.mixer_opt.zero_grad(set_to_none=True)
        for agent in self.agents:
            agent.q_opt.zero_grad(set_to_none=True)

        with _autocast():
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

        # --- Actor updates (team objective): maximize Q_tot predicted by mixer ---
        # Freeze critic (per-agent Q networks + mixer) during actor update.
        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

        for agent in self.agents:
            agent.actor_opt.zero_grad(set_to_none=True)

        with _autocast():
            q_mean_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                params_pred = self.agents[i].actor(s_i)  # (B, A, P)
                q_pred = self.agents[i].q_net(s_i, params_pred)  # (B, A)
                q_mean_list.append(q_pred.mean(dim=1, keepdim=True))  # (B,1)

            agent_qs_actor = torch.cat(q_mean_list, dim=1)  # (B, N)
            q_tot_actor = self.mixer(agent_qs_actor, global_state)  # (B,1)
            loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            # Restore grads for next iteration before returning.
            for agent in self.agents:
                for p in agent.q_net.parameters():
                    p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
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

        # Restore critic grads.
        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = True
        for p in self.mixer.parameters():
            p.requires_grad = True

        if self.use_amp:
            self.scaler.update()

        # --- Target updates ---
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

    def train_step_value_expansion(
        self,
        *,
        seq_buffer,
        seq_len: int,
        world_model,
        value_teacher,
        td_cfg,
        alpha_model: float,
        n_channel: int,
        n_des: int,
        power_min_dbm: float | None = None,
        power_max_dbm: float | None = None,
    ) -> Optional[dict]:
        """
        QMIX critic update with Value Expansion (mixed TD target) using a recurrent world model.

          y_real  = r_t + gamma * Q_tot_target(s_{t+1}, u*(s_{t+1}))
          y_model = G_hat^{lambda,K} from the world model rollout
          y       = (1-alpha) * y_real + alpha * y_model

        Important:
        - This function uses stop-gradient for y (computed under no_grad) when updating the critic.
        - `seq_buffer` must provide contiguous sequences sampled by env_id (no reset inside).
        """
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        alpha_model = float(alpha_model)
        if not (0.0 <= alpha_model <= 1.0):
            raise ValueError(f"alpha_model must be in [0,1], got {alpha_model}")

        # Try sampling a batch of sequences; if replay not ready, skip.
        try:
            batch = seq_buffer.sample_sequences(batch_size=int(self.batch_size), seq_len=int(seq_len))
        except Exception:
            return None

        from algorithms.world_model.action_encoding import encode_joint_action_exec
        from algorithms.world_model.value_consistency import rollout_td_lambda_return

        state_seq = torch.from_numpy(batch["state_seq"]).to(self.device).to(torch.float32)  # (B,L,Ds)
        next_state_seq = torch.from_numpy(batch["next_state_seq"]).to(self.device).to(torch.float32)  # (B,L,Ds)
        action_discrete_seq = (
            torch.from_numpy(batch["action_discrete_seq"]).to(self.device).to(torch.long)
        )  # (B,L,N)
        action_params_seq = (
            torch.from_numpy(batch["action_params_seq"]).to(self.device).to(torch.float32)
        )  # (B,L,N,AP)
        reward_seq = torch.from_numpy(batch["reward_seq"]).to(self.device).to(torch.float32)  # (B,L,1)
        done_seq = torch.from_numpy(batch["done_seq"]).to(self.device).to(torch.float32)  # (B,L,1)

        # Use the last step of each sequence as the RL transition.
        global_state = state_seq[:, -1, :]  # (B,Ds)
        next_global_state = next_state_seq[:, -1, :]  # (B,Ds)

        bsz = int(global_state.shape[0])
        state = global_state.view(bsz, int(self.n_agents), int(self.state_dim))  # (B,N,S)
        next_state = next_global_state.view(bsz, int(self.n_agents), int(self.state_dim))  # (B,N,S)

        action_discrete = action_discrete_seq[:, -1, :]  # (B,N)
        action_params = action_params_seq[:, -1, :, :]  # (B,N,AP)
        reward = reward_seq[:, -1, :]  # (B,1)
        done = done_seq[:, -1, :]  # (B,1)

        # Encode joint actions for the whole context sequence (for model rollouts).
        bsz, l, n_agents = action_discrete_seq.shape
        ad_flat = action_discrete_seq.reshape(int(bsz * l), int(n_agents))
        ap_flat = action_params_seq.reshape(int(bsz * l), int(n_agents), -1)
        action_enc_flat = encode_joint_action_exec(
            ad_flat,
            ap_flat,
            n_agents=int(self.n_agents),
            n_channel=int(n_channel),
            n_des=int(n_des),
            n_actions=int(self.n_actions),
            param_dim=int(self.param_dim),
            power_min_dbm=power_min_dbm,
            power_max_dbm=power_max_dbm,
        )
        action_enc_seq = action_enc_flat.view(int(bsz), int(l), -1)  # (B,L,Du)

        def _autocast():
            return (
                torch_amp.autocast("cuda", enabled=self.use_amp)
                if torch_amp is not None
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )

        # --- Q update (Double DQN per-agent + QMIX mixing) ---
        self.mixer_opt.zero_grad(set_to_none=True)
        for agent in self.agents:
            agent.q_opt.zero_grad(set_to_none=True)

        with _autocast():
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
                # y_real
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
                y_real = reward + (1.0 - done) * self.gamma * next_q_tot

                if alpha_model > 0.0:
                    # y_model = model TD(lambda) return (no grad for critic update)
                    def _policy_fn(s_flat: torch.Tensor):
                        a_star, p_star = value_teacher.greedy_action(s_flat)
                        u_star = encode_joint_action_exec(
                            a_star,
                            p_star,
                            n_agents=int(self.n_agents),
                            n_channel=int(n_channel),
                            n_des=int(n_des),
                            n_actions=int(self.n_actions),
                            param_dim=int(self.param_dim),
                            power_min_dbm=power_min_dbm,
                            power_max_dbm=power_max_dbm,
                        )
                        return u_star, a_star, p_star

                    y_model, _ = rollout_td_lambda_return(
                        wm=world_model,
                        state_seq=state_seq,
                        action_seq=action_enc_seq,
                        policy_fn=_policy_fn,
                        q_tot_target_fn=value_teacher.q_tot_target,
                        cfg=td_cfg,
                    )
                    td_target = (1.0 - alpha_model) * y_real + alpha_model * y_model
                else:
                    td_target = y_real

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

        # --- Actor updates (team objective): maximize Q_tot predicted by mixer ---
        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

        for agent in self.agents:
            agent.actor_opt.zero_grad(set_to_none=True)

        with _autocast():
            q_mean_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                params_pred = self.agents[i].actor(s_i)
                q_pred = self.agents[i].q_net(s_i, params_pred)
                q_mean_list.append(q_pred.mean(dim=1, keepdim=True))

            agent_qs_actor = torch.cat(q_mean_list, dim=1)
            q_tot_actor = self.mixer(agent_qs_actor, global_state)
            loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            for agent in self.agents:
                for p in agent.q_net.parameters():
                    p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
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

        # Restore critic grads.
        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = True
        for p in self.mixer.parameters():
            p.requires_grad = True

        if self.use_amp:
            self.scaler.update()

        # --- Target updates ---
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


__all__ = ["MPDQNQMIXTrainer"]
