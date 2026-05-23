import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp as torch_amp

from algorithms.mpdqn.model import BatchedIndependentMPDQNActor, BatchedIndependentMPDQNQNetwork
from algorithms.mpdqn.profiling import profile_section, set_profiler, should_log_loss
from algorithms.mpdqn.qmix.joint_replay_buffer import MPDQNJointReplayBuffer
from algorithms.mpdqn.qmix.mixer import QMIXMixer


def _clip_batched_agent_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros((), dtype=torch.float32)

    n_agents = int(params[0].shape[0])
    device = params[0].grad.device
    total_sq = torch.zeros(n_agents, device=device, dtype=torch.float32)
    for p in params:
        if int(p.shape[0]) != n_agents:
            raise ValueError("All batched parameters must have agent axis first")
        grad = p.grad.detach().float().reshape(n_agents, -1)
        total_sq += grad.pow(2).sum(dim=1)

    total_norm = torch.sqrt(total_sq)
    clip_coef = (float(max_norm) / (total_norm + float(eps))).clamp(max=1.0)
    for p in params:
        view_shape = (n_agents,) + (1,) * (p.grad.ndim - 1)
        p.grad.mul_(clip_coef.to(device=p.grad.device, dtype=p.grad.dtype).view(view_shape))
    return total_norm


class _AgentNetworkView:
    def __init__(self, module, agent_idx: int):
        self.module = module
        self.agent_idx = int(agent_idx)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self.module.agent_state_dict(self.agent_idx)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.module.load_agent_state_dict(self.agent_idx, state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def __call__(self, *args, **kwargs):
        return self.module.forward_agent(self.agent_idx, *args, **kwargs)


class _BatchedQMIXAgentView:
    def __init__(self, trainer: "MPDQNQMIXTrainer", agent_idx: int):
        self.trainer = trainer
        self.agent_idx = int(agent_idx)
        self.state_dim = int(trainer.state_dim)
        self.n_actions = int(trainer.n_actions)
        self.param_dim = int(trainer.param_dim)
        self.total_param_dim = self.n_actions * self.param_dim
        self.actor = _AgentNetworkView(trainer.actor, self.agent_idx)
        self.q_net = _AgentNetworkView(trainer.q_net, self.agent_idx)
        self.target_actor = _AgentNetworkView(trainer.target_actor, self.agent_idx)
        self.target_q_net = _AgentNetworkView(trainer.target_q_net, self.agent_idx)
        self.profiler = None

    def select_action_batch(self, states: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.trainer.select_action_batch_for_agent(self.agent_idx, states, epsilon)

    def select_action(self, state: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        actions, params = self.select_action_batch(np.asarray(state, dtype=np.float32).reshape(1, -1), epsilon)
        return int(actions[0]), params[0]


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
        value_target_clip: Optional[float] = 1000.0,
        loss_log_interval: int = 1,
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
        self.loss_log_interval = int(loss_log_interval)
        self.profiler = None

        if lr_mixer is None:
            lr_mixer = float(lr_q)

        self.actor = BatchedIndependentMPDQNActor(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            param_dim=self.param_dim,
        ).to(self.device)
        self.q_net = BatchedIndependentMPDQNQNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            param_dim=self.param_dim,
        ).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=float(lr_q))
        self.agents = [_BatchedQMIXAgentView(self, i) for i in range(self.n_agents)]

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

    def set_profiler(self, profiler) -> None:
        set_profiler(self, profiler)

    def _clip_value_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_target_clip is None or self.value_target_clip <= 0.0:
            return x
        return torch.clamp(x, min=-self.value_target_clip, max=self.value_target_clip)

    def select_actions(self, states: List[np.ndarray], epsilon: float) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")

        ad, ap = self.select_action_batch_all(np.asarray(states, dtype=np.float32).reshape(1, self.n_agents, -1), epsilon)
        return [(int(ad[0, i]), ap[0, i, :]) for i in range(self.n_agents)]

    def select_action_batch_for_agent(self, agent_idx: int, states: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must be (B,{self.state_dim}), got {states.shape}")

        state_t = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            params_all = self.actor.forward_agent(int(agent_idx), state_t)
            q_values = self.q_net.forward_agent(int(agent_idx), state_t, params_all)

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

    def select_action_batch_all(self, states: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 3 or states.shape[1] != self.n_agents or states.shape[2] != self.state_dim:
            raise ValueError(f"states must be (E,{self.n_agents},{self.state_dim}), got {states.shape}")

        state_t = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            params_all = self.actor(state_t)  # (E,N,A,P)
            q_values = self.q_net(state_t, params_all)  # (E,N,A)

        greedy = torch.argmax(q_values, dim=2).detach().cpu().numpy().astype(np.int32)
        n_envs = int(states.shape[0])
        if float(epsilon) <= 0.0:
            action_discrete = greedy
        elif float(epsilon) >= 1.0:
            action_discrete = np.random.randint(0, self.n_actions, size=(n_envs, self.n_agents), dtype=np.int32)
        else:
            rnd = np.random.randint(0, self.n_actions, size=(n_envs, self.n_agents), dtype=np.int32)
            explore = np.random.random(size=(n_envs, self.n_agents)) < float(epsilon)
            action_discrete = np.where(explore, rnd, greedy).astype(np.int32)

        params_flat = params_all.detach().cpu().numpy().reshape(n_envs, self.n_agents, -1).astype(np.float32)
        return action_discrete, params_flat

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

    def store_transition_batch(
        self,
        states: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        state_arr = np.asarray(states, dtype=np.float32)
        next_state_arr = np.asarray(next_states, dtype=np.float32)
        action_discrete_arr = np.asarray(action_discrete, dtype=np.int64)
        action_params_arr = np.asarray(action_params, dtype=np.float32)
        reward_global = np.mean(np.asarray(rewards, dtype=np.float32), axis=1).astype(np.float32)
        done_arr = np.asarray(dones, dtype=np.float32).reshape(-1)

        self.buffer.add_batch(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_global,
            next_state=next_state_arr,
            done=done_arr,
        )

    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None

        with profile_section(self, "sample_batch"):
            batch = self.buffer.sample(self.batch_size)
        with profile_section(self, "cpu_to_gpu"):
            state = torch.from_numpy(batch["state"]).to(self.device)  # (B,N,S)
            action_discrete = torch.from_numpy(batch["action_discrete"]).long().to(self.device)  # (B,N)
            action_params = torch.from_numpy(batch["action_params"]).to(self.device)  # (B,N,A*P)
            reward = torch.from_numpy(batch["reward"]).to(self.device).view(-1, 1)  # (B,1)
            next_state = torch.from_numpy(batch["next_state"]).to(self.device)  # (B,N,S)
            done = torch.from_numpy(batch["done"]).to(self.device).view(-1, 1)  # (B,1)

        global_state = state.reshape(state.shape[0], -1)  # (B, N*S)
        next_global_state = next_state.reshape(next_state.shape[0], -1)  # (B, N*S)

        def _autocast():
            return torch_amp.autocast("cuda", enabled=self.use_amp)

        # --- Q update (Double DQN per-agent + QMIX mixing) ---
        with profile_section(self, "critic_zero_grad"):
            self.mixer_opt.zero_grad(set_to_none=True)
            self.q_opt.zero_grad(set_to_none=True)

        with profile_section(self, "critic_forward"):
            with _autocast():
                action_params_view = action_params.view(-1, self.n_agents, self.n_actions, self.param_dim)
                q_all = self.q_net(state, action_params_view)  # (B,N,A)
                agent_qs = q_all.gather(2, action_discrete.unsqueeze(-1)).squeeze(-1)  # (B,N)
                q_tot = self.mixer(agent_qs, global_state)  # (B,1)

                with torch.no_grad():
                    next_params_eval = self.actor(next_state)
                    next_q_eval = self.q_net(next_state, next_params_eval)
                    next_action = torch.argmax(next_q_eval, dim=2, keepdim=True)

                    next_params_target = self.target_actor(next_state)
                    next_q_target_all = self.target_q_net(next_state, next_params_target)
                    next_agent_qs = next_q_target_all.gather(2, next_action).squeeze(-1)  # (B,N)
                    next_q_tot = self.target_mixer(next_agent_qs, next_global_state)  # (B,1)
                    td_target = reward + (1.0 - done) * self.gamma * next_q_tot
                    td_target = self._clip_value_target(td_target)

                loss_q = F.smooth_l1_loss(q_tot, td_target)

        if not torch.isfinite(loss_q):
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            with profile_section(self, "critic_backward"):
                self.scaler.scale(loss_q).backward()
            if self.max_grad_norm > 0.0:
                with profile_section(self, "critic_clip_grad"):
                    self.scaler.unscale_(self.mixer_opt)
                    self.scaler.unscale_(self.q_opt)
                    torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                    _clip_batched_agent_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            with profile_section(self, "critic_optimizer_step"):
                self.scaler.step(self.mixer_opt)
                self.scaler.step(self.q_opt)
        else:
            with profile_section(self, "critic_backward"):
                loss_q.backward()
            if self.max_grad_norm > 0.0:
                with profile_section(self, "critic_clip_grad"):
                    torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                    _clip_batched_agent_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            with profile_section(self, "critic_optimizer_step"):
                self.mixer_opt.step()
                self.q_opt.step()

        # --- Actor updates (team objective): maximize Q_tot predicted by mixer ---
        # Freeze critic (per-agent Q networks + mixer) during actor update.
        with profile_section(self, "freeze_critic"):
            for p in self.q_net.parameters():
                p.requires_grad = False
            for p in self.mixer.parameters():
                p.requires_grad = False

        with profile_section(self, "actor_zero_grad"):
            self.actor_opt.zero_grad(set_to_none=True)

        with profile_section(self, "actor_forward"):
            with _autocast():
                params_pred = self.actor(state)  # (B,N,A,P)
                q_pred = self.q_net(state, params_pred)  # (B,N,A)
                # Use the original MP-DQN-style surrogate: maximize the mean Q over
                # all discrete actions under the actor-produced parameters.
                agent_qs_actor = q_pred.mean(dim=2)  # (B,N)
                q_tot_actor = self.mixer(agent_qs_actor, global_state)  # (B,1)
                loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            # Restore grads for next iteration before returning.
            for p in self.q_net.parameters():
                p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
            # Q optimizers may already have stepped under AMP above. Finalize the
            # scaler state so the next iteration can safely call unscale_ again.
            if self.use_amp:
                self.scaler.update()
            return {"loss_q": float(loss_q.item()), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            with profile_section(self, "actor_backward"):
                self.scaler.scale(loss_actor_total).backward()
            if self.max_grad_norm > 0.0:
                with profile_section(self, "actor_clip_grad"):
                    self.scaler.unscale_(self.actor_opt)
                    _clip_batched_agent_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            with profile_section(self, "actor_optimizer_step"):
                self.scaler.step(self.actor_opt)
        else:
            with profile_section(self, "actor_backward"):
                loss_actor_total.backward()
            if self.max_grad_norm > 0.0:
                with profile_section(self, "actor_clip_grad"):
                    _clip_batched_agent_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            with profile_section(self, "actor_optimizer_step"):
                self.actor_opt.step()

        # Restore critic grads.
        with profile_section(self, "restore_critic_grad"):
            for p in self.q_net.parameters():
                p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True

        if self.use_amp:
            with profile_section(self, "scaler_update"):
                self.scaler.update()

        # --- Target updates ---
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            with profile_section(self, "target_update"):
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())

        loss_q_value = None
        loss_actor_value = None
        if should_log_loss(self):
            with profile_section(self, "loss_item"):
                loss_q_value = float(loss_q.item())
                loss_actor_value = float(loss_actor_total.item())

        return {
            "loss_q": loss_q_value,
            "loss_actor": loss_actor_value,
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
            return torch_amp.autocast("cuda", enabled=self.use_amp)

        # --- Q update (Double DQN per-agent + QMIX mixing) ---
        self.mixer_opt.zero_grad(set_to_none=True)
        self.q_opt.zero_grad(set_to_none=True)

        with _autocast():
            action_params_view = action_params.view(-1, self.n_agents, self.n_actions, self.param_dim)
            q_all = self.q_net(state, action_params_view)
            agent_qs = q_all.gather(2, action_discrete.unsqueeze(-1)).squeeze(-1)  # (B,N)
            q_tot = self.mixer(agent_qs, global_state)  # (B,1)

            with torch.no_grad():
                # y_real
                next_params_eval = self.actor(next_state)
                next_q_eval = self.q_net(next_state, next_params_eval)
                next_action = torch.argmax(next_q_eval, dim=2, keepdim=True)

                next_params_target = self.target_actor(next_state)
                next_q_target_all = self.target_q_net(next_state, next_params_target)
                next_agent_qs = next_q_target_all.gather(2, next_action).squeeze(-1)  # (B,N)
                next_q_tot = self.target_mixer(next_agent_qs, next_global_state)  # (B,1)
                y_real = reward + (1.0 - done) * self.gamma * next_q_tot
                y_real = self._clip_value_target(y_real)

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
                    y_model = self._clip_value_target(y_model)
                    if torch.isfinite(y_model).all():
                        td_target = (1.0 - alpha_model) * y_real + alpha_model * y_model
                    else:
                        td_target = y_real
                else:
                    td_target = y_real
                td_target = self._clip_value_target(td_target)

            loss_q = F.smooth_l1_loss(q_tot, td_target)

        if not torch.isfinite(loss_q):
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_q).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.mixer_opt)
                self.scaler.unscale_(self.q_opt)
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                _clip_batched_agent_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.scaler.step(self.mixer_opt)
            self.scaler.step(self.q_opt)
        else:
            loss_q.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                _clip_batched_agent_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.mixer_opt.step()
            self.q_opt.step()

        # --- Actor updates (team objective): maximize Q_tot predicted by mixer ---
        for p in self.q_net.parameters():
            p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

        self.actor_opt.zero_grad(set_to_none=True)

        with _autocast():
            params_pred = self.actor(state)
            q_pred = self.q_net(state, params_pred)
            agent_qs_actor = q_pred.mean(dim=2)
            q_tot_actor = self.mixer(agent_qs_actor, global_state)
            loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            for p in self.q_net.parameters():
                p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
            # Q optimizers may already have stepped under AMP above. Finalize the
            # scaler state so the next iteration can safely call unscale_ again.
            if self.use_amp:
                self.scaler.update()
            return {"loss_q": float(loss_q.item()), "loss_actor": float("nan"), "skipped": 1}

        if self.use_amp:
            self.scaler.scale(loss_actor_total).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.actor_opt)
                _clip_batched_agent_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_opt)
        else:
            loss_actor_total.backward()
            if self.max_grad_norm > 0.0:
                _clip_batched_agent_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

        # Restore critic grads.
        for p in self.q_net.parameters():
            p.requires_grad = True
        for p in self.mixer.parameters():
            p.requires_grad = True

        if self.use_amp:
            self.scaler.update()

        # --- Target updates ---
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        loss_q_value = None
        loss_actor_value = None
        if should_log_loss(self):
            loss_q_value = float(loss_q.item())
            loss_actor_value = float(loss_actor_total.item())

        return {
            "loss_q": loss_q_value,
            "loss_actor": loss_actor_value,
        }


__all__ = ["MPDQNQMIXTrainer"]
