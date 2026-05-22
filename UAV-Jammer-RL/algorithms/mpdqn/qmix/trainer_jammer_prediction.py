from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as torch_amp

from algorithms.mpdqn.qmix.mixer import QMIXMixer


class JammerPredictionHead(nn.Module):
    """Predict next-slot jammed-channel probabilities from recent sensing history."""

    def __init__(self, *, history_len: int, n_channel: int, hidden_dim: int = 64):
        super().__init__()
        self.history_len = int(history_len)
        self.n_channel = int(n_channel)
        in_dim = int(self.history_len * self.n_channel)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_channel),
        )

    def forward(self, sensing_history: torch.Tensor) -> torch.Tensor:
        if sensing_history.ndim != 3:
            raise ValueError(f"sensing_history must be (B,W,K), got {tuple(sensing_history.shape)}")
        if int(sensing_history.shape[1]) != self.history_len or int(sensing_history.shape[2]) != self.n_channel:
            raise ValueError(
                "sensing_history shape mismatch: expected "
                f"(*,{self.history_len},{self.n_channel}), got {tuple(sensing_history.shape)}"
            )
        return self.net(sensing_history.reshape(sensing_history.shape[0], -1))


class JammerAwareMPDQNActor(nn.Module):
    """MP-DQN actor that consumes local observation plus jammer-risk features."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        n_channel: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.n_channel = int(n_channel)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.n_channel, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_actions * self.param_dim),
            nn.Sigmoid(),
        )

    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        out = self.net(augmented_state)
        return out.view(augmented_state.shape[0], self.n_actions, self.param_dim)


class JammerAwareMPDQNQNetwork(nn.Module):
    """MP-DQN Q network that consumes local observation plus jammer-risk features."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        n_channel: int,
        hidden_dim: int = 128,
        q_hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.n_channel = int(n_channel)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.n_channel, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )

        q_input_dim = int(hidden_dim) + self.n_actions + self.param_dim
        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, int(q_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(q_hidden_dim), 1),
        )

    def forward(self, augmented_state: torch.Tensor, action_params: torch.Tensor) -> torch.Tensor:
        batch_size = augmented_state.shape[0]
        features = self.state_encoder(augmented_state)
        features = features.unsqueeze(1).expand(-1, self.n_actions, -1)
        action_onehot = torch.eye(self.n_actions, device=augmented_state.device).unsqueeze(0)
        action_onehot = action_onehot.expand(batch_size, -1, -1)
        x = torch.cat([features, action_onehot, action_params], dim=2)
        x = x.reshape(batch_size * self.n_actions, -1)
        q = self.q_head(x).view(batch_size, self.n_actions)
        return q


class JammerAwareMPDQNAgent:
    """
    MP-DQN agent with an auxiliary jammer predictor.

    The predictor is trained by BCE. Its sigmoid probabilities are concatenated
    to the local observation before the actor and Q-network.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        n_channel: int,
        jammer_history_len: int,
        buffer_capacity: int = 1,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        lr_jammer: Optional[float] = None,
        target_update_interval: int = 200,
        use_amp: bool = False,
        max_grad_norm: float = 10.0,
        jammer_pred_hidden_dim: int = 64,
        use_jammer_feature: bool = True,
        device: Optional[str] = None,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.total_param_dim = self.n_actions * self.param_dim
        self.n_channel = int(n_channel)
        self.jammer_history_len = int(jammer_history_len)
        self.use_jammer_feature = bool(use_jammer_feature)
        self.feature_scale = 1.0

        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)
        self.buffer_capacity = int(buffer_capacity)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")

        if lr_jammer is None:
            lr_jammer = float(lr_q)

        self.jammer_predictor = JammerPredictionHead(
            history_len=self.jammer_history_len,
            n_channel=self.n_channel,
            hidden_dim=int(jammer_pred_hidden_dim),
        ).to(self.device)
        self.actor = JammerAwareMPDQNActor(self.state_dim, self.n_actions, self.param_dim, self.n_channel).to(
            self.device
        )
        self.q_net = JammerAwareMPDQNQNetwork(self.state_dim, self.n_actions, self.param_dim, self.n_channel).to(
            self.device
        )

        self.target_jammer_predictor = copy.deepcopy(self.jammer_predictor).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=float(lr_q))
        self.jammer_predictor_opt = torch.optim.Adam(self.jammer_predictor.parameters(), lr=float(lr_jammer))

    def _default_history(self, state: torch.Tensor) -> torch.Tensor:
        sensing = state[:, -self.n_channel :].contiguous()
        return sensing.unsqueeze(1).expand(-1, self.jammer_history_len, -1).contiguous()

    def augment_state(
        self,
        state: torch.Tensor,
        sensing_history: Optional[torch.Tensor] = None,
        *,
        target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sensing_history is None:
            sensing_history = self._default_history(state)
        predictor = self.target_jammer_predictor if bool(target) else self.jammer_predictor
        logits = predictor(sensing_history)
        probs = torch.sigmoid(logits)
        if self.use_jammer_feature:
            feature = probs.detach() * float(self.feature_scale)
        else:
            feature = torch.zeros_like(probs)
        augmented = torch.cat([state, feature], dim=1)
        return augmented, logits, probs

    def select_action(self, state: np.ndarray, epsilon: float, sensing_history: Optional[np.ndarray] = None):
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)
        hist_t = None
        if sensing_history is not None:
            hist = np.asarray(sensing_history, dtype=np.float32).reshape(1, self.jammer_history_len, self.n_channel)
            hist_t = torch.from_numpy(hist).to(self.device)

        with torch.no_grad():
            aug_state, _, _ = self.augment_state(state_t, hist_t, target=False)
            params_all = self.actor(aug_state)
            q_values = self.q_net(aug_state, params_all)

        if random.random() < float(epsilon):
            action_discrete = random.randrange(self.n_actions)
        else:
            action_discrete = int(torch.argmax(q_values, dim=1).item())

        params_flat = params_all.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action_discrete, params_flat

    def select_action_batch(
        self,
        states: np.ndarray,
        epsilon: float,
        sensing_history: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must be (B,{self.state_dim}), got shape={states.shape}")
        state_t = torch.from_numpy(states).to(self.device)

        hist_t = None
        if sensing_history is not None:
            hist = np.asarray(sensing_history, dtype=np.float32)
            expected = (states.shape[0], self.jammer_history_len, self.n_channel)
            if hist.shape != expected:
                raise ValueError(f"sensing_history must be {expected}, got {hist.shape}")
            hist_t = torch.from_numpy(hist).to(self.device)

        with torch.no_grad():
            aug_state, _, _ = self.augment_state(state_t, hist_t, target=False)
            params_all = self.actor(aug_state)
            q_values = self.q_net(aug_state, params_all)

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


class JammerAwareJointReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._buffer: Deque[
            Tuple[
                np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool,
                np.ndarray, np.ndarray, np.ndarray,
            ]
        ] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        *,
        state: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        jammer_target: np.ndarray,
        sensing_history: np.ndarray,
        next_sensing_history: np.ndarray,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action_discrete, dtype=np.int64),
                np.asarray(action_params, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                np.asarray(jammer_target, dtype=np.float32),
                np.asarray(sensing_history, dtype=np.float32),
                np.asarray(next_sensing_history, dtype=np.float32),
            )
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.choice(len(self._buffer), size=int(batch_size), replace=False)
        (
            states, actions, params, rewards, next_states, dones,
            jammer_targets, sensing_histories, next_sensing_histories,
        ) = zip(*(self._buffer[i] for i in idx))
        return {
            "state": np.asarray(states, dtype=np.float32),
            "action_discrete": np.asarray(actions, dtype=np.int64),
            "action_params": np.asarray(params, dtype=np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),
            "next_state": np.asarray(next_states, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
            "jammer_target": np.asarray(jammer_targets, dtype=np.float32),
            "sensing_history": np.asarray(sensing_histories, dtype=np.float32),
            "next_sensing_history": np.asarray(next_sensing_histories, dtype=np.float32),
        }


class JammerAwareSequenceReplayBuffer:
    """Sequence replay with an extra next-slot jammer target for auxiliary supervision."""

    _Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool, np.ndarray]

    def __init__(self, *, n_envs: int, capacity: int = 500_000):
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity_total = capacity
        self.capacity_per_env = max(1, int(capacity) // int(self.n_envs))
        self._buffers: List[Deque[JammerAwareSequenceReplayBuffer._Transition]] = [
            deque(maxlen=int(self.capacity_per_env)) for _ in range(int(self.n_envs))
        ]

    def __len__(self) -> int:
        return int(sum(len(b) for b in self._buffers))

    def add(
        self,
        *,
        env_id: int,
        state: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        reward_team: float,
        next_state: np.ndarray,
        done: bool,
        jammer_target: np.ndarray,
    ) -> None:
        e = int(env_id)
        if e < 0 or e >= self.n_envs:
            raise ValueError(f"env_id out of range: {e} not in [0,{self.n_envs})")
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
        ad = np.asarray(action_discrete, dtype=np.int64).reshape(-1)
        ap = np.asarray(action_params, dtype=np.float32)
        if ap.ndim == 1:
            ap = ap.reshape(int(ad.shape[0]), -1)
        jt = np.asarray(jammer_target, dtype=np.float32).reshape(-1)
        self._buffers[e].append((s, ad, ap, float(reward_team), ns, bool(done), jt))

    def sample_sequences(self, *, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        ready_envs = [i for i, b in enumerate(self._buffers) if len(b) >= seq_len]
        if len(ready_envs) == 0:
            raise ValueError(f"No env has enough transitions for seq_len={seq_len}")

        rng = np.random.default_rng()
        state_list = []
        ad_list = []
        ap_list = []
        r_list = []
        ns_list = []
        d_list = []
        jt_list = []
        env_id_list = []

        max_tries = 128
        for _ in range(batch_size):
            ok = False
            for _try in range(max_tries):
                e = int(rng.choice(ready_envs))
                buf = self._buffers[e]
                start = int(rng.integers(0, len(buf) - seq_len + 1))
                if any(bool(buf[start + k][5]) for k in range(seq_len)):
                    continue
                seq = [buf[start + k] for k in range(seq_len)]
                continuous = True
                for k in range(seq_len - 1):
                    if not np.allclose(seq[k][4], seq[k + 1][0], atol=1e-5, rtol=1e-5):
                        continuous = False
                        break
                if not continuous:
                    continue

                s_seq, ad_seq, ap_seq, r_seq, ns_seq, done_seq, jt_seq = zip(*seq)
                state_list.append(np.stack(s_seq, axis=0))
                ad_list.append(np.stack(ad_seq, axis=0))
                ap_list.append(np.stack(ap_seq, axis=0))
                r_list.append(np.asarray(r_seq, dtype=np.float32).reshape(seq_len, 1))
                ns_list.append(np.stack(ns_seq, axis=0))
                d_list.append(np.asarray(done_seq, dtype=np.float32).reshape(seq_len, 1))
                jt_list.append(np.stack(jt_seq, axis=0))
                env_id_list.append(e)
                ok = True
                break
            if not ok:
                raise RuntimeError(
                    f"Failed to sample a valid sequence after {max_tries} tries. "
                    "Consider increasing capacity, lowering seq_len, or ensuring done/reset is handled."
                )

        return {
            "state_seq": np.asarray(state_list, dtype=np.float32),
            "action_discrete_seq": np.asarray(ad_list, dtype=np.int64),
            "action_params_seq": np.asarray(ap_list, dtype=np.float32),
            "reward_seq": np.asarray(r_list, dtype=np.float32),
            "next_state_seq": np.asarray(ns_list, dtype=np.float32),
            "done_seq": np.asarray(d_list, dtype=np.float32),
            "jammer_target_seq": np.asarray(jt_list, dtype=np.float32),
            "env_id": np.asarray(env_id_list, dtype=np.int32),
        }


@dataclass(frozen=True)
class JammerAwareMPDQNQMIXDims:
    n_agents: int
    agent_state_dim: int
    n_actions: int
    param_dim: int
    n_channel: int
    jammer_history_len: int


class JammerAwareMPDQNQMIXValueTeacher:
    """Value-teacher adapter for jammer-aware MP-DQN/QMIX networks."""

    def __init__(self, trainer: "JammerAwareMPDQNQMIXTrainer", dims: JammerAwareMPDQNQMIXDims):
        self.trainer = trainer
        self.dims = dims
        for agent in self.trainer.agents:
            for p in agent.target_q_net.parameters():
                p.requires_grad_(False)
            for p in agent.target_jammer_predictor.parameters():
                p.requires_grad_(False)
            agent.target_q_net.eval()
            agent.target_jammer_predictor.eval()
        for p in self.trainer.target_mixer.parameters():
            p.requires_grad_(False)
        self.trainer.target_mixer.eval()

    def _reshape_state(self, global_state: torch.Tensor) -> torch.Tensor:
        if global_state.ndim != 2:
            raise ValueError(f"global_state must be (B, N*S), got {tuple(global_state.shape)}")
        bsz = int(global_state.shape[0])
        return global_state.view(bsz, int(self.dims.n_agents), int(self.dims.agent_state_dim))

    def _history_from_state(self, state: torch.Tensor) -> torch.Tensor:
        sensing = state[:, :, -int(self.dims.n_channel) :]
        return sensing.unsqueeze(2).expand(-1, -1, int(self.dims.jammer_history_len), -1).contiguous()

    @torch.no_grad()
    def greedy_action(self, global_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._reshape_state(global_state)
        history = self._history_from_state(state)
        bsz = int(state.shape[0])

        a_list = []
        params_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            h_i = history[:, i, :, :]
            aug_i, _, _ = self.trainer.agents[i].augment_state(s_i, h_i, target=False)
            params_all = self.trainer.agents[i].actor(aug_i)
            q_eval = self.trainer.agents[i].q_net(aug_i, params_all)
            a_i = torch.argmax(q_eval, dim=1)
            a_list.append(a_i)
            params_list.append(params_all.reshape(bsz, -1))

        action_discrete = torch.stack(a_list, dim=1).to(torch.long)
        action_params_flat = torch.stack(params_list, dim=1).to(torch.float32)
        return action_discrete, action_params_flat

    def q_tot_target(
        self,
        global_state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params_flat: torch.Tensor,
    ) -> torch.Tensor:
        state = self._reshape_state(global_state)
        history = self._history_from_state(state)
        bsz = int(state.shape[0])

        q_sa_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            h_i = history[:, i, :, :]
            a_i = action_discrete[:, i].view(-1, 1)
            params_i = action_params_flat[:, i, :].view(bsz, int(self.dims.n_actions), int(self.dims.param_dim))
            aug_i, _, _ = self.trainer.agents[i].augment_state(s_i, h_i, target=True)
            q_all_i = self.trainer.agents[i].target_q_net(aug_i, params_i)
            q_sa_i = q_all_i.gather(1, a_i)
            q_sa_list.append(q_sa_i)

        agent_qs = torch.cat(q_sa_list, dim=1)
        q_tot = self.trainer.target_mixer(agent_qs, global_state)
        return q_tot.to(torch.float32)


class JammerAwareMPDQNQMIXTrainer:
    """
    QMIX + MP-DQN trainer with explicit next-slot jammer prediction.

    This preserves the existing CTDE/QMIX and value-expansion path, adding only:
      - per-agent jammer prediction heads;
      - a BCE auxiliary loss;
      - predicted jammer probabilities concatenated to actor/Q inputs.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        global_state_dim: int,
        n_channel: int,
        jammer_history_len: int = 4,
        jammer_pred_hidden_dim: int = 64,
        jammer_aux_weight: float = 0.1,
        use_jammer_feature: bool = True,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_actor: float = 1e-3,
        lr_q: float = 1e-3,
        lr_jammer: Optional[float] = None,
        lr_mixer: Optional[float] = None,
        target_update_interval: int = 200,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
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
        self.n_channel = int(n_channel)
        self.jammer_history_len = int(jammer_history_len)
        self.jammer_aux_weight = float(jammer_aux_weight)
        self.use_jammer_feature = bool(use_jammer_feature)
        self.feature_scale = 1.0

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

        self.agents = [
            JammerAwareMPDQNAgent(
                state_dim=self.state_dim,
                n_actions=self.n_actions,
                param_dim=self.param_dim,
                n_channel=self.n_channel,
                jammer_history_len=self.jammer_history_len,
                buffer_capacity=1,
                batch_size=self.batch_size,
                gamma=self.gamma,
                lr_actor=lr_actor,
                lr_q=lr_q,
                lr_jammer=lr_jammer,
                target_update_interval=self.target_update_interval,
                use_amp=self.use_amp,
                max_grad_norm=self.max_grad_norm,
                jammer_pred_hidden_dim=int(jammer_pred_hidden_dim),
                use_jammer_feature=self.use_jammer_feature,
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

        self.buffer = JammerAwareJointReplayBuffer(capacity=int(buffer_capacity))
        self.learn_steps = 0

    def _clip_value_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_target_clip is None or self.value_target_clip <= 0.0:
            return x
        return torch.clamp(x, min=-self.value_target_clip, max=self.value_target_clip)

    def _extract_history_from_seq(self, global_seq: torch.Tensor) -> torch.Tensor:
        if global_seq.ndim != 3:
            raise ValueError(f"global_seq must be (B,L,Ds), got {tuple(global_seq.shape)}")
        bsz, seq_len, _ = global_seq.shape
        local = global_seq.view(bsz, seq_len, self.n_agents, self.state_dim)
        sensing = local[:, :, :, -self.n_channel :]
        if int(seq_len) >= self.jammer_history_len:
            hist = sensing[:, -self.jammer_history_len :, :, :]
        else:
            pad_len = self.jammer_history_len - int(seq_len)
            pad = sensing[:, :1, :, :].expand(-1, pad_len, -1, -1)
            hist = torch.cat([pad, sensing], dim=1)
        return hist.permute(0, 2, 1, 3).contiguous()

    def set_feature_scale(self, scale: float) -> None:
        """Set the multiplicative scale on jammer-predictor features in augmented states.

        Used to ramp the feature in from 0 (BCE-only warmup) to 1 (full feature) over
        the first few hundred episodes, so the actor/Q never sees raw noise from an
        un-trained predictor.
        """
        s = float(scale)
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        self.feature_scale = float(s)
        for agent in self.agents:
            agent.feature_scale = float(s)

    def select_actions(
        self,
        states: List[np.ndarray],
        epsilon: float,
        sensing_histories: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent states, got {len(states)}")
        actions: List[Tuple[int, np.ndarray]] = []
        for i in range(self.n_agents):
            hist_i = None if sensing_histories is None else sensing_histories[i]
            action_discrete, action_params = self.agents[i].select_action(states[i], epsilon, hist_i)
            actions.append((int(action_discrete), np.asarray(action_params, dtype=np.float32)))
        return actions

    def store_transition(
        self,
        states: List[np.ndarray],
        actions: List[Tuple[int, np.ndarray]],
        rewards: np.ndarray,
        next_states: List[np.ndarray],
        done: bool = False,
        jammer_target: Optional[np.ndarray] = None,
        sensing_histories: Optional[np.ndarray] = None,
        next_sensing_histories: Optional[np.ndarray] = None,
    ) -> None:
        state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0)
        next_state_arr = np.stack([np.asarray(s, dtype=np.float32) for s in next_states], axis=0)
        action_discrete_arr = np.asarray([int(a[0]) for a in actions], dtype=np.int64)
        action_params_arr = np.stack([np.asarray(a[1], dtype=np.float32).reshape(-1) for a in actions], axis=0)
        reward_global = float(np.mean(np.asarray(rewards, dtype=np.float32)))
        if jammer_target is None:
            jammer_target_arr = np.zeros((self.n_channel,), dtype=np.float32)
        else:
            jammer_target_arr = np.asarray(jammer_target, dtype=np.float32).reshape(self.n_channel)
        expected = (self.n_agents, self.jammer_history_len, self.n_channel)
        if sensing_histories is None:
            current_sensing = state_arr[:, -self.n_channel :]
            sensing_history_arr = np.repeat(
                current_sensing[:, None, :],
                int(self.jammer_history_len),
                axis=1,
            ).astype(np.float32)
        else:
            sensing_history_arr = np.asarray(sensing_histories, dtype=np.float32)
            if sensing_history_arr.shape != expected:
                raise ValueError(f"sensing_histories must be {expected}, got {sensing_history_arr.shape}")

        if next_sensing_histories is None:
            # Fallback: roll the current history forward by one and append the next-state sensing tail.
            rolled = np.roll(sensing_history_arr, shift=-1, axis=1)
            rolled[:, -1, :] = next_state_arr[:, -self.n_channel :]
            next_sensing_history_arr = rolled.astype(np.float32)
        else:
            next_sensing_history_arr = np.asarray(next_sensing_histories, dtype=np.float32)
            if next_sensing_history_arr.shape != expected:
                raise ValueError(f"next_sensing_histories must be {expected}, got {next_sensing_history_arr.shape}")

        self.buffer.add(
            state=state_arr,
            action_discrete=action_discrete_arr,
            action_params=action_params_arr,
            reward=reward_global,
            next_state=next_state_arr,
            done=bool(done),
            jammer_target=jammer_target_arr,
            sensing_history=sensing_history_arr,
            next_sensing_history=next_sensing_history_arr,
        )

    def _jammer_aux_loss(
        self,
        *,
        state: torch.Tensor,
        history: torch.Tensor,
        jammer_target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        target = jammer_target.to(torch.float32)
        for i in range(self.n_agents):
            s_i = state[:, i, :]
            h_i = history[:, i, :, :]
            _, logits_i, _ = self.agents[i].augment_state(s_i, h_i, target=False)
            losses.append(F.binary_cross_entropy_with_logits(logits_i, target))
        return torch.stack(losses).mean()

    def _step_critic_optimizers(self, loss_total: torch.Tensor) -> None:
        if self.use_amp:
            self.scaler.scale(loss_total).backward()
            if self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.mixer_opt)
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    self.scaler.unscale_(agent.q_opt)
                    self.scaler.unscale_(agent.jammer_predictor_opt)
                    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(agent.jammer_predictor.parameters(), self.max_grad_norm)
            self.scaler.step(self.mixer_opt)
            for agent in self.agents:
                self.scaler.step(agent.q_opt)
                self.scaler.step(agent.jammer_predictor_opt)
        else:
            loss_total.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.max_grad_norm)
                for agent in self.agents:
                    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(agent.jammer_predictor.parameters(), self.max_grad_norm)
            self.mixer_opt.step()
            for agent in self.agents:
                agent.q_opt.step()
                agent.jammer_predictor_opt.step()

    def _target_update_if_needed(self) -> None:
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            for agent in self.agents:
                agent.target_actor.load_state_dict(agent.actor.state_dict())
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())
                agent.target_jammer_predictor.load_state_dict(agent.jammer_predictor.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _train_from_tensors(
        self,
        *,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        global_state: torch.Tensor,
        next_global_state: torch.Tensor,
        history: torch.Tensor,
        next_history: torch.Tensor,
        jammer_target: torch.Tensor,
        td_target_override: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        def _autocast():
            return torch_amp.autocast("cuda", enabled=self.use_amp)

        self.mixer_opt.zero_grad(set_to_none=True)
        for agent in self.agents:
            agent.q_opt.zero_grad(set_to_none=True)
            agent.jammer_predictor_opt.zero_grad(set_to_none=True)

        with _autocast():
            q_sa_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                h_i = history[:, i, :, :]
                a_i = action_discrete[:, i].view(-1, 1)
                params_i = action_params[:, i, :].view(-1, self.n_actions, self.param_dim)
                aug_i, _, _ = self.agents[i].augment_state(s_i, h_i, target=False)
                q_all_i = self.agents[i].q_net(aug_i, params_i)
                q_sa_list.append(q_all_i.gather(1, a_i))

            agent_qs = torch.cat(q_sa_list, dim=1)
            q_tot = self.mixer(agent_qs, global_state)

            if td_target_override is None:
                with torch.no_grad():
                    next_q_list = []
                    for i in range(self.n_agents):
                        ns_i = next_state[:, i, :]
                        nh_i = next_history[:, i, :, :]
                        aug_eval_i, _, _ = self.agents[i].augment_state(ns_i, nh_i, target=False)
                        next_params_eval = self.agents[i].actor(aug_eval_i)
                        next_q_eval = self.agents[i].q_net(aug_eval_i, next_params_eval)
                        next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                        aug_target_i, _, _ = self.agents[i].augment_state(ns_i, nh_i, target=True)
                        next_params_target = self.agents[i].target_actor(aug_target_i)
                        next_q_target_all = self.agents[i].target_q_net(aug_target_i, next_params_target)
                        next_q_list.append(next_q_target_all.gather(1, next_action))

                    next_agent_qs = torch.cat(next_q_list, dim=1)
                    next_q_tot = self.target_mixer(next_agent_qs, next_global_state)
                    td_target = reward + (1.0 - done) * self.gamma * next_q_tot
                    td_target = self._clip_value_target(td_target)
            else:
                td_target = td_target_override

            loss_q = F.smooth_l1_loss(q_tot, td_target)
            loss_jammer = self._jammer_aux_loss(state=state, history=history, jammer_target=jammer_target)
            loss_total = loss_q + float(self.jammer_aux_weight) * loss_jammer

        if not torch.isfinite(loss_total):
            return {"loss_q": float("nan"), "loss_actor": float("nan"), "loss_jammer": float("nan"), "skipped": 1}

        self._step_critic_optimizers(loss_total)

        for agent in self.agents:
            for p in agent.q_net.parameters():
                p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

        for agent in self.agents:
            agent.actor_opt.zero_grad(set_to_none=True)

        with _autocast():
            q_actor_list = []
            for i in range(self.n_agents):
                s_i = state[:, i, :]
                h_i = history[:, i, :, :]
                aug_i, _, _ = self.agents[i].augment_state(s_i, h_i, target=False)
                params_pred = self.agents[i].actor(aug_i)
                q_pred = self.agents[i].q_net(aug_i, params_pred)
                q_actor_list.append(q_pred.mean(dim=1, keepdim=True))

            agent_qs_actor = torch.cat(q_actor_list, dim=1)
            q_tot_actor = self.mixer(agent_qs_actor, global_state)
            loss_actor_total = -q_tot_actor.mean()

        if not torch.isfinite(loss_actor_total):
            for agent in self.agents:
                for p in agent.q_net.parameters():
                    p.requires_grad = True
            for p in self.mixer.parameters():
                p.requires_grad = True
            if self.use_amp:
                self.scaler.update()
            return {
                "loss_q": float(loss_q.item()),
                "loss_actor": float("nan"),
                "loss_jammer": float(loss_jammer.item()),
                "skipped": 1,
            }

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

        self._target_update_if_needed()

        return {
            "loss_q": float(loss_q.item()),
            "loss_actor": float(loss_actor_total.item()),
            "loss_jammer": float(loss_jammer.item()),
        }

    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(batch["state"]).to(self.device).to(torch.float32)
        next_state = torch.from_numpy(batch["next_state"]).to(self.device).to(torch.float32)
        action_discrete = torch.from_numpy(batch["action_discrete"]).to(self.device).to(torch.long)
        action_params = torch.from_numpy(batch["action_params"]).to(self.device).to(torch.float32)
        reward = torch.from_numpy(batch["reward"]).to(self.device).to(torch.float32).view(-1, 1)
        done = torch.from_numpy(batch["done"]).to(self.device).to(torch.float32).view(-1, 1)
        jammer_target = torch.from_numpy(batch["jammer_target"]).to(self.device).to(torch.float32)
        history = torch.from_numpy(batch["sensing_history"]).to(self.device).to(torch.float32)
        next_history = torch.from_numpy(batch["next_sensing_history"]).to(self.device).to(torch.float32)

        global_state = state.reshape(state.shape[0], -1)
        next_global_state = next_state.reshape(next_state.shape[0], -1)

        return self._train_from_tensors(
            state=state,
            next_state=next_state,
            action_discrete=action_discrete,
            action_params=action_params,
            reward=reward,
            done=done,
            global_state=global_state,
            next_global_state=next_global_state,
            history=history,
            next_history=next_history,
            jammer_target=jammer_target,
        )

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
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        alpha_model = float(alpha_model)
        if not (0.0 <= alpha_model <= 1.0):
            raise ValueError(f"alpha_model must be in [0,1], got {alpha_model}")

        try:
            batch = seq_buffer.sample_sequences(batch_size=int(self.batch_size), seq_len=int(seq_len))
        except Exception:
            return None

        from algorithms.world_model.action_encoding import encode_joint_action_exec
        from algorithms.world_model.value_consistency import rollout_td_lambda_return

        state_seq = torch.from_numpy(batch["state_seq"]).to(self.device).to(torch.float32)
        next_state_seq = torch.from_numpy(batch["next_state_seq"]).to(self.device).to(torch.float32)
        action_discrete_seq = torch.from_numpy(batch["action_discrete_seq"]).to(self.device).to(torch.long)
        action_params_seq = torch.from_numpy(batch["action_params_seq"]).to(self.device).to(torch.float32)
        reward_seq = torch.from_numpy(batch["reward_seq"]).to(self.device).to(torch.float32)
        done_seq = torch.from_numpy(batch["done_seq"]).to(self.device).to(torch.float32)
        jammer_target_seq = torch.from_numpy(batch["jammer_target_seq"]).to(self.device).to(torch.float32)

        global_state = state_seq[:, -1, :]
        next_global_state = next_state_seq[:, -1, :]

        bsz = int(global_state.shape[0])
        state = global_state.view(bsz, self.n_agents, self.state_dim)
        next_state = next_global_state.view(bsz, self.n_agents, self.state_dim)

        action_discrete = action_discrete_seq[:, -1, :]
        action_params = action_params_seq[:, -1, :, :]
        reward = reward_seq[:, -1, :]
        done = done_seq[:, -1, :]
        jammer_target = jammer_target_seq[:, -1, :]

        history = self._extract_history_from_seq(state_seq)
        next_history = self._extract_history_from_seq(next_state_seq)

        bsz, seq_l, n_agents = action_discrete_seq.shape
        ad_flat = action_discrete_seq.reshape(int(bsz * seq_l), int(n_agents))
        ap_flat = action_params_seq.reshape(int(bsz * seq_l), int(n_agents), -1)
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
        action_enc_seq = action_enc_flat.view(int(bsz), int(seq_l), -1)

        with torch.no_grad():
            next_q_list = []
            for i in range(self.n_agents):
                ns_i = next_state[:, i, :]
                nh_i = next_history[:, i, :, :]
                aug_eval_i, _, _ = self.agents[i].augment_state(ns_i, nh_i, target=False)
                next_params_eval = self.agents[i].actor(aug_eval_i)
                next_q_eval = self.agents[i].q_net(aug_eval_i, next_params_eval)
                next_action = torch.argmax(next_q_eval, dim=1, keepdim=True)

                aug_target_i, _, _ = self.agents[i].augment_state(ns_i, nh_i, target=True)
                next_params_target = self.agents[i].target_actor(aug_target_i)
                next_q_target_all = self.agents[i].target_q_net(aug_target_i, next_params_target)
                next_q_list.append(next_q_target_all.gather(1, next_action))

            next_agent_qs = torch.cat(next_q_list, dim=1)
            next_q_tot = self.target_mixer(next_agent_qs, next_global_state)
            y_real = reward + (1.0 - done) * self.gamma * next_q_tot
            y_real = self._clip_value_target(y_real)

            if alpha_model > 0.0:
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

        return self._train_from_tensors(
            state=state,
            next_state=next_state,
            action_discrete=action_discrete,
            action_params=action_params,
            reward=reward,
            done=done,
            global_state=global_state,
            next_global_state=next_global_state,
            history=history,
            next_history=next_history,
            jammer_target=jammer_target,
            td_target_override=td_target,
        )


__all__ = [
    "JammerPredictionHead",
    "JammerAwareMPDQNAgent",
    "JammerAwareJointReplayBuffer",
    "JammerAwareSequenceReplayBuffer",
    "JammerAwareMPDQNQMIXDims",
    "JammerAwareMPDQNQMIXValueTeacher",
    "JammerAwareMPDQNQMIXTrainer",
]
