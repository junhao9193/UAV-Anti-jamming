from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class JointWorldModelConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    n_layers: int = 1
    stochastic_dim: int = 32
    min_std: float = 0.1
    kl_beta: float = 0.1
    free_nats: float = 1.0


@dataclass
class RSSMHiddenState:
    deter: torch.Tensor
    stoch: torch.Tensor


@dataclass
class RSSMObserveOutput:
    delta_seq: torch.Tensor
    reward_seq: torch.Tensor
    feature_seq: torch.Tensor
    hidden: RSSMHiddenState
    prior_mean_seq: torch.Tensor
    prior_std_seq: torch.Tensor
    post_mean_seq: torch.Tensor
    post_std_seq: torch.Tensor


class JointWorldModel(nn.Module):
    """
    RSSM-style joint world model.

    Deterministic state:
      h_t = GRU(h_{t-1}, [z_{t-1}, a_t])

    Stochastic latent:
      p(z_t | h_t)           prior
      q(z_t | h_t, s_t)      posterior (training / context encoding)

    Prediction heads use the posterior feature [h_t, z_t] on real sequences and
    the prior feature during imagined rollouts.
    """

    def __init__(self, cfg: JointWorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.hidden_dim = int(cfg.hidden_dim)
        self.n_layers = int(cfg.n_layers)
        self.stochastic_dim = int(cfg.stochastic_dim)
        self.min_std = float(cfg.min_std)
        self.kl_beta = float(cfg.kl_beta)
        self.free_nats = float(cfg.free_nats)

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.stochastic_dim <= 0:
            raise ValueError("stochastic_dim must be positive")
        if self.min_std <= 0.0:
            raise ValueError("min_std must be positive")

        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.GELU(),
        )

        self.gru_cells = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            input_dim = self.stochastic_dim + self.hidden_dim if layer_idx == 0 else self.hidden_dim
            self.gru_cells.append(nn.GRUCell(input_dim, self.hidden_dim))

        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim),
        )

        feat_dim = self.hidden_dim + self.stochastic_dim
        self.head_state = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.head_reward = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int, *, device: torch.device | None = None) -> RSSMHiddenState:
        dev = device if device is not None else next(self.parameters()).device
        deter = torch.zeros((self.n_layers, int(batch_size), self.hidden_dim), dtype=torch.float32, device=dev)
        stoch = torch.zeros((int(batch_size), self.stochastic_dim), dtype=torch.float32, device=dev)
        return RSSMHiddenState(deter=deter, stoch=stoch)

    def _ensure_seq(self, x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or int(x.shape[2]) != int(dim):
            raise ValueError(f"{name} must be (B,L,{dim}) or (B,{dim}), got {tuple(x.shape)}")
        return x

    def _split_stats(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def _sample_latent(self, mean: torch.Tensor, std: torch.Tensor, *, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        eps = torch.randn_like(std)
        return mean + std * eps

    def _feature_from_hidden(self, hidden: RSSMHiddenState) -> torch.Tensor:
        return torch.cat([hidden.deter[-1], hidden.stoch], dim=-1)

    def _transition_deter(self, prev_hidden: RSSMHiddenState, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or int(action.shape[1]) != self.action_dim:
            raise ValueError(f"action must be (B,{self.action_dim}), got {tuple(action.shape)}")

        action_emb = self.action_embed(action)
        x = torch.cat([prev_hidden.stoch, action_emb], dim=-1)

        deter_layers = []
        layer_input = x
        for layer_idx, cell in enumerate(self.gru_cells):
            h_prev = prev_hidden.deter[layer_idx]
            h_next = cell(layer_input, h_prev)
            deter_layers.append(h_next)
            layer_input = h_next
        return torch.stack(deter_layers, dim=0)

    def _prior(self, deter_top: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._split_stats(self.prior_net(deter_top))

    def _posterior(self, deter_top: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or int(state.shape[1]) != self.state_dim:
            raise ValueError(f"state must be (B,{self.state_dim}), got {tuple(state.shape)}")
        x = torch.cat([deter_top, state], dim=-1)
        return self._split_stats(self.posterior_net(x))

    def observe(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: RSSMHiddenState | None = None,
        *,
        sample: bool,
    ) -> RSSMObserveOutput:
        state_seq = self._ensure_seq(state_seq, self.state_dim, "state_seq")
        action_seq = self._ensure_seq(action_seq, self.action_dim, "action_seq")
        if int(state_seq.shape[0]) != int(action_seq.shape[0]) or int(state_seq.shape[1]) != int(action_seq.shape[1]):
            raise ValueError(
                f"state_seq/action_seq must share (B,L), got {tuple(state_seq.shape)} vs {tuple(action_seq.shape)}"
            )

        batch_size, seq_len, _ = state_seq.shape
        prev_hidden = hidden if hidden is not None else self.init_hidden(int(batch_size), device=state_seq.device)

        delta_list = []
        reward_list = []
        feat_list = []
        prior_mean_list = []
        prior_std_list = []
        post_mean_list = []
        post_std_list = []

        for t in range(int(seq_len)):
            action_t = action_seq[:, t, :]
            state_t = state_seq[:, t, :]

            deter_t = self._transition_deter(prev_hidden, action_t)
            deter_top_t = deter_t[-1]
            prior_mean_t, prior_std_t = self._prior(deter_top_t)
            post_mean_t, post_std_t = self._posterior(deter_top_t, state_t)
            stoch_t = self._sample_latent(post_mean_t, post_std_t, sample=sample)

            hidden_t = RSSMHiddenState(deter=deter_t, stoch=stoch_t)
            feat_t = self._feature_from_hidden(hidden_t)

            delta_list.append(self.head_state(feat_t))
            reward_list.append(self.head_reward(feat_t))
            feat_list.append(feat_t)
            prior_mean_list.append(prior_mean_t)
            prior_std_list.append(prior_std_t)
            post_mean_list.append(post_mean_t)
            post_std_list.append(post_std_t)
            prev_hidden = hidden_t

        return RSSMObserveOutput(
            delta_seq=torch.stack(delta_list, dim=1),
            reward_seq=torch.stack(reward_list, dim=1),
            feature_seq=torch.stack(feat_list, dim=1),
            hidden=prev_hidden,
            prior_mean_seq=torch.stack(prior_mean_list, dim=1),
            prior_std_seq=torch.stack(prior_std_list, dim=1),
            post_mean_seq=torch.stack(post_mean_list, dim=1),
            post_std_seq=torch.stack(post_std_list, dim=1),
        )

    def encode(
        self,
        *,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: RSSMHiddenState | None = None,
    ) -> Tuple[torch.Tensor, RSSMHiddenState]:
        obs = self.observe(state_seq=state_seq, action_seq=action_seq, hidden=hidden, sample=False)
        return obs.feature_seq, obs.hidden

    def forward(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: RSSMHiddenState | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RSSMHiddenState]:
        obs = self.observe(state_seq=state_seq, action_seq=action_seq, hidden=hidden, sample=self.training)
        return obs.delta_seq, obs.reward_seq, obs.feature_seq, obs.hidden

    def predict_from_hidden(
        self,
        *,
        state: torch.Tensor,
        hidden: RSSMHiddenState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or int(state.shape[1]) != self.state_dim:
            raise ValueError(f"state must be (B,{self.state_dim}), got {tuple(state.shape)}")
        feat = self._feature_from_hidden(hidden)
        delta = self.head_state(feat)
        reward = self.head_reward(feat)
        next_state = state + delta
        return next_state, reward

    def imagine_step(
        self,
        *,
        action: torch.Tensor,
        hidden: RSSMHiddenState,
        sample: bool = False,
    ) -> Tuple[RSSMHiddenState, torch.Tensor, torch.Tensor]:
        deter_next = self._transition_deter(hidden, action)
        prior_mean, prior_std = self._prior(deter_next[-1])
        stoch_next = self._sample_latent(prior_mean, prior_std, sample=sample)
        hidden_next = RSSMHiddenState(deter=deter_next, stoch=stoch_next)
        return hidden_next, prior_mean, prior_std

    def advance_hidden(
        self,
        *,
        next_action: torch.Tensor,
        hidden: RSSMHiddenState,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, RSSMHiddenState]:
        hidden_next, _, _ = self.imagine_step(action=next_action, hidden=hidden, sample=sample)
        feat_next = self._feature_from_hidden(hidden_next)
        return feat_next, hidden_next


__all__ = [
    "JointWorldModel",
    "JointWorldModelConfig",
    "RSSMHiddenState",
    "RSSMObserveOutput",
]
