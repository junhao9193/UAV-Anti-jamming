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


class JointWorldModel(nn.Module):
    """
    Joint recurrent world model (GRU):

      h_t = GRU([s_t, u_t], h_{t-1})
      Δŝ_t, r̂_t = heads(h_t)
      ŝ_{t+1} = s_t + Δŝ_t

    Notes:
    - We intentionally do NOT model `done` (your env is mostly continuing / time-limit truncated).
    - We keep delta-state prediction, as recommended in `doc/世界模型构建指南.md`.
    - To *train* or *use* the GRU memory, you still need contiguous sequences sampled from replay.
    """

    def __init__(self, cfg: JointWorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.hidden_dim = int(cfg.hidden_dim)
        self.n_layers = int(cfg.n_layers)
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.gru = nn.GRU(
            input_size=int(self.state_dim + self.action_dim),
            hidden_size=int(self.hidden_dim),
            num_layers=int(self.n_layers),
            batch_first=True,
        )

        self.head_state = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.head_reward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros((self.n_layers, int(batch_size), self.hidden_dim), dtype=torch.float32, device=dev)

    def _ensure_seq(self, x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,D) -> (B,1,D)
        if x.ndim != 3 or int(x.shape[2]) != int(dim):
            raise ValueError(f"{name} must be (B,L,{dim}) or (B,{dim}), got {tuple(x.shape)}")
        return x

    def encode(
        self,
        *,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the GRU over a (state, action) sequence.

        Args:
            state_seq:  (B,L,Ds) or (B,Ds)
            action_seq: (B,L,Du) or (B,Du)
            hidden:     (n_layers,B,H) optional
        Returns:
            h_seq:  (B,L,H) last-layer outputs
            h_last: (n_layers,B,H) final hidden for all layers
        """
        state_seq = self._ensure_seq(state_seq, self.state_dim, "state_seq")
        action_seq = self._ensure_seq(action_seq, self.action_dim, "action_seq")
        if int(state_seq.shape[0]) != int(action_seq.shape[0]) or int(state_seq.shape[1]) != int(action_seq.shape[1]):
            raise ValueError(
                f"state_seq/action_seq must share (B,L), got {tuple(state_seq.shape)} vs {tuple(action_seq.shape)}"
            )

        x = torch.cat([state_seq, action_seq], dim=2)  # (B,L,Ds+Du)
        if hidden is None:
            hidden = self.init_hidden(int(x.shape[0]), device=x.device)
        h_seq, h_last = self.gru(x, hidden)
        return h_seq, h_last

    def forward(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_seq:  (B,L,Ds)
            reward_seq: (B,L,1)
            h_seq:      (B,L,H) last-layer outputs
            h_last:     (n_layers,B,H)
        """
        h_seq, h_last = self.encode(state_seq=state_seq, action_seq=action_seq, hidden=hidden)
        delta_seq = self.head_state(h_seq)
        reward_seq = self.head_reward(h_seq)
        return delta_seq, reward_seq, h_seq, h_last

    def predict_from_hidden(self, *, state: torch.Tensor, h_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict (next_state, reward) from current state and GRU output h_t (last layer).
        """
        if state.ndim != 2 or int(state.shape[1]) != self.state_dim:
            raise ValueError(f"state must be (B,{self.state_dim}), got {tuple(state.shape)}")
        if h_out.ndim != 2 or int(h_out.shape[1]) != self.hidden_dim:
            raise ValueError(f"h_out must be (B,{self.hidden_dim}), got {tuple(h_out.shape)}")

        delta = self.head_state(h_out)
        reward = self.head_reward(h_out)
        next_state = state + delta
        return next_state, reward

    def advance_hidden(
        self,
        *,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance hidden state by consuming the next (s,u).

        Args:
            next_state: (B,Ds)
            next_action:(B,Du)
            hidden:     (n_layers,B,H) current hidden at time t
        Returns:
            h_out_next: (B,H) GRU output at t+1 (last layer)
            hidden_next:(n_layers,B,H)
        """
        if next_state.ndim != 2 or int(next_state.shape[1]) != self.state_dim:
            raise ValueError(f"next_state must be (B,{self.state_dim}), got {tuple(next_state.shape)}")
        if next_action.ndim != 2 or int(next_action.shape[1]) != self.action_dim:
            raise ValueError(f"next_action must be (B,{self.action_dim}), got {tuple(next_action.shape)}")
        if hidden.ndim != 3 or int(hidden.shape[0]) != self.n_layers or int(hidden.shape[2]) != self.hidden_dim:
            raise ValueError(
                f"hidden must be ({self.n_layers},B,{self.hidden_dim}), got {tuple(hidden.shape)}"
            )

        x = torch.cat([next_state, next_action], dim=1).unsqueeze(1)  # (B,1,Ds+Du)
        h_out, hidden_next = self.gru(x, hidden)
        return h_out.squeeze(1), hidden_next


__all__ = ["JointWorldModel", "JointWorldModelConfig"]

