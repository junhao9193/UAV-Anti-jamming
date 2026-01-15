from __future__ import division

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cont_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cont_dim = int(cont_dim)
        self.n_agents = int(n_agents)

        self.base = nn.Sequential(
            nn.Linear(self.obs_dim + self.n_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logits_head = nn.Linear(hidden_dim, self.n_actions)
        self.alpha_beta_head = nn.Linear(hidden_dim, 2 * self.cont_dim)

    def forward(self, obs: torch.Tensor, agent_id_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, agent_id_onehot], dim=-1)
        h = self.base(x)
        logits = self.logits_head(h)
        alpha_beta = self.alpha_beta_head(h)
        alpha_raw, beta_raw = torch.chunk(alpha_beta, 2, dim=-1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        return logits, alpha, beta


class CentralValueNet(nn.Module):
    def __init__(
        self,
        global_state_dim: int,
        n_agents: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.n_agents = int(n_agents)
        self.net = nn.Sequential(
            nn.Linear(self.global_state_dim + self.n_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor, agent_id_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_state, agent_id_onehot], dim=-1)
        return self.net(x).squeeze(-1)


__all__ = ["HybridActor", "CentralValueNet"]

