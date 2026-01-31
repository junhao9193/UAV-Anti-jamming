from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXMixer(nn.Module):
    """
    QMIX mixing network:
      Q_tot = f(q_1, ..., q_n, s_global)
    with monotonicity enforced by non-negative mixing weights produced by hypernetworks.
    """

    def __init__(
        self,
        n_agents: int,
        global_state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.global_state_dim = int(global_state_dim)
        self.mixing_hidden_dim = int(mixing_hidden_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.global_state_dim, int(hypernet_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hypernet_hidden_dim), self.n_agents * self.mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(self.global_state_dim, self.mixing_hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.global_state_dim, int(hypernet_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hypernet_hidden_dim), self.mixing_hidden_dim * 1),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.global_state_dim, int(hypernet_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hypernet_hidden_dim), 1),
        )

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (batch, n_agents) or (batch, n_agents, 1)
            global_state: (batch, global_state_dim)
        Returns:
            q_tot: (batch, 1)
        """
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        if agent_qs.dim() != 2:
            raise ValueError(f"agent_qs must be (B,N) or (B,N,1), got shape={tuple(agent_qs.shape)}")

        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)  # (B,1,N)

        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.n_agents, self.mixing_hidden_dim)  # (B,N,H)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.mixing_hidden_dim)  # (B,1,H)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B,1,H)

        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.mixing_hidden_dim, 1)  # (B,H,1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)  # (B,1,1)

        q_tot = torch.bmm(hidden, w2) + b2  # (B,1,1)
        return q_tot.view(batch_size, 1)


__all__ = ["QMIXMixer"]
