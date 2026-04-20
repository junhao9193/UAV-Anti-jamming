from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mpdqn.qmix.mixer import QMIXMixer


class QPLEXMixer(nn.Module):
    """QPLEX-style duplex dueling mixer adapted to hybrid-action MP-DQN.

    This implementation keeps the monotonic value backbone of QMIX for the joint value
    term and adds a positive, state-action-conditioned weighting over local advantages:

        Q_tot = V_tot(s, q_max) + sum_i lambda_i(s, q, q_max) * (q_i - q_i_max)

    where:
      - q_i      is the chosen local action-value,
      - q_i_max  is the local greedy utility,
      - lambda_i > 0 is produced by an attention-style hypernetwork.

    It is a practical hybrid-action adaptation of the QPLEX idea rather than a strict,
    line-by-line reproduction of the original discrete-only formulation.
    """

    def __init__(
        self,
        n_agents: int,
        global_state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.global_state_dim = int(global_state_dim)
        self.mixing_hidden_dim = int(mixing_hidden_dim)
        self.hypernet_hidden_dim = int(hypernet_hidden_dim)
        self.n_heads = int(n_heads)

        self.value_mixer = QMIXMixer(
            n_agents=self.n_agents,
            global_state_dim=self.global_state_dim,
            mixing_hidden_dim=self.mixing_hidden_dim,
            hypernet_hidden_dim=self.hypernet_hidden_dim,
        )
        self.lambda_net = nn.Sequential(
            nn.Linear(self.global_state_dim + 2 * self.n_agents, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.n_heads * self.n_agents),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        max_agent_qs: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        if max_agent_qs.dim() == 3:
            max_agent_qs = max_agent_qs.squeeze(-1)
        if agent_qs.dim() != 2 or max_agent_qs.dim() != 2:
            raise ValueError(
                f"agent_qs/max_agent_qs must be (B,N) or (B,N,1), got {tuple(agent_qs.shape)} / {tuple(max_agent_qs.shape)}"
            )
        if agent_qs.shape != max_agent_qs.shape:
            raise ValueError(
                f"agent_qs and max_agent_qs must match, got {tuple(agent_qs.shape)} vs {tuple(max_agent_qs.shape)}"
            )

        value_tot = self.value_mixer(max_agent_qs, global_state)  # (B,1)

        features = torch.cat([global_state, agent_qs, max_agent_qs], dim=1)
        lambda_logits = self.lambda_net(features).view(agent_qs.shape[0], self.n_heads, self.n_agents)
        lambda_weights = F.softmax(lambda_logits, dim=-1) * float(self.n_agents)
        lambda_weights = lambda_weights.mean(dim=1)  # (B,N), positive and normalized in expectation

        advantages = agent_qs - max_agent_qs
        adv_tot = torch.sum(lambda_weights * advantages, dim=1, keepdim=True)
        return value_tot + adv_tot


__all__ = ["QPLEXMixer"]
