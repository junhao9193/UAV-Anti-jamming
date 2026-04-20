from __future__ import division

import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    """Value Decomposition Network mixer: Q_tot = sum_i Q_i.

    The global state is accepted for interface compatibility with QMIX-style trainers,
    but is intentionally ignored.
    """

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor | None = None) -> torch.Tensor:
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        if agent_qs.dim() != 2:
            raise ValueError(f"agent_qs must be (B,N) or (B,N,1), got shape={tuple(agent_qs.shape)}")
        return agent_qs.sum(dim=1, keepdim=True)


__all__ = ["VDNMixer"]
