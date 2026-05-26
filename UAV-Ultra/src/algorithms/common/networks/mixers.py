"""值分解 mixer：VDN / QMIX / QPLEX 三套（baseline 三个 mixer 文件合并）。

设计约束（Stage 4 plan locked #9）：保留 baseline 子模块属性名（``hyper_w1`` / ``hyper_b1``
/ ``hyper_w2`` / ``hyper_b2`` / ``value_mixer`` / ``lambda_net``），``strict=True``
load_state_dict 回归测试是基石。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VDNMixer(nn.Module):
    """VDN：Q_tot = sum_i Q_i；无可学习参数。``global_state`` 仅为接口兼容性，被忽略。"""

    def forward(
        self,
        agent_qs: torch.Tensor,
        global_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        if agent_qs.dim() != 2:
            raise ValueError(
                f"agent_qs must be (B,N) or (B,N,1), got shape={tuple(agent_qs.shape)}"
            )
        return agent_qs.sum(dim=1, keepdim=True)


class QMIXMixer(nn.Module):
    """QMIX 单调 mixer：hypernet 输出非负权重保 monotonicity（``torch.abs``）。"""

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
        self.hypernet_hidden_dim = int(hypernet_hidden_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.global_state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.n_agents * self.mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(self.global_state_dim, self.mixing_hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.global_state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, self.mixing_hidden_dim * 1),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.global_state_dim, self.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        if agent_qs.dim() != 2:
            raise ValueError(
                f"agent_qs must be (B,N) or (B,N,1), got shape={tuple(agent_qs.shape)}"
            )

        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)  # (B,1,N)

        w1 = torch.abs(self.hyper_w1(global_state)).view(
            batch_size, self.n_agents, self.mixing_hidden_dim
        )  # (B,N,H)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.mixing_hidden_dim)  # (B,1,H)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B,1,H)

        w2 = torch.abs(self.hyper_w2(global_state)).view(
            batch_size, self.mixing_hidden_dim, 1
        )  # (B,H,1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)  # (B,1,1)

        q_tot = torch.bmm(hidden, w2) + b2  # (B,1,1)
        return q_tot.view(batch_size, 1)


class QPLEXMixer(nn.Module):
    """QPLEX 双重 dueling mixer：QMIX 单调 value backbone + 正权重 advantage 加权。

    Q_tot = V_tot(s, q_max) + sum_i lambda_i(s, q, q_max) * (q_i - q_i_max)

    其中 lambda_i > 0 由多头 attention 风格的 hypernet 生成。复用 ``QMIXMixer``
    作为 value backbone（``self.value_mixer``）。
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
                "agent_qs/max_agent_qs must be (B,N) or (B,N,1), got "
                f"{tuple(agent_qs.shape)} / {tuple(max_agent_qs.shape)}"
            )
        if agent_qs.shape != max_agent_qs.shape:
            raise ValueError(
                "agent_qs and max_agent_qs must match, got "
                f"{tuple(agent_qs.shape)} vs {tuple(max_agent_qs.shape)}"
            )

        value_tot = self.value_mixer(max_agent_qs, global_state)  # (B,1)

        features = torch.cat([global_state, agent_qs, max_agent_qs], dim=1)
        lambda_logits = self.lambda_net(features).view(
            agent_qs.shape[0], self.n_heads, self.n_agents
        )
        lambda_weights = F.softmax(lambda_logits, dim=-1) * float(self.n_agents)
        lambda_weights = lambda_weights.mean(dim=1)  # (B,N)

        advantages = agent_qs - max_agent_qs
        adv_tot = torch.sum(lambda_weights * advantages, dim=1, keepdim=True)
        return value_tot + adv_tot


__all__ = ["VDNMixer", "QMIXMixer", "QPLEXMixer"]
