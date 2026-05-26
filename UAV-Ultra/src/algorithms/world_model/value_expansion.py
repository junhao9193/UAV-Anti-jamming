"""TD(λ) 模型想象 rollout（baseline ``value_consistency.py`` 改名）。

供 Stage 5 ``wm_alternating`` callback 与 QMIX value-expansion 训练循环消费。
**世界模型 rollout，非环境 rollout** —— 在 RSSM 潜状态中前向 ``rollout_k`` 步，
基于 q_tot_target 函数 bootstrap 一个 TD(λ) 回报。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from src.algorithms.world_model.model import JointWorldModel


@dataclass(frozen=True)
class TDlambdaConfig:
    gamma: float = 0.99
    lam: float = 0.8
    rollout_k: int = 4


def td_lambda_truncated(returns_n: torch.Tensor, lam: float) -> torch.Tensor:
    """从 ``(B, K)`` 形式的 n-step returns 计算截断 TD(λ)。"""
    if returns_n.ndim != 2:
        raise ValueError(f"returns_n must be (B,K), got {tuple(returns_n.shape)}")

    bsz, k = returns_n.shape
    if k <= 0:
        raise ValueError("K must be positive")

    lam = float(lam)
    weights = torch.zeros((k,), dtype=returns_n.dtype, device=returns_n.device)
    if k == 1:
        weights[0] = 1.0
    else:
        for n in range(1, k):
            weights[n - 1] = (1.0 - lam) * (lam ** (n - 1))
        weights[k - 1] = lam ** (k - 1)

    g = (returns_n * weights.view(1, -1)).sum(dim=1, keepdim=True)
    return g


def rollout_td_lambda_return(
    *,
    wm: JointWorldModel,
    state_seq: torch.Tensor,
    action_seq: torch.Tensor,
    policy_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_tot_target_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: TDlambdaConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从真实上下文末端开始，前向 ``rollout_k`` 步模型想象 rollout，构造 TD(λ) 回报。

    与 baseline ``value_consistency.py:39-95`` 1:1。
    """
    gamma = float(cfg.gamma)
    lam = float(cfg.lam)
    k = int(cfg.rollout_k)
    if k <= 0:
        raise ValueError("rollout_k must be positive")

    state_seq = wm._ensure_seq(state_seq, wm.state_dim, "state_seq")
    action_seq = wm._ensure_seq(action_seq, wm.action_dim, "action_seq")

    _, hidden = wm.encode(state_seq=state_seq, action_seq=action_seq)
    s_curr = state_seq[:, -1, :]

    rewards_list = []
    bootstrap_list = []

    for _ in range(k):
        s_next, r_hat = wm.predict_from_hidden(state=s_curr, hidden=hidden)
        rewards_list.append(r_hat.squeeze(1))

        with torch.no_grad():
            u_next_enc, a_next, p_next = policy_fn(s_next)
        q_next = q_tot_target_fn(s_next, a_next, p_next)
        bootstrap_list.append(q_next.squeeze(1))

        _, hidden = wm.advance_hidden(next_action=u_next_enc, hidden=hidden, sample=False)
        s_curr = s_next

    rewards_hat = torch.stack(rewards_list, dim=1)
    bootstrap = torch.stack(bootstrap_list, dim=1)

    bsz = int(rewards_hat.shape[0])
    returns_n = []
    for n in range(1, k + 1):
        r_sum = torch.zeros((bsz,), dtype=rewards_hat.dtype, device=rewards_hat.device)
        for j in range(n):
            r_sum = r_sum + (gamma**j) * rewards_hat[:, j]
        g_n = r_sum + (gamma**n) * bootstrap[:, n - 1]
        returns_n.append(g_n)

    returns_n = torch.stack(returns_n, dim=1)
    g_lambda = td_lambda_truncated(returns_n, lam=lam)
    return g_lambda, rewards_hat


__all__ = ["TDlambdaConfig", "rollout_td_lambda_return", "td_lambda_truncated"]
