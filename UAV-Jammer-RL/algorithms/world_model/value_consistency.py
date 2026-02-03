from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from algorithms.world_model.model import JointWorldModel


@dataclass(frozen=True)
class TDlambdaConfig:
    gamma: float = 0.99
    lam: float = 0.8
    rollout_k: int = 4


def td_lambda_truncated(returns_n: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Truncated TD(lambda) mixture.

    Args:
        returns_n: (B, K) where returns_n[:, n-1] = G^(n)
    Returns:
        g_lambda: (B, 1)
    """
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
    # Context up to current time t (last element is s_t / u_t).
    state_seq: torch.Tensor,  # (B,L,Ds)
    action_seq: torch.Tensor,  # (B,L,Du)
    # Policy for u*(s): returns (u_enc, action_discrete, action_params_flat)
    policy_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    # Target Q_tot(s,u): returns (B,1)
    q_tot_target_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: TDlambdaConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build TD(lambda) return Ĝ_t^{λ,K} from GRU world model rollouts.

    This matches the doc semantics:
      - Start from real context ending at (s_t, u_t)
      - Use the model to rollout forward K steps
      - Bootstrap with target Q_tot at imagined states

    Returns:
        g_lambda: (B,1)
        rewards_hat: (B,K) predicted rewards during rollouts (for diagnostics)
    """
    gamma = float(cfg.gamma)
    lam = float(cfg.lam)
    k = int(cfg.rollout_k)
    if k <= 0:
        raise ValueError("rollout_k must be positive")

    state_seq = wm._ensure_seq(state_seq, wm.state_dim, "state_seq")
    action_seq = wm._ensure_seq(action_seq, wm.action_dim, "action_seq")

    # Encode context to get hidden at time t (after consuming x_t=[s_t,u_t]).
    h_seq, h_all = wm.encode(state_seq=state_seq, action_seq=action_seq)
    s_curr = state_seq[:, -1, :]  # (B,Ds)
    h_out = h_seq[:, -1, :]  # (B,H) last-layer output at time t

    rewards_list = []
    bootstrap_list = []

    for _ in range(k):
        # Predict reward r̂_t and next state ŝ_{t+1} from h know-how.
        s_next, r_hat = wm.predict_from_hidden(state=s_curr, h_out=h_out)  # (B,Ds), (B,1)
        rewards_list.append(r_hat.squeeze(1))

        # u*(ŝ_{t+1}) and Q_tot target (no grad through policy; allow grad to ŝ through Q).
        with torch.no_grad():
            u_next_enc, a_next, p_next = policy_fn(s_next)
        q_next = q_tot_target_fn(s_next, a_next, p_next)  # (B,1)
        bootstrap_list.append(q_next.squeeze(1))

        # Advance recurrent hidden by consuming x_{t+1}=[ŝ_{t+1}, u_{t+1}].
        h_out, h_all = wm.advance_hidden(next_state=s_next, next_action=u_next_enc, hidden=h_all)
        s_curr = s_next

    rewards_hat = torch.stack(rewards_list, dim=1)  # (B,K)
    bootstrap = torch.stack(bootstrap_list, dim=1)  # (B,K)

    # Build n-step returns G^(n) for n=1..K
    bsz = int(rewards_hat.shape[0])
    returns_n = []
    for n in range(1, k + 1):
        r_sum = torch.zeros((bsz,), dtype=rewards_hat.dtype, device=rewards_hat.device)
        for j in range(n):
            r_sum = r_sum + (gamma**j) * rewards_hat[:, j]
        g_n = r_sum + (gamma**n) * bootstrap[:, n - 1]
        returns_n.append(g_n)

    returns_n = torch.stack(returns_n, dim=1)  # (B,K)
    g_lambda = td_lambda_truncated(returns_n, lam=lam)  # (B,1)
    return g_lambda, rewards_hat


__all__ = ["TDlambdaConfig", "rollout_td_lambda_return", "td_lambda_truncated"]

