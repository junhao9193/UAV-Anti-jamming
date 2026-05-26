"""世界模型想象 rollout 辅助（**模型想象 rollout，非环境 rollout**）。

提供 K 步前向 imagine 的纯函数 helper，供 Stage 5 ``wm_alternating`` callback 用于
策略训练的 imagined trajectories。``value_expansion.py`` 也消费同一抽象。
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch

from src.algorithms.world_model.model import JointWorldModel, RSSMHiddenState


def imagine_rollout(
    *,
    wm: JointWorldModel,
    state_seq: torch.Tensor,
    action_seq: torch.Tensor,
    policy_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    rollout_k: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """从真实上下文末端编码 hidden，再前向 ``rollout_k`` 步想象。

    Returns:
        List of length ``rollout_k``，每项 ``(s_next, r_hat, u_next_enc, a_next, p_next)``。
    """
    rollout_k = int(rollout_k)
    if rollout_k <= 0:
        raise ValueError("rollout_k must be positive")

    state_seq = wm._ensure_seq(state_seq, wm.state_dim, "state_seq")
    action_seq = wm._ensure_seq(action_seq, wm.action_dim, "action_seq")

    _, hidden = wm.encode(state_seq=state_seq, action_seq=action_seq)
    s_curr = state_seq[:, -1, :]

    steps: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(rollout_k):
        s_next, r_hat = wm.predict_from_hidden(state=s_curr, hidden=hidden)
        with torch.no_grad():
            u_next_enc, a_next, p_next = policy_fn(s_next)
        _, hidden = wm.advance_hidden(next_action=u_next_enc, hidden=hidden, sample=False)
        steps.append((s_next, r_hat, u_next_enc, a_next, p_next))
        s_curr = s_next
    return steps


__all__ = ["imagine_rollout"]
