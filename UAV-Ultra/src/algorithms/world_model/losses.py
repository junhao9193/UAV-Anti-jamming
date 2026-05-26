"""世界模型损失函数（**仅 loss 公式**，不含 optimizer / update step）。

Stage 4 plan locked decision #3：world_model 无独立训练器；本模块提供 loss 计算原子，
Stage 5 callback 装配 update step。

三项核心 loss：
- ``L_S``：state 一步增量预测的 SmoothL1（target 是 ``s_{t+1} - s_t``）。
- ``L_R``：reward 预测的 SmoothL1。
- ``L_KL``：posterior 与 prior 的 KL 散度（带 ``free_nats`` 下界）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.algorithms.world_model.model import RSSMObserveOutput


def state_delta_loss(
    delta_pred_seq: torch.Tensor,
    state_seq: torch.Tensor,
    next_state_seq: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """state 增量 SmoothL1：predict ``delta_pred`` 应匹配 ``next_state - state``。"""
    if delta_pred_seq.shape != state_seq.shape:
        raise ValueError(
            f"delta_pred_seq {tuple(delta_pred_seq.shape)} must match state_seq {tuple(state_seq.shape)}"
        )
    if next_state_seq.shape != state_seq.shape:
        raise ValueError(
            f"next_state_seq {tuple(next_state_seq.shape)} must match state_seq {tuple(state_seq.shape)}"
        )
    delta_target = next_state_seq - state_seq
    return F.smooth_l1_loss(delta_pred_seq, delta_target, beta=float(beta))


def reward_loss(
    reward_pred_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """reward 预测 SmoothL1。统一形状到 ``(B, L, 1)``。"""
    if reward_pred_seq.shape != reward_seq.shape:
        raise ValueError(
            f"reward_pred_seq {tuple(reward_pred_seq.shape)} must match reward_seq {tuple(reward_seq.shape)}"
        )
    return F.smooth_l1_loss(reward_pred_seq, reward_seq, beta=float(beta))


def kl_loss(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    *,
    free_nats: float = 1.0,
) -> torch.Tensor:
    """KL[posterior || prior]，带 free_nats 下界（baseline 公式）。"""
    if post_mean.shape != prior_mean.shape or post_std.shape != prior_std.shape:
        raise ValueError(
            "posterior/prior mean & std shapes must match: "
            f"{tuple(post_mean.shape)} / {tuple(prior_mean.shape)}"
        )
    # 闭式 KL: log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2 sigma_p^2) - 0.5
    log_ratio = torch.log(prior_std + 1e-12) - torch.log(post_std + 1e-12)
    var_ratio = (post_std**2 + (post_mean - prior_mean) ** 2) / (2.0 * (prior_std**2) + 1e-12)
    kl = (log_ratio + var_ratio - 0.5).sum(dim=-1)
    if float(free_nats) > 0.0:
        kl = torch.clamp(kl, min=float(free_nats))
    return kl.mean()


def compute_wm_losses(
    out: RSSMObserveOutput,
    *,
    state_seq: torch.Tensor,
    next_state_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    free_nats: float = 1.0,
    alpha: float = 1.0,
    kl_beta: float = 0.1,
    robust_beta: float = 1.0,
    q_teacher: torch.Tensor | None = None,
    g_lambda: torch.Tensor | None = None,
    eta: float = 0.0,
    vc_robust_beta: float = 10.0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """组合 baseline world-model loss，返回详情字典 + 加权总和。

    返回:
        details: ``{"L_S", "L_R", "L_KL"}`` (+ ``"L_VC"`` 当 ``eta>0`` 且两 tensor 都传入)
        total: ``L_S + alpha * L_R + kl_beta * L_KL`` (+ ``eta * L_VC``)

    L_VC 公式严格对齐 baseline ``world_model/trainer.py:211``::

        L_VC = SmoothL1(q_teacher, g_lambda, beta=vc_robust_beta)

    其中 ``q_teacher`` 与 ``g_lambda`` 都应已在调用方过 ``trainer._clip_value_target``。
    """
    l_s = state_delta_loss(
        out.delta_seq, state_seq, next_state_seq, beta=float(robust_beta)
    )
    l_r = reward_loss(out.reward_seq, reward_seq, beta=float(robust_beta))
    l_kl = kl_loss(
        out.post_mean_seq, out.post_std_seq,
        out.prior_mean_seq, out.prior_std_seq,
        free_nats=float(free_nats),
    )
    total = l_s + float(alpha) * l_r + float(kl_beta) * l_kl
    details: Dict[str, torch.Tensor] = {"L_S": l_s, "L_R": l_r, "L_KL": l_kl}

    if float(eta) > 0.0 and q_teacher is not None and g_lambda is not None:
        if q_teacher.shape != g_lambda.shape:
            raise ValueError(
                f"q_teacher {tuple(q_teacher.shape)} must match g_lambda {tuple(g_lambda.shape)}"
            )
        l_vc = F.smooth_l1_loss(q_teacher, g_lambda, beta=float(vc_robust_beta))
        total = total + float(eta) * l_vc
        details["L_VC"] = l_vc

    return details, total


__all__ = [
    "state_delta_loss",
    "reward_loss",
    "kl_loss",
    "compute_wm_losses",
]
