"""World-model loss atoms: baseline SmoothL1 + weighted total regression."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.algorithms.world_model import (
    RSSMHiddenState,
    RSSMObserveOutput,
    compute_wm_losses,
    kl_loss,
    reward_loss,
    state_delta_loss,
)


def test_state_and_reward_losses_use_baseline_smooth_l1():
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    delta_pred = torch.tensor([[[2.0, 0.5]]])

    reward_pred = torch.tensor([[[-3.0]]])
    reward = torch.zeros(1, 1, 1)

    torch.testing.assert_close(
        state_delta_loss(delta_pred, state, next_state),
        F.smooth_l1_loss(delta_pred, next_state - state, beta=1.0),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        reward_loss(reward_pred, reward),
        F.smooth_l1_loss(reward_pred, reward, beta=1.0),
        rtol=0,
        atol=1e-12,
    )


def test_kl_loss_matches_closed_form_with_free_nats():
    post_mean = torch.zeros(2, 1, 3)
    prior_mean = torch.zeros(2, 1, 3)
    post_std = torch.ones(2, 1, 3)
    prior_std = torch.ones(2, 1, 3)

    out = kl_loss(post_mean, post_std, prior_mean, prior_std, free_nats=1.0)
    torch.testing.assert_close(out, torch.tensor(1.0), rtol=0, atol=1e-12)


def test_compute_wm_losses_uses_baseline_weighted_total():
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    reward = torch.zeros(1, 1, 1)
    out = RSSMObserveOutput(
        delta_seq=torch.tensor([[[2.0, 0.5]]]),
        reward_seq=torch.tensor([[[-3.0]]]),
        feature_seq=torch.zeros(1, 1, 4),
        hidden=RSSMHiddenState(deter=torch.zeros(1, 1, 2), stoch=torch.zeros(1, 2)),
        prior_mean_seq=torch.zeros(1, 1, 2),
        prior_std_seq=torch.ones(1, 1, 2),
        post_mean_seq=torch.zeros(1, 1, 2),
        post_std_seq=torch.ones(1, 1, 2),
    )

    details, total = compute_wm_losses(
        out,
        state_seq=state,
        next_state_seq=next_state,
        reward_seq=reward,
        alpha=2.0,
        kl_beta=0.25,
        free_nats=1.0,
    )

    expected_total = details["L_S"] + 2.0 * details["L_R"] + 0.25 * details["L_KL"]
    torch.testing.assert_close(total, expected_total, rtol=0, atol=1e-12)


def _make_dummy_rssm_output(batch=1, seq=1, state_dim=2, feat_dim=4):
    return RSSMObserveOutput(
        delta_seq=torch.zeros(batch, seq, state_dim),
        reward_seq=torch.zeros(batch, seq, 1),
        feature_seq=torch.zeros(batch, seq, feat_dim),
        hidden=RSSMHiddenState(deter=torch.zeros(batch, seq, state_dim), stoch=torch.zeros(batch, state_dim)),
        prior_mean_seq=torch.zeros(batch, seq, state_dim),
        prior_std_seq=torch.ones(batch, seq, state_dim),
        post_mean_seq=torch.zeros(batch, seq, state_dim),
        post_std_seq=torch.ones(batch, seq, state_dim),
    )


def test_compute_wm_losses_eta_zero_is_field_identical_to_no_vc_path():
    """L_VC kwarg 在 eta=0 时 total/details 必须与不传 q_teacher/g_lambda 完全相等。"""
    out = _make_dummy_rssm_output()
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    reward = torch.zeros(1, 1, 1)
    q_teacher = torch.tensor([[1.0]])
    g_lambda = torch.tensor([[5.0]])

    d1, t1 = compute_wm_losses(out, state_seq=state, next_state_seq=next_state, reward_seq=reward)
    d2, t2 = compute_wm_losses(
        out, state_seq=state, next_state_seq=next_state, reward_seq=reward,
        q_teacher=q_teacher, g_lambda=g_lambda, eta=0.0,
    )
    torch.testing.assert_close(t1, t2, rtol=0, atol=1e-12)
    assert set(d1) == set(d2) == {"L_S", "L_R", "L_KL"}


def test_compute_wm_losses_eta_positive_includes_smooth_l1_beta10():
    out = _make_dummy_rssm_output()
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    reward = torch.zeros(1, 1, 1)
    q_teacher = torch.tensor([[0.0]])
    g_lambda = torch.tensor([[3.0]])
    eta = 0.5

    details, total = compute_wm_losses(
        out, state_seq=state, next_state_seq=next_state, reward_seq=reward,
        q_teacher=q_teacher, g_lambda=g_lambda, eta=eta,
    )
    expected_l_vc = F.smooth_l1_loss(q_teacher, g_lambda, beta=10.0)
    torch.testing.assert_close(details["L_VC"], expected_l_vc, rtol=0, atol=1e-12)
    base = details["L_S"] + details["L_R"] + 0.1 * details["L_KL"]
    expected_total = base + eta * expected_l_vc
    torch.testing.assert_close(total, expected_total, rtol=0, atol=1e-12)


def test_compute_wm_losses_l_vc_shape_mismatch_raises():
    out = _make_dummy_rssm_output()
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    reward = torch.zeros(1, 1, 1)
    bad_q = torch.zeros((2, 1))
    bad_g = torch.zeros((1, 1))
    with pytest.raises(ValueError, match="q_teacher .* must match g_lambda"):
        compute_wm_losses(
            out, state_seq=state, next_state_seq=next_state, reward_seq=reward,
            q_teacher=bad_q, g_lambda=bad_g, eta=0.5,
        )


def test_compute_wm_losses_l_vc_gradient_flows_to_g_lambda():
    """关键梯度边界回归：g_lambda 必须保留梯度路径，否则 WM 训不到。"""
    out = _make_dummy_rssm_output()
    state = torch.zeros(1, 1, 2)
    next_state = torch.zeros(1, 1, 2)
    reward = torch.zeros(1, 1, 1)
    p = torch.tensor([2.0], requires_grad=True)
    g_lambda = (p * 3.0).reshape(1, 1)   # g_lambda 通过 p 求梯度
    q_teacher = torch.zeros((1, 1))
    _, total = compute_wm_losses(
        out, state_seq=state, next_state_seq=next_state, reward_seq=reward,
        q_teacher=q_teacher, g_lambda=g_lambda, eta=1.0,
    )
    total.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert not torch.allclose(p.grad, torch.zeros_like(p.grad))
