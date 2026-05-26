"""TD(λ) rollout 公式回归 + 与 baseline ``rollout_td_lambda_return`` 等价性。"""

from __future__ import annotations

import torch

from src.algorithms.world_model import (
    JointWorldModel,
    JointWorldModelConfig,
    TDlambdaConfig,
    rollout_td_lambda_return,
    td_lambda_truncated,
)


B, L, S, A = 4, 2, 5, 7
HIDDEN, STOCH = 12, 6


def test_td_lambda_truncated_handles_single_step():
    """K=1 时权重 [1.0]。"""
    returns = torch.tensor([[10.0], [20.0]], dtype=torch.float64)
    out = td_lambda_truncated(returns, lam=0.8)
    assert out.shape == (2, 1)
    torch.testing.assert_close(out, returns, rtol=0, atol=1e-12)


def test_td_lambda_truncated_weights_sum_to_one_under_unit_returns():
    """权重之和 = 1（对常值 returns 输出 == const）。"""
    K = 4
    returns = torch.ones(3, K, dtype=torch.float64)
    out = td_lambda_truncated(returns, lam=0.7)
    torch.testing.assert_close(out, torch.ones(3, 1, dtype=torch.float64), rtol=0, atol=1e-9)


def test_rollout_td_lambda_return_matches_baseline(baseline_import):
    """与 baseline 1:1（float32 路径，``init_hidden`` 硬编码）。"""
    baseline_model_mod = baseline_import("algorithms.world_model.model")
    baseline_vc_mod = baseline_import("algorithms.world_model.value_consistency")
    torch.manual_seed(0)

    bcfg = baseline_model_mod.JointWorldModelConfig(
        state_dim=S, action_dim=A, hidden_dim=HIDDEN, n_layers=1,
        stochastic_dim=STOCH,
    )
    old_wm = baseline_model_mod.JointWorldModel(bcfg).cpu().eval()
    new_wm = JointWorldModel(
        JointWorldModelConfig(state_dim=S, action_dim=A, hidden_dim=HIDDEN, stochastic_dim=STOCH)
    ).cpu().eval()
    new_wm.load_state_dict(old_wm.state_dict(), strict=True)

    state_seq = torch.randn(B, L, S)
    action_seq = torch.randn(B, L, A)

    def _policy_fn(s):
        u_enc = torch.zeros(s.shape[0], A, dtype=torch.float32)
        a_disc = torch.zeros(s.shape[0], 1, dtype=torch.long)
        p = torch.zeros(s.shape[0], 1, 1, dtype=torch.float32)
        return u_enc, a_disc, p

    def _q_tot_target_fn(s, a, p):
        return s.mean(dim=1, keepdim=True)

    base_cfg = baseline_vc_mod.TDlambdaConfig(gamma=0.95, lam=0.7, rollout_k=3)
    new_cfg = TDlambdaConfig(gamma=0.95, lam=0.7, rollout_k=3)

    with torch.no_grad():
        g_old, r_old = baseline_vc_mod.rollout_td_lambda_return(
            wm=old_wm, state_seq=state_seq, action_seq=action_seq,
            policy_fn=_policy_fn, q_tot_target_fn=_q_tot_target_fn, cfg=base_cfg,
        )
        g_new, r_new = rollout_td_lambda_return(
            wm=new_wm, state_seq=state_seq, action_seq=action_seq,
            policy_fn=_policy_fn, q_tot_target_fn=_q_tot_target_fn, cfg=new_cfg,
        )
    torch.testing.assert_close(g_new, g_old, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(r_new, r_old, rtol=1e-6, atol=1e-6)
