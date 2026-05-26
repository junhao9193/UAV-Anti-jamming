"""``JointWorldModel`` 测试：构造 / shape / observe 输出 / state_dict copy 回归。"""

from __future__ import annotations

import torch

from src.algorithms.world_model import (
    JointWorldModel,
    JointWorldModelConfig,
    RSSMHiddenState,
)


B, L, S, A = 5, 3, 7, 11  # 质数维度
HIDDEN, STOCH = 16, 8


def _make_cfg():
    return JointWorldModelConfig(
        state_dim=S,
        action_dim=A,
        hidden_dim=HIDDEN,
        n_layers=1,
        stochastic_dim=STOCH,
        min_std=0.1,
        kl_beta=0.1,
        free_nats=1.0,
    )


def test_joint_world_model_observe_output_shapes():
    """注：``init_hidden`` 硬编码 float32（baseline 同样），保持 float32 路径。"""
    torch.manual_seed(0)
    wm = JointWorldModel(_make_cfg()).eval()
    state_seq = torch.randn(B, L, S)
    action_seq = torch.randn(B, L, A)
    with torch.no_grad():
        out = wm.observe(state_seq=state_seq, action_seq=action_seq, sample=False)
    assert out.delta_seq.shape == (B, L, S)
    assert out.reward_seq.shape == (B, L, 1)
    assert out.feature_seq.shape == (B, L, HIDDEN + STOCH)
    assert out.prior_mean_seq.shape == (B, L, STOCH)
    assert out.post_mean_seq.shape == (B, L, STOCH)
    assert isinstance(out.hidden, RSSMHiddenState)


def test_joint_world_model_state_dict_copy_regression(baseline_import):
    """与 baseline 同 state_dict + 同输入 → 同输出。float32 路径（``init_hidden`` 硬编码）。"""
    baseline_mod = baseline_import("algorithms.world_model.model")
    torch.manual_seed(0)

    bcfg = baseline_mod.JointWorldModelConfig(
        state_dim=S, action_dim=A, hidden_dim=HIDDEN, n_layers=1,
        stochastic_dim=STOCH, min_std=0.1, kl_beta=0.1, free_nats=1.0,
    )
    old = baseline_mod.JointWorldModel(bcfg).cpu().eval()
    new = JointWorldModel(_make_cfg()).cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    state_seq = torch.randn(B, L, S)
    action_seq = torch.randn(B, L, A)
    with torch.no_grad():
        out_old = old.observe(state_seq=state_seq, action_seq=action_seq, sample=False)
        out_new = new.observe(state_seq=state_seq, action_seq=action_seq, sample=False)
    # float32 路径放宽到 1e-6（baseline 自身浮点噪声）
    torch.testing.assert_close(out_new.delta_seq, out_old.delta_seq, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_new.reward_seq, out_old.reward_seq, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_new.post_mean_seq, out_old.post_mean_seq, rtol=1e-6, atol=1e-6)


def test_joint_world_model_init_hidden_shapes():
    wm = JointWorldModel(_make_cfg())
    h = wm.init_hidden(batch_size=4)
    assert h.deter.shape == (1, 4, HIDDEN)
    assert h.stoch.shape == (4, STOCH)


def test_joint_world_model_predict_from_hidden_clips_state():
    """``predict_from_hidden`` 应该 clip 到 [state_clip_low, state_clip_high]。"""
    torch.manual_seed(0)
    cfg = JointWorldModelConfig(
        state_dim=S, action_dim=A, hidden_dim=HIDDEN, stochastic_dim=STOCH,
        state_clip_low=-1.0, state_clip_high=1.0,
    )
    wm = JointWorldModel(cfg).eval()
    state = torch.ones(B, S) * 5.0  # 极端值，预测后应被 clip
    h = wm.init_hidden(B)
    next_state, _ = wm.predict_from_hidden(state=state, hidden=h)
    assert (next_state >= -1.0).all() and (next_state <= 1.0).all()
