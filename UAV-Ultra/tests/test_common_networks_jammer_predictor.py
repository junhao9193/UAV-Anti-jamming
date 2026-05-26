"""Stage 8 JP-aware 网络 + JP head 单测（plan §test_plan）。"""

from __future__ import annotations

import pytest
import torch

from src.algorithms.common.networks.jammer_predictor import (
    JammerAwareMPDQNActor,
    JammerAwareMPDQNQNetwork,
    JammerPredictionHead,
)


def test_jammer_prediction_head_state_dict_keys_match_baseline_layout():
    head = JammerPredictionHead(history_len=4, n_channel=6, hidden_dim=64)
    keys = set(head.state_dict().keys())
    assert keys == {
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    }


def test_jammer_prediction_head_forward_shape_and_state_dict_copy():
    torch.manual_seed(0)
    h1 = JammerPredictionHead(history_len=4, n_channel=6, hidden_dim=64)
    h2 = JammerPredictionHead(history_len=4, n_channel=6, hidden_dim=64)
    h2.load_state_dict(h1.state_dict(), strict=True)
    history = torch.randn(8, 4, 6)
    out1 = h1(history)
    out2 = h2(history)
    assert out1.shape == (8, 6)
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-12)


def test_jammer_prediction_head_shape_mismatch_raises():
    head = JammerPredictionHead(history_len=4, n_channel=6, hidden_dim=8)
    with pytest.raises(ValueError, match="shape mismatch"):
        head(torch.zeros(2, 5, 6))   # wrong H
    with pytest.raises(ValueError, match="shape mismatch"):
        head(torch.zeros(2, 4, 5))   # wrong C
    with pytest.raises(ValueError, match="must be"):
        head(torch.zeros(2, 4))      # wrong ndim (1D check first)


def test_jammer_aware_actor_first_linear_input_dim_and_state_dict_copy():
    torch.manual_seed(0)
    a1 = JammerAwareMPDQNActor(state_dim=18, n_actions=4, param_dim=2, n_channel=6, hidden_dim=128)
    a2 = JammerAwareMPDQNActor(state_dim=18, n_actions=4, param_dim=2, n_channel=6, hidden_dim=128)
    # first Linear input = state_dim + n_channel = 24
    assert a1.net[0].in_features == 24
    a2.load_state_dict(a1.state_dict(), strict=True)
    aug = torch.randn(7, 24)
    out1 = a1(aug)
    out2 = a2(aug)
    # Actor output shape: (B, n_actions, param_dim)
    assert out1.shape == (7, 4, 2)
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-12)
    # Sigmoid 末层 → 值在 [0, 1]
    assert float(out1.min()) >= 0.0
    assert float(out1.max()) <= 1.0


def test_jammer_aware_qnetwork_first_linear_input_dim_and_state_dict_copy():
    torch.manual_seed(0)
    q1 = JammerAwareMPDQNQNetwork(
        state_dim=18, n_actions=4, param_dim=2, n_channel=6, hidden_dim=128, q_hidden_dim=128,
    )
    q2 = JammerAwareMPDQNQNetwork(
        state_dim=18, n_actions=4, param_dim=2, n_channel=6, hidden_dim=128, q_hidden_dim=128,
    )
    # state_encoder first Linear input = 18 + 6 = 24
    assert q1.state_encoder[0].in_features == 24
    # q_head first Linear input = hidden + n_actions + param_dim = 128 + 4 + 2 = 134
    assert q1.q_head[0].in_features == 128 + 4 + 2
    q2.load_state_dict(q1.state_dict(), strict=True)
    aug = torch.randn(5, 24)
    params = torch.randn(5, 4, 2)
    out1 = q1(aug, params)
    out2 = q2(aug, params)
    assert out1.shape == (5, 4)
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-12)


def test_jammer_aware_qnetwork_state_dict_keys_contain_state_encoder_and_q_head():
    q = JammerAwareMPDQNQNetwork(state_dim=18, n_actions=4, param_dim=2, n_channel=6, hidden_dim=32)
    keys = set(q.state_dict().keys())
    # baseline 子模块命名一致
    assert any(k.startswith("state_encoder.") for k in keys)
    assert any(k.startswith("q_head.") for k in keys)
    # 没有混淆为 actor 风格的 net.*
    assert not any(k.startswith("net.") for k in keys)
