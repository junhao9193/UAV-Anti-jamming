"""``MPDQNActor`` / ``MPDQNQNetwork`` 与 baseline 的 state_dict 拷贝回归。

Stage 4 plan 通过标准：CPU + float64 + 质数维度 + ``strict=True`` + ``atol=1e-12``。
"""

from __future__ import annotations

import numpy as np
import torch

from src.algorithms.common.networks import MPDQNActor, MPDQNQNetwork


# 质数维度：B=7, state=11, A=5, P=3 —— 错轴会立刻形状错误
B, STATE_DIM, N_ACTIONS, PARAM_DIM = 7, 11, 5, 3


def test_mpdqn_actor_state_dict_copy_regression(baseline_import):
    """新 MPDQNActor 与 baseline MPDQNActor 同 state_dict 后输出逐位等价。"""
    baseline_model_mod = baseline_import("algorithms.mpdqn.model")
    torch.manual_seed(0)

    old = baseline_model_mod.MPDQNActor(STATE_DIM, N_ACTIONS, PARAM_DIM).double().cpu().eval()
    new = MPDQNActor(STATE_DIM, N_ACTIONS, PARAM_DIM).double().cpu().eval()

    # strict=True：子模块属性名必须完全一致（plan locked decision #9）
    new.load_state_dict(old.state_dict(), strict=True)

    state = torch.randn(B, STATE_DIM, dtype=torch.float64)
    with torch.no_grad():
        out_old = old(state)
        out_new = new(state)
    assert out_old.shape == (B, N_ACTIONS, PARAM_DIM)
    assert out_new.shape == out_old.shape
    torch.testing.assert_close(out_new, out_old, rtol=0, atol=1e-12)


def test_mpdqn_qnetwork_state_dict_copy_regression(baseline_import):
    baseline_model_mod = baseline_import("algorithms.mpdqn.model")
    torch.manual_seed(0)

    old = baseline_model_mod.MPDQNQNetwork(STATE_DIM, N_ACTIONS, PARAM_DIM).double().cpu().eval()
    new = MPDQNQNetwork(STATE_DIM, N_ACTIONS, PARAM_DIM).double().cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    state = torch.randn(B, STATE_DIM, dtype=torch.float64)
    params = torch.rand(B, N_ACTIONS, PARAM_DIM, dtype=torch.float64)
    with torch.no_grad():
        out_old = old(state, params)
        out_new = new(state, params)
    assert out_old.shape == (B, N_ACTIONS)
    assert out_new.shape == out_old.shape
    torch.testing.assert_close(out_new, out_old, rtol=0, atol=1e-12)


def test_mpdqn_submodule_attribute_names_match_baseline(baseline_import):
    """plan locked #9：子模块属性名必须保持 baseline 命名。"""
    baseline_model_mod = baseline_import("algorithms.mpdqn.model")
    new_actor = MPDQNActor(STATE_DIM, N_ACTIONS, PARAM_DIM)
    old_actor = baseline_model_mod.MPDQNActor(STATE_DIM, N_ACTIONS, PARAM_DIM)
    assert set(new_actor.state_dict().keys()) == set(old_actor.state_dict().keys())

    new_q = MPDQNQNetwork(STATE_DIM, N_ACTIONS, PARAM_DIM)
    old_q = baseline_model_mod.MPDQNQNetwork(STATE_DIM, N_ACTIONS, PARAM_DIM)
    assert set(new_q.state_dict().keys()) == set(old_q.state_dict().keys())
