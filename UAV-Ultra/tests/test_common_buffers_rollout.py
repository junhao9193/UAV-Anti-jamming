"""``compute_gae`` 公式回归（plan 通过标准：手算 3-step GAE 验证）。"""

from __future__ import annotations

import numpy as np

from src.algorithms.common.buffers import compute_gae


def test_compute_gae_3step_hand_calculation_single_agent():
    """T=3, n_agents=1，手算 GAE 与函数输出一致。"""
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[0.5], [0.8], [1.2]], dtype=np.float32)
    dones = np.zeros((3, 1), dtype=np.float32)
    last_value = np.array([1.5], dtype=np.float32)
    gamma = 0.9
    lam = 0.95

    returns, advantages = compute_gae(
        rewards, values, dones, last_value, gamma=gamma, gae_lambda=lam
    )

    # 手算（与 baseline 公式 1:1）：
    # t=2: next_values=last_value=1.5；delta = 3.0 + 0.9*1.5*1 - 1.2 = 3.15
    #      last_gae = 3.15 + 0.9*0.95*1*0 = 3.15
    # t=1: next_values=values[2]=1.2；delta = 2.0 + 0.9*1.2*1 - 0.8 = 2.28
    #      last_gae = 2.28 + 0.9*0.95*1*3.15 = 2.28 + 2.69325 = 4.97325
    # t=0: next_values=values[1]=0.8；delta = 1.0 + 0.9*0.8*1 - 0.5 = 1.22
    #      last_gae = 1.22 + 0.9*0.95*1*4.97325 = 1.22 + 4.25212875 = 5.47212875
    expected_adv = np.array([[5.47212875], [4.97325], [3.15]], dtype=np.float32)
    expected_ret = expected_adv + values

    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-5, atol=1e-6)


def test_compute_gae_terminal_breaks_bootstrap():
    """done=1 时 next_value 贡献清零。"""
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [0.8]], dtype=np.float32)
    dones = np.array([[0.0], [1.0]], dtype=np.float32)
    last_value = np.array([100.0], dtype=np.float32)

    returns, advantages = compute_gae(
        rewards, values, dones, last_value, gamma=0.9, gae_lambda=0.95
    )
    # t=1: next_non_terminal=0 → delta = 2 + 0 - 0.8 = 1.2；last_gae = 1.2
    # t=0: next_values=values[1]=0.8, next_non_terminal=1; last_gae from t=1 was 1.2 then * 0 (done[1]=1) on transition
    # Actually: next_non_terminal at t=1 was 0, so last_gae_after_t1 = delta_t1 + ... * 0 * last_gae_initial = 1.2
    # Then at t=0: last_gae = (1 + 0.9*0.8*1 - 0.5) + 0.9*0.95*1*1.2 = 1.22 + 1.026 = 2.246
    expected_adv_t1 = 1.2
    expected_adv_t0 = 1.22 + 0.9 * 0.95 * 1.0 * 1.2
    np.testing.assert_allclose(advantages[1, 0], expected_adv_t1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(advantages[0, 0], expected_adv_t0, rtol=1e-5, atol=1e-6)


def test_compute_gae_multi_agent_independent():
    """n_agents=3，每个 agent 独立 GAE。"""
    T, N = 4, 3
    rewards = np.random.RandomState(0).randn(T, N).astype(np.float32)
    values = np.random.RandomState(1).randn(T, N).astype(np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    last_value = np.zeros((N,), dtype=np.float32)

    returns_multi, advs_multi = compute_gae(
        rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95
    )
    assert returns_multi.shape == (T, N)
    assert advs_multi.shape == (T, N)

    # 验证每个 agent 与单 agent 独立计算一致
    for ag in range(N):
        r_a = rewards[:, ag:ag + 1]
        v_a = values[:, ag:ag + 1]
        d_a = dones[:, ag:ag + 1]
        lv_a = last_value[ag:ag + 1]
        ret_a, adv_a = compute_gae(r_a, v_a, d_a, lv_a, gamma=0.99, gae_lambda=0.95)
        np.testing.assert_allclose(advs_multi[:, ag], adv_a[:, 0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(returns_multi[:, ag], ret_a[:, 0], rtol=1e-5, atol=1e-6)
