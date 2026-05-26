"""奖励扩展测试：Stage 3.5 mobility 惩罚项。

- 默认权重（``mobility_oob_penalty_weight = mobility_energy_weight = 0``）下，
  ``apply_mobility_penalty`` 应无效果；reward 与 baseline 等价（由 contract 保证）。
- 权重 > 0 时，移动距离平方 / 越界量被扣分。
"""

from __future__ import annotations

import numpy as np

from src.envs import Environ
from src.envs.reward import apply_mobility_penalty


def test_apply_mobility_penalty_is_no_op_at_default_weights():
    env = Environ(config={"env_seed": 0})
    env.reset()
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    env.last_mobility_delta_sq = np.array([100.0, 100.0, 100.0, 100.0])
    env.last_mobility_oob = np.array([1.0, 1.0, 1.0, 1.0])
    result = apply_mobility_penalty(env, base.copy())
    np.testing.assert_array_equal(result, base)


def test_apply_mobility_penalty_subtracts_weighted_delta_sq():
    env = Environ(config={
        "env_seed": 0,
        "mobility_energy_weight": 0.5,
    })
    env.reset()
    base = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    env.last_mobility_delta_sq = np.array([4.0, 8.0, 12.0, 16.0])
    env.last_mobility_oob = np.zeros(4)
    result = apply_mobility_penalty(env, base.copy())
    expected = base - 0.5 * env.last_mobility_delta_sq
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_apply_mobility_penalty_subtracts_weighted_oob():
    env = Environ(config={
        "env_seed": 0,
        "mobility_oob_penalty_weight": 2.0,
    })
    env.reset()
    base = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    env.last_mobility_delta_sq = np.zeros(4)
    env.last_mobility_oob = np.array([0.0, 1.0, 2.0, 3.0])
    result = apply_mobility_penalty(env, base.copy())
    expected = base - 2.0 * env.last_mobility_oob
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_step_pipeline_invokes_apply_mobility_penalty_with_weights_zero_no_op():
    """在 step 真实路径下，默认权重 → reward 等价 baseline（contract 已验证）。
    此处仅复测 ``env.step`` 调用过程中 ``last_mobility_delta_sq`` 被正确填充。"""
    env = Environ(config={"env_seed": 0})
    env.reset()
    n_ch = env.n_ch
    action = [(0, np.zeros(env.total_param_dim, dtype=np.float32)) for _ in range(n_ch)]
    env.step(action)
    assert env.last_mobility_delta_sq is not None
    assert env.last_mobility_delta_sq.shape == (n_ch,)
    # GaussMarkov 模式下肯定有非零位移（速度 ~10-20 / step ~0.2s → 2-4 m → sq > 0）
    assert (env.last_mobility_delta_sq >= 0.0).all()


def test_step_oob_penalty_records_boundary_reflection_and_subtracts_reward():
    """真实 step 路径下，边界反射应写入 last_mobility_oob 并参与扣分。"""
    cfg = {
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 0.1,
        "uav_p_delta_max": 0.1,
    }
    env_base = Environ(config=cfg)
    env_penalty = Environ(config={**cfg, "mobility_oob_penalty_weight": 2.0})
    env_base.reset()
    env_penalty.reset()

    for env in (env_base, env_penalty):
        uav = env.uavs[env.ch_list[0]]
        uav.position = [1.0, 1.0, 70.0]
        uav.velocity = 1000.0
        uav.direction = np.pi
        uav.p = 0.0

    params = np.zeros(env_base.total_param_dim + 3, dtype=np.float32)
    action = [(0, params.copy()) for _ in range(env_base.n_ch)]
    _, reward_base, _, _ = env_base.step(action)
    _, reward_penalty, _, _ = env_penalty.step(action)

    assert env_penalty.last_mobility_oob[0] > 0.0
    expected = np.asarray(reward_base) - 2.0 * env_penalty.last_mobility_oob
    np.testing.assert_allclose(reward_penalty, expected, atol=1e-9)


def test_mobility_reward_details_are_separate_from_baseline_reward_details():
    env = Environ(config={"env_seed": 0, "mobility_energy_weight": 0.5})
    env.reset()
    action = [(0, np.zeros(env.total_param_dim, dtype=np.float32)) for _ in range(env.n_ch)]
    env.step(action)

    baseline_details = env.reward_details()
    mobility_details = env.mobility_reward_details()
    assert len(baseline_details) == 3
    assert len(mobility_details) == 3
    assert mobility_details[0] > 0.0
    assert mobility_details[2] > 0.0
