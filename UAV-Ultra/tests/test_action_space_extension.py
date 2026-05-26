"""动作空间扩展测试：Stage 3.5 policy 模式追加 3 维 mobility delta。

- gauss_markov 模式：``per_ch_param_dim == total_param_dim``。
- policy 模式：``per_ch_param_dim == total_param_dim + 3``；env 内缩放后的 delta 在
  ``[-uav_*_delta_max, +uav_*_delta_max]`` 内。
- 错误维度的 action 应被 ``decompose`` 立刻拒绝（ValueError）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import specs
from src.config.loader import load_env_config
from src.envs import Environ


def test_per_ch_param_dim_gauss_markov():
    cfg = load_env_config()
    assert specs.per_ch_param_dim(cfg) == specs.total_param_dim(cfg)


def test_per_ch_param_dim_policy_adds_three():
    cfg = load_env_config(overrides={
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 0.1,
        "uav_p_delta_max": 0.05,
    })
    assert specs.per_ch_param_dim(cfg) == specs.total_param_dim(cfg) + 3


def test_decompose_rejects_wrong_param_size_in_gauss_markov_mode():
    env = Environ(config={"env_seed": 0})
    env.reset()
    n_ch = env.n_ch
    # 多塞 3 维 → 期望 total_param_dim，实际 +3 → ValueError
    bad_params = np.zeros(env.total_param_dim + 3, dtype=np.float32)
    action = [(0, bad_params) for _ in range(n_ch)]
    with pytest.raises(ValueError, match="action_params size"):
        env.step(action)


def test_decompose_accepts_legacy_tuple_action_in_gauss_markov_mode():
    """baseline/vecenv 兼容：action 可为 (discrete_arr, params_arr)。"""
    env = Environ(config={"env_seed": 0})
    env.reset()
    action = (
        np.zeros(env.n_ch, dtype=np.int64),
        np.zeros((env.n_ch, env.total_param_dim), dtype=np.float32),
    )
    next_state, reward, done, info = env.step(action)
    assert np.asarray(next_state).shape[0] == env.n_ch
    assert np.asarray(reward).shape == (env.n_ch,)
    assert done is False
    assert "jammer_channels_current" in info


def test_step_accepts_return_info_false_for_vecenv_fast_path():
    env = Environ(config={"env_seed": 0})
    env.reset()
    action = (
        np.zeros(env.n_ch, dtype=np.int64),
        np.zeros((env.n_ch, env.total_param_dim), dtype=np.float32),
    )
    _, _, _, info = env.step(action, return_info=False)
    assert info == {}


def test_decompose_caches_scaled_mobility_deltas_in_policy_mode():
    vel_max = 1.5
    dir_max = 0.3
    p_max = 0.07
    env = Environ(config={
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": vel_max,
        "uav_direction_delta_max": dir_max,
        "uav_p_delta_max": p_max,
    })
    env.reset()
    n_ch = env.n_ch
    raw_deltas = np.array([[0.5, -1.0, 0.25]] * n_ch, dtype=np.float32)
    expected_scaled = np.column_stack([
        raw_deltas[:, 0] * vel_max,
        raw_deltas[:, 1] * dir_max,
        raw_deltas[:, 2] * p_max,
    ])

    # 注意：env.step 内部会调 mobility 进而清空 _last_mobility_deltas 的语义不变；
    # 这里只验证 decompose 写入是否正确，因此直接调 action_space.decompose。
    from src.envs.action_space import decompose
    action = [
        (0, np.concatenate([np.zeros(env.total_param_dim, dtype=np.float32), raw_deltas[i]]))
        for i in range(n_ch)
    ]
    decompose(env, action)
    np.testing.assert_allclose(env._last_mobility_deltas, expected_scaled, atol=1e-9)


def test_decompose_clips_raw_delta_to_unit_box_before_scaling():
    env = Environ(config={
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 1.0,
        "uav_p_delta_max": 1.0,
    })
    env.reset()
    n_ch = env.n_ch
    from src.envs.action_space import decompose
    raw = np.array([[2.0, -3.0, 5.0]] * n_ch, dtype=np.float32)  # 越出 [-1, 1]
    action = [
        (0, np.concatenate([np.zeros(env.total_param_dim, dtype=np.float32), raw[i]]))
        for i in range(n_ch)
    ]
    decompose(env, action)
    # 应被 clip 到 [-1, 1] * delta_max = [-1, 1]
    assert env._last_mobility_deltas.min() >= -1.0 - 1e-9
    assert env._last_mobility_deltas.max() <= 1.0 + 1e-9


def test_gauss_markov_does_not_cache_mobility_deltas():
    env = Environ(config={"env_seed": 0})
    env.reset()
    from src.envs.action_space import decompose
    action = [
        (0, np.zeros(env.total_param_dim, dtype=np.float32))
        for _ in range(env.n_ch)
    ]
    decompose(env, action)
    assert env._last_mobility_deltas is None
