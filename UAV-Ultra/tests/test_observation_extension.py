"""观测扩展测试：Stage 3.5 ``observation_include_mobility`` 开关。

- 关 → 每个 CH 观测长度 = ``(n_des + 1) * n_channel`` = 18，与 baseline 一致。
- 开 → 每个 CH 观测长度 = 18 + 6 = 24；末尾 6 维是 CH 自身归一化 (pos_xyz, vel, dir, p)。
"""

from __future__ import annotations

import math

import numpy as np

from src.config import specs
from src.config.loader import load_env_config
from src.envs import Environ


def test_state_last_dim_18_when_mobility_obs_off():
    cfg = load_env_config()
    env = Environ(config={"env_seed": 0})
    state = env.reset()
    state_arr = np.asarray(state)
    assert state_arr.shape == (cfg.n_ch, 18)
    assert specs.state_dim(cfg) == 18


def test_state_last_dim_24_when_mobility_obs_on():
    cfg = load_env_config(overrides={"observation_include_mobility": True})
    env = Environ(config={"env_seed": 0, "observation_include_mobility": True})
    state = env.reset()
    state_arr = np.asarray(state)
    assert state_arr.shape == (cfg.n_ch, 24)
    assert specs.state_dim(cfg) == 24


def test_mobility_feature_block_is_in_unit_range_after_reset():
    """开启 mobility obs 后，末尾 6 维都应在合理归一化范围内：
       - 位置归一化 ∈ [0, 1]
       - 速度 / 2π 角度归一化 ∈ [0, ~1+]（可能略 > 1，因 velocity_max=20 / 20=1.0 是 baseline 上界）
    """
    env = Environ(config={"env_seed": 0, "observation_include_mobility": True})
    state = env.reset()
    state_arr = np.asarray(state)
    mobility_block = state_arr[:, -6:]
    # 位置归一化 ∈ [0, 1]（reset 时位置都在长方体内）
    assert (mobility_block[:, 0] >= 0.0).all()
    assert (mobility_block[:, 0] <= 1.0).all()
    assert (mobility_block[:, 1] >= 0.0).all()
    assert (mobility_block[:, 1] <= 1.0).all()
    assert (mobility_block[:, 2] >= 0.0).all()
    assert (mobility_block[:, 2] <= 1.0).all()
    # direction / p 是 angle/(2π) ∈ [0, 1)
    assert (mobility_block[:, 4] >= 0.0).all()
    assert (mobility_block[:, 4] < 1.0).all()
    assert (mobility_block[:, 5] >= 0.0).all()
    assert (mobility_block[:, 5] < 1.0).all()


def test_baseline_csi_and_sensing_unchanged_when_mobility_obs_on():
    """开启 mobility obs 不应改变前 18 维的 CSI + sensing 段（同 seed 下逐位相等）。"""
    env_off = Environ(config={"env_seed": 0})
    env_on = Environ(config={"env_seed": 0, "observation_include_mobility": True})
    state_off = np.asarray(env_off.reset())
    state_on = np.asarray(env_on.reset())
    np.testing.assert_array_equal(state_off, state_on[:, :18])
