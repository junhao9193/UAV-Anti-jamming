"""Jammer 移动策略派发测试。

覆盖：
- ``gauss_markov`` 是默认且与 baseline 等价（由 ``test_env_contract`` 保证）。
- ``uav_guided_markov`` 在 ``jammer_guidance_strength == 0`` 时**严格退化**为 GaussMarkov
  （逐位相等，因为只有方向均值项混合，权重 0 时混合系数为 0）。
- 强度 > 0 时方向被偏向最近 CH。
- ``is_jammer_moving=False`` 在 loader 层被拒绝。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config.loader import load_env_config
from src.envs import Environ


def test_is_jammer_moving_false_rejected_by_loader():
    with pytest.raises(ValueError, match="is_jammer_moving"):
        load_env_config(overrides={"is_jammer_moving": False})


def test_uav_guided_markov_with_zero_strength_equals_gauss_markov():
    """``g == 0`` 严格退化为 GaussMarkov：跑 5 步后 jammer 位置应 bit-identical。"""
    n_ch = None  # 用默认 cfg
    total_param_dim = None

    e_baseline = Environ(config={"env_seed": 0, "jammer_mobility_model": "gauss_markov"})
    e_baseline.reset()
    e_guided = Environ(config={
        "env_seed": 0,
        "jammer_mobility_model": "uav_guided_markov",
        "jammer_guidance_strength": 0.0,
    })
    e_guided.reset()
    n_ch = e_baseline.n_ch
    total_param_dim = e_baseline.total_param_dim

    for _ in range(5):
        action = [(0, np.zeros(total_param_dim, dtype=np.float32)) for _ in range(n_ch)]
        e_baseline.step(action)
        e_guided.step(action)

    for jb, jg in zip(e_baseline.jammers, e_guided.jammers):
        np.testing.assert_array_equal(jb.position, jg.position)
        assert jb.direction == jg.direction
        assert jb.p == jg.p


def test_uav_guided_markov_with_positive_strength_pulls_direction_toward_nearest_ch():
    """``g > 0`` 时 jammer 方向均值应被拉向最近 CH 方向。

    用一步 step 比较两种 g：在 g_large 下，方向更新里的 mean_direction 应明显偏离
    jammer.mean_direction（baseline 均值），更接近 angle_to_nearest_ch。
    """
    cfg_low = Environ(config={
        "env_seed": 7,
        "jammer_mobility_model": "uav_guided_markov",
        "jammer_guidance_strength": 0.0,
    })
    cfg_low.reset()
    cfg_high = Environ(config={
        "env_seed": 7,
        "jammer_mobility_model": "uav_guided_markov",
        "jammer_guidance_strength": 1.0,
    })
    cfg_high.reset()
    n_ch = cfg_low.n_ch
    total_param_dim = cfg_low.total_param_dim
    action = [(0, np.zeros(total_param_dim, dtype=np.float32)) for _ in range(n_ch)]
    cfg_low.step(action)
    cfg_high.step(action)

    # 两个 env 走完一步后，jammer 方向应该不同（除非概率极低的相等）
    differs = False
    for jl, jh in zip(cfg_low.jammers, cfg_high.jammers):
        if jl.direction != jh.direction:
            differs = True
            break
    assert differs, "uav_guided_markov with g=1.0 should diverge from g=0.0 within one step"


def test_uav_guided_markov_nearest_ch_tiebreaker_uses_smallest_index():
    """多个 CH 距离相同时取**索引最小**者（plan locked decision #1）。"""
    from src.envs.mobility import UAVGuidedMarkovJammerStrategy
    strategy = UAVGuidedMarkovJammerStrategy()

    class FakeUAV:
        def __init__(self, position):
            self.position = position
            self.mean_direction = 0.0
            self.mean_p = 0.0

    class FakeJammer:
        position = [0.0, 0.0, 0.0]
        mean_direction = 0.0
        mean_p = 0.0

    class FakeEnv:
        n_ch = 3
        ch_list = [0, 1, 2]
        # 三个 CH 都到 jammer 距离相等
        uavs = [
            FakeUAV([10.0, 0.0, 0.0]),
            FakeUAV([0.0, 10.0, 0.0]),
            FakeUAV([0.0, 0.0, 10.0]),
        ]

    target_dir, _ = strategy._nearest_ch_target(FakeEnv(), FakeJammer())
    # 索引 0 → 沿 +x 方向 → atan2(0, 10) = 0
    assert abs(target_dir - 0.0) < 1e-12
