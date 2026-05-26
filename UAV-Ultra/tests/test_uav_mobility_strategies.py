"""UAV CH 移动策略测试：GaussMarkov 与 Policy。

GaussMarkov 走 baseline 路径，等价性由 ``test_env_contract`` 覆盖；这里仅验证：
- 100 步轨迹不会越出边界（边界反射生效）。
- PolicyUAVStrategy 受 delta 严格控制：相同 delta 输入 → 速度/方向/仰角按公式更新。
"""

from __future__ import annotations

import math

import numpy as np

from src.envs import Environ


def _run_steps(env: Environ, steps: int) -> list:
    """喂入零 mobility delta 的合法动作，跑 ``steps`` 步，返回每步 CH 位置历史。"""
    n_ch = env.n_ch
    total_param_dim = env.total_param_dim
    is_policy = env.uav_mobility_control == "policy"
    per_ch = total_param_dim + (3 if is_policy else 0)
    history = []
    for _ in range(steps):
        action = [
            (0, np.zeros(per_ch, dtype=np.float32))
            for _ in range(n_ch)
        ]
        env.step(action)
        history.append([env.uavs[env.ch_list[i]].position[:] for i in range(n_ch)])
    return history


def test_gauss_markov_uav_strategy_respects_world_box():
    """GaussMarkov 100 步轨迹必须落在 ``[0, length] x [0, width] x [low, high]`` 内。"""
    env = Environ(config={"env_seed": 0})
    env.reset()
    positions = _run_steps(env, steps=100)
    for step_positions in positions:
        for pos in step_positions:
            assert 0.0 <= pos[0] <= env.length
            assert 0.0 <= pos[1] <= env.width
            assert env.low_height <= pos[2] <= env.high_height


def test_policy_uav_strategy_only_applies_scaled_action_delta():
    """Policy 模式下，喂入正值 velocity_delta 应使 velocity 严格增大（在 max 内）。"""
    env = Environ(config={
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 0.2,
        "uav_p_delta_max": 0.1,
    })
    env.reset()
    total_param_dim = env.total_param_dim
    n_ch = env.n_ch

    # action 中末尾 3 维 = [+1, 0, 0]，即每步最大正速度增量、方向/仰角不动
    velocities_before = [env.uavs[env.ch_list[i]].velocity for i in range(n_ch)]

    action = []
    for _ in range(n_ch):
        params = np.zeros(total_param_dim + 3, dtype=np.float32)
        params[-3] = 1.0   # velocity_delta = +1.0 * uav_velocity_delta_max
        action.append((0, params))
    env.step(action)

    velocities_after = [env.uavs[env.ch_list[i]].velocity for i in range(n_ch)]
    for v_before, v_after in zip(velocities_before, velocities_after):
        assert v_after >= v_before, "velocity should not decrease under +Δv action"
        assert abs(v_after - (v_before + 1.0)) < 1e-9


def test_policy_uav_strategy_clamps_velocity_at_zero():
    """negative delta 不应让速度变负。"""
    env = Environ(config={
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1000.0,  # 极端大，确保能压到 0
        "uav_direction_delta_max": 0.1,
        "uav_p_delta_max": 0.1,
    })
    env.reset()
    total_param_dim = env.total_param_dim
    n_ch = env.n_ch

    action = []
    for _ in range(n_ch):
        params = np.zeros(total_param_dim + 3, dtype=np.float32)
        params[-3] = -1.0  # 极大负速度增量
        action.append((0, params))
    env.step(action)

    for i in range(n_ch):
        assert env.uavs[env.ch_list[i]].velocity == 0.0


def test_policy_uav_strategy_wraps_direction_modulo_two_pi():
    """direction 在 [0, 2π) 内 wrap。"""
    env = Environ(config={
        "env_seed": 0,
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 0.001,
        "uav_direction_delta_max": 100.0,  # 极大方向增量，验证 wrap
        "uav_p_delta_max": 0.1,
    })
    env.reset()
    total_param_dim = env.total_param_dim
    n_ch = env.n_ch

    action = []
    for _ in range(n_ch):
        params = np.zeros(total_param_dim + 3, dtype=np.float32)
        params[-2] = 1.0
        action.append((0, params))
    env.step(action)

    two_pi = 2.0 * math.pi
    for i in range(n_ch):
        d = env.uavs[env.ch_list[i]].direction
        assert 0.0 <= d < two_pi
