"""动作空间：MP-DQN 参数化动作分解 + Stage 3.5 policy 模式的 3 维 mobility delta。

与 baseline ``UAV-Jammer-RL/envs/core.py:1025-1050`` 1:1 对应；Stage 3.5 时如果
``cfg.uav_mobility_control == "policy"``，则每个 CH 的连续参数末尾追加 3 维
``(velocity_delta, direction_delta, p_delta) ∈ [-1, 1]``。

参数维度公式见 ``src.config.specs.per_ch_param_dim``：
- gauss_markov 模式：``total_param_dim``
- policy 模式：       ``total_param_dim + 3``

PolicyUAVStrategy 应用规则（每 CH，每 step）：
- ``velocity ← max(0, velocity + delta_v * uav_velocity_delta_max)``
- ``direction ← (direction + delta_d * uav_direction_delta_max) mod 2π``
- ``p ← (p + delta_p * uav_p_delta_max) mod 2π``
- 之后沿用 baseline 位置更新 + 边界反射；**不执行**末尾 Gauss-Markov 随机更新。

本模块只负责**解析与缓存**：将 raw delta 缩放后写入 ``env._last_mobility_deltas``，
真正的应用在 ``mobility.PolicyUAVStrategy.update_ch_positions``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.envs.environment import Environ


def decompose(env: "Environ", action) -> None:
    """把 action 列表分解到 ``env.uav_channels`` / ``uav_powers`` / ``uav_jump_count``。

    Action 结构：``[(discrete_action, all_action_params), ...]``，长度 ``n_ch``。
    - gauss_markov 模式：``all_action_params`` 长度 = ``total_param_dim``
    - policy 模式：       长度 = ``total_param_dim + 3``，末尾 3 维为 mobility delta

    policy 模式下缩放后的 mobility delta 缓存在 ``env._last_mobility_deltas``，
    shape ``(n_ch, 3)``；``GaussMarkov`` 模式则置为 ``None``。
    """
    is_policy = env.uav_mobility_control == "policy"
    expected_params = env.total_param_dim + (3 if is_policy else 0)

    mobility_buffer = np.zeros((env.n_ch, 3), dtype=np.float32) if is_policy else None
    vel_max = float(env.uav_velocity_delta_max)
    dir_max = float(env.uav_direction_delta_max)
    p_max = float(env.uav_p_delta_max)

    if (
        isinstance(action, tuple)
        and len(action) == 2
        and isinstance(action[0], np.ndarray)
        and isinstance(action[1], np.ndarray)
    ):
        action_discrete_arr = np.asarray(action[0], dtype=np.int64).reshape(-1)
        action_params_arr = np.asarray(action[1], dtype=np.float32)
        if action_discrete_arr.size != env.n_ch:
            raise ValueError(f"Invalid action_discrete size: got {action_discrete_arr.size}, expected {env.n_ch}")
        if action_params_arr.shape != (env.n_ch, expected_params):
            raise ValueError(
                "Invalid action_params shape: got "
                f"{action_params_arr.shape}, expected ({env.n_ch}, {expected_params}) "
                f"(mode={env.uav_mobility_control})"
            )
        action_iter = (
            (int(action_discrete_arr[i]), action_params_arr[i])
            for i in range(env.n_ch)
        )
    else:
        action_iter = (action[i] for i in range(env.n_ch))

    for i, (discrete_action, all_action_params) in enumerate(action_iter):
        discrete_action = int(discrete_action)

        all_action_params = np.asarray(all_action_params, dtype=np.float32).reshape(-1)
        if all_action_params.size != expected_params:
            raise ValueError(
                f"Invalid action_params size for cluster {i}: got {all_action_params.size}, "
                f"expected {expected_params} (mode={env.uav_mobility_control})"
            )

        # 截取 power 部分（与 baseline 同切片）
        start = discrete_action * env.param_dim_per_action
        end = start + env.param_dim_per_action
        power_norm = np.clip(all_action_params[start:end], 0.0, 1.0)

        # 解码离散动作为 n_des 个信道选择（baseline n_channel 进制，最低位对应 j=0）
        decoded = discrete_action
        for j in range(env.n_des):
            channel_last = env.uav_channels[i][j]
            env.uav_channels[i][j] = int(decoded % env.n_channel)
            env.uav_powers[i][j] = float(
                env.uav_power_min
                + float(power_norm[j]) * (env.uav_power_max - env.uav_power_min)
            )
            if env.uav_channels[i][j] != channel_last:
                env.uav_jump_count[i][j] += 1
            decoded = int(decoded / env.n_channel)

        if is_policy:
            raw = all_action_params[env.total_param_dim:env.total_param_dim + 3]
            # 末尾 3 维在 [-1, 1] 上；保险起见 clip 一下再按 *_delta_max 缩放
            raw = np.clip(raw, -1.0, 1.0)
            mobility_buffer[i, 0] = float(raw[0]) * vel_max
            mobility_buffer[i, 1] = float(raw[1]) * dir_max
            mobility_buffer[i, 2] = float(raw[2]) * p_max

    env._last_mobility_deltas = mobility_buffer


__all__ = ["decompose"]
