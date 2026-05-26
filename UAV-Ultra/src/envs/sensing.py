"""频谱感知：连续能量图 + 距离过滤 + 噪声 + z-score 标准化。

与 baseline ``UAV-Jammer-RL/envs/core.py:536-590`` 1:1 对应。本模块**仅**负责生成
每个 CH 的频谱感知向量（长度 ``n_channel``）；CSI 与全状态装配在 ``observation.py``。

边界划清（plan §2 关键设计目标 "观测边界"）：
- ``sensing.py`` 处理 jammer / UAV 占用估计 + ``sensing_noise_std`` 噪声 + 标准化。
- ``observation.py`` 处理 CSI + ``csi_noise_std`` + 全状态拼装。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.envs.environment import Environ


def spectrum_energy_map(env: "Environ") -> np.ndarray:
    """计算每个 CH 的频谱感知向量，返回 shape ``(n_ch, n_channel)`` float32。

    与 baseline ``get_state()`` 的频谱感知段（行 536-590）逐操作对应：
    1. jammer 占用：仅统计 ``sensing_jammer_range`` 内的 jammer；权重 ``sensing_w_jammer``。
    2. 其他 CH 占用：仅统计 ``sensing_uav_range`` 内的 CH；权重 ``sensing_w_uav``，按 CH 计。
    3. 噪声：``sensing_noise_std > 0`` 时叠加 ``N(0, σ)``；用 ``env._rng``，与 baseline 同 RNG。
    4. z-score 标准化 + clip 到 ``[-1, 1]``。
    """
    n_ch = int(env.n_ch)
    n_channel = int(env.n_channel)

    jammer_ch_list = (
        list(env.jammer_channels) if not isinstance(env.jammer_channels, list) else env.jammer_channels
    )
    other_used_sets = [
        set(map(int, env.uav_channels[k].reshape(-1).tolist())) for k in range(n_ch)
    ]

    ch_tx_ids = np.asarray(
        [int(env.uav_pairs[k][0][0]) for k in range(n_ch)], dtype=np.int32
    )
    ch_positions = np.asarray(
        [env.uavs[idx].position for idx in ch_tx_ids], dtype=np.float32
    )  # (n_ch, 3)
    jammer_positions = (
        np.asarray([j.position for j in env.jammers], dtype=np.float32)
        if len(env.jammers) > 0
        else None
    )

    result = np.zeros((n_ch, n_channel), dtype=np.float32)

    for i in range(n_ch):
        z = np.zeros([n_channel], dtype=np.float32)

        # jammer 占用（距离过滤）
        jammer_set_i: set[int] = set()
        if jammer_positions is None:
            jammer_set_i = set(map(int, jammer_ch_list))
        else:
            d_j = np.linalg.norm(jammer_positions - ch_positions[i], axis=1)
            for jammer_idx, ch in enumerate(jammer_ch_list):
                if (jammer_idx < d_j.shape[0]
                        and float(d_j[jammer_idx]) <= float(env.sensing_jammer_range)):
                    jammer_set_i.add(int(ch))

        if jammer_set_i:
            z[np.asarray(list(jammer_set_i), dtype=np.int32)] += float(env.sensing_w_jammer)

        # 其他 CH 占用（按 CH 计，按距离过滤）
        d_ch = np.linalg.norm(ch_positions - ch_positions[i], axis=1)
        for k in range(n_ch):
            if k == i:
                continue
            if float(d_ch[k]) > float(env.sensing_uav_range):
                continue
            used = other_used_sets[k]
            if used:
                z[np.asarray(list(used), dtype=np.int32)] += float(env.sensing_w_uav)

        # 感知噪声（用 env._rng，与 baseline 同 RNG 流，调用次序保持）
        if env.sensing_noise_std > 0.0:
            z += env._rng.normal(0.0, env.sensing_noise_std, size=n_channel).astype(np.float32)

        # z-score 标准化 + clip
        mu = float(np.mean(z))
        std = float(np.std(z))
        if std < 1e-6:
            z_norm = np.zeros_like(z, dtype=np.float32)
        else:
            z_norm = (z - mu) / (std + 1e-12)
        result[i] = np.clip(z_norm, -1.0, 1.0).astype(np.float32)

    return result


__all__ = ["spectrum_energy_map"]
