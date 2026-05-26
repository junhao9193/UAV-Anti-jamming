"""观测层：CSI 装配 + 频谱感知拼接 + 可选 Stage 3.5 mobility 特征。

与 baseline ``UAV-Jammer-RL/envs/core.py:484-598`` 1:1 对应：

- CSI 部分 (``csi_features``)：对每个 (CH, dest) 链路，按当前发射机→接收机的路损向量构造
  ``(n_channel,)`` CSI，做 ``(x - csi_pathloss_offset) / csi_pathloss_scale`` 归一化；
  ``csi_noise_std > 0`` 时叠加 ``env._rng.normal(...)``；``csi_clip`` 为 True 时 clip 到 [-1, 1]。
- 拼装 (``get_state``)：每个 CH 输出 ``concat(CSI(n_des * n_channel), sensing(n_channel))``；
  Stage 3.5 ``observation_include_mobility=True`` 时追加 6 维 ``(pos_xyz, vel, dir, p)``
  的归一化值。
- ``baseline`` 中存在 ``policy == "Sensing_Based_Method"`` 分支，UAV-Ultra Stage 3 仅迁
  默认 MP-DQN 路径（其他 policy 路径在 baseline 已注明 NotImplementedError）。

观测向量布局的**唯一真相源**就在本文件；任何字段增删必须同步更新 ``config/specs.py``，
``test_config_specs`` 的 ``len(state) == specs.state_dim(cfg)`` 是强制断言。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import numpy as np

from src.envs.sensing import spectrum_energy_map

if TYPE_CHECKING:
    from src.envs.environment import Environ


def csi_features(env: "Environ") -> np.ndarray:
    """每个 (CH, dest) 链路的归一化 CSI，shape ``(n_ch, n_des, n_channel)`` float32。"""
    n_ch = int(env.n_ch)
    n_des = int(env.n_des)
    n_channel = int(env.n_channel)
    csi = np.zeros([n_ch, n_des, n_channel], dtype=np.float32)

    for i in range(n_ch):
        for j in range(n_des):
            tra_id = env.uav_pairs[i][j][0]
            rec_id = env.uav_pairs[i][j][1]
            pathloss_vec = env.UAVchannels_loss_db[tra_id, rec_id, :].astype(np.float32)
            csi_ij = (pathloss_vec - env.csi_pathloss_offset) / env.csi_pathloss_scale
            if env.csi_noise_std > 0.0:
                csi_ij = csi_ij + env._rng.normal(
                    0.0, env.csi_noise_std, size=n_channel
                ).astype(np.float32)
            if env.csi_clip:
                csi_ij = np.clip(csi_ij, -1.0, 1.0)
            csi[i, j, :] = csi_ij
    return csi


def _mobility_features_per_ch(env: "Environ") -> np.ndarray:
    """Stage 3.5: 每个 CH 自身的归一化 (pos_xyz, vel, dir, p)，shape ``(n_ch, 6)``。

    归一化方式：
    - 位置：``x / length``、``y / width``、``z / high_height``（除数都是配置常数）
    - 速度：``velocity / 20.0``（baseline 初始上界）
    - 方向 / 仰角：``angle / (2π)``（baseline 取值范围）
    """
    feats = np.zeros((env.n_ch, 6), dtype=np.float32)
    two_pi = 2.0 * math.pi
    for ch in range(env.n_ch):
        i = env.ch_list[ch]
        uav = env.uavs[i]
        feats[ch, 0] = float(uav.position[0]) / float(env.length)
        feats[ch, 1] = float(uav.position[1]) / float(env.width)
        feats[ch, 2] = float(uav.position[2]) / float(env.high_height)
        feats[ch, 3] = float(uav.velocity) / 20.0
        feats[ch, 4] = (float(uav.direction) % two_pi) / two_pi
        feats[ch, 5] = (float(uav.p) % two_pi) / two_pi
    return feats


def get_state(env: "Environ") -> List[np.ndarray]:
    """组装观测列表，长度 ``n_ch``；每个元素是 1D float32 数组，长度 ``state_dim``。

    baseline 返回 Python list（每个元素 ``np.ndarray``），保持兼容；Stage 3 不改契约。
    """
    if not env._episode_initialized:
        raise RuntimeError("Environment episode state is not initialized. Call reset() before get_state().")

    csi = csi_features(env)
    sensing = spectrum_energy_map(env)
    include_mobility = bool(env.observation_include_mobility)
    if include_mobility:
        mobility = _mobility_features_per_ch(env)

    joint_state: List[np.ndarray] = []
    for i in range(env.n_ch):
        parts = [csi[i].reshape(-1), sensing[i]]
        if include_mobility:
            parts.append(mobility[i])
        joint_state.append(np.concatenate(parts).astype(np.float32))
    return joint_state


__all__ = ["csi_features", "get_state"]
