"""信道层：3D 路损 + 频率选择性 + AR(1) 快衰落 + 静态信道损耗叠加。

设计原则（来自 REFACTOR.md Constraint 2）：

- 与 baseline ``UAV-Jammer-RL/envs/channels.py`` 全文 + ``core.py:289-317 / 973-1010 /
  118-192`` 1:1 对应；任何数值偏差都会破坏 ``test_env_contract`` 的 golden master 对齐。
- 本模块不 import torch；只依赖 numpy。
- 路损公式与 baseline 完全一致：``103.8 + 20.9 * log10(distance_km + tiny)``，
  其中 distance 单位米、加 ``0.001`` m 避免 ``log10(0)``。
- 快衰落状态 ``_uav_fast_h`` / ``_jammer_fast_h`` 由 ``Environ`` 持有，本模块只负责
  ``sample_complex_gaussian`` 和 ``update_fast_fading``，便于 ``Environ`` 在 reset
  时清空状态、step 时推进。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 仅类型注解时引用 Environ，运行时不形成 import 环
    from src.envs.environment import Environ


def _position_array(positions) -> np.ndarray:
    """把任意可迭代的 3D 位置序列归一化为 ``(N, 3)`` float64 数组。"""
    arr = np.asarray(positions, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr.reshape(-1, 3)


def pathloss_matrix(tx_positions, rx_positions) -> np.ndarray:
    """计算 tx → rx 的 dB 路损矩阵，输出 shape ``(len(tx), len(rx))``。

    公式与 baseline ``channels.py:11-16`` 完全一致，不可改动。
    """
    tx = _position_array(tx_positions)
    rx = _position_array(rx_positions)
    diff = tx[:, np.newaxis, :] - rx[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=-1) + 0.001
    return 103.8 + 20.9 * np.log10(distance * 1e-3)


def pathloss_between(position_A, position_B) -> float:
    return float(pathloss_matrix([position_A], [position_B])[0, 0])


class UAVchannels:
    """UAV-UAV 链路路损缓存。结构与 baseline ``channels.py:23-37`` 一致。"""

    def __init__(self, n_uav: int, n_channel: int, BS_position) -> None:
        del n_channel, BS_position  # 旧实现已不使用，仅为签名兼容保留参数
        self.n_uav = int(n_uav)
        self.positions = None
        self.PathLoss = None

    def update_positions(self, positions) -> None:
        self.positions = positions

    def update_pathloss(self) -> None:
        self.PathLoss = pathloss_matrix(self.positions, self.positions)

    def get_path_loss(self, position_A, position_B) -> float:
        return pathloss_between(position_A, position_B)


class Jammerchannels:
    """Jammer-UAV 链路路损缓存。结构与 baseline ``channels.py:39-55`` 一致。"""

    def __init__(self, n_jammer: int, n_uav: int, n_channel: int, BS_position) -> None:
        del n_channel, BS_position
        self.n_jammer = int(n_jammer)
        self.n_uav = int(n_uav)
        self.positions = None
        self.uav_positions = None
        self.PathLoss = None

    def update_positions(self, positions, uav_positions) -> None:
        self.positions = positions
        self.uav_positions = uav_positions

    def update_pathloss(self) -> None:
        self.PathLoss = pathloss_matrix(self.positions, self.uav_positions)

    def get_path_loss(self, position_A, position_B) -> float:
        return pathloss_between(position_A, position_B)


def sample_complex_gaussian(shape, rng: np.random.Generator) -> np.ndarray:
    """采样 ``CN(0, 1)`` 复高斯系数，shape 与 baseline ``core.py:289-292`` 一致。"""
    real = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    imag = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def update_fast_fading(env: "Environ") -> None:
    """推进快衰落 AR(1) 状态。

    与 baseline ``core.py:294-317`` 一致：禁用时清空；首次调用初始化 ``h_0``；
    后续按 ``h_t = ρ * h_{t-1} + √(1-ρ²) * w_t`` 更新；w_t 来自 ``_fast_fading_rng``。
    """
    if not env.enable_fast_fading:
        env._uav_fast_h = None
        env._jammer_fast_h = None
        return
    if env._uav_fast_h is None or env._jammer_fast_h is None:
        env._uav_fast_h = sample_complex_gaussian(
            (int(env.n_uav), int(env.n_uav), int(env.n_channel)),
            env._fast_fading_rng,
        )
        env._jammer_fast_h = sample_complex_gaussian(
            (int(env.n_jammer), int(env.n_uav), int(env.n_channel)),
            env._fast_fading_rng,
        )
        return
    rho = float(env.fast_fading_rho)
    sigma = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    env._uav_fast_h = (
        rho * env._uav_fast_h
        + sigma * sample_complex_gaussian(env._uav_fast_h.shape, env._fast_fading_rng)
    ).astype(np.complex64)
    env._jammer_fast_h = (
        rho * env._jammer_fast_h
        + sigma * sample_complex_gaussian(env._jammer_fast_h.shape, env._fast_fading_rng)
    ).astype(np.complex64)


def renew_channels(env: "Environ") -> None:
    """完整刷新 UAV/Jammer 链路损耗矩阵。

    与 baseline ``core.py:973-1010`` 1:1 对应。计算顺序、`np.repeat` 维度、
    `channel_loss_db` 和 `*_channel_selectivity_db` 的叠加方式都不可改动。
    最终写入 ``env.UAVchannels_loss_db`` / ``env.Jammerchannels_loss_db``，
    供 link_budget / observation 直接读取。
    """
    uav_positions = [u.position for u in env.uavs]
    jammer_positions = [j.position for j in env.jammers]

    # baseline 中先更新 Jammerchannels 再更新 UAVchannels；保持顺序。
    env.Jammerchannels.update_positions(jammer_positions, uav_positions)
    env.UAVchannels.update_positions(uav_positions)
    env.Jammerchannels.update_pathloss()
    env.UAVchannels.update_pathloss()
    update_fast_fading(env)

    # UAV 链路：基础路损 + 静态信道损耗 + 频率选择性 + （可选）快衰落
    uav_channels_loss_db = np.repeat(
        env.UAVchannels.PathLoss[:, :, np.newaxis], env.n_channel, axis=2
    )
    uav_channels_loss_db = (
        uav_channels_loss_db
        + env.channel_loss_db.reshape(1, 1, -1)
        + env.uav_channel_selectivity_db
    )
    if env.enable_fast_fading and env._uav_fast_h is not None:
        uav_fast_db = (
            20.0 * np.log10(np.abs(env._uav_fast_h) + float(env.fast_fading_eps))
            - env._rayleigh_mean_db
        )
        if env.fast_fading_db_clip_low is not None and env.fast_fading_db_clip_high is not None:
            uav_fast_db = np.clip(
                uav_fast_db, env.fast_fading_db_clip_low, env.fast_fading_db_clip_high
            )
        env.UAVchannels_loss_db = uav_channels_loss_db - uav_fast_db.astype(np.float32)
    else:
        env.UAVchannels_loss_db = uav_channels_loss_db

    # Jammer 链路：同结构
    jammer_channels_loss_db = np.repeat(
        env.Jammerchannels.PathLoss[:, :, np.newaxis], env.n_channel, axis=2
    )
    jammer_channels_loss_db = (
        jammer_channels_loss_db
        + env.channel_loss_db.reshape(1, 1, -1)
        + env.jammer_channel_selectivity_db
    )
    if env.enable_fast_fading and env._jammer_fast_h is not None:
        jammer_fast_db = (
            20.0 * np.log10(np.abs(env._jammer_fast_h) + float(env.fast_fading_eps))
            - env._rayleigh_mean_db
        )
        if env.fast_fading_db_clip_low is not None and env.fast_fading_db_clip_high is not None:
            jammer_fast_db = np.clip(
                jammer_fast_db, env.fast_fading_db_clip_low, env.fast_fading_db_clip_high
            )
        env.Jammerchannels_loss_db = jammer_channels_loss_db - jammer_fast_db.astype(np.float32)
    else:
        env.Jammerchannels_loss_db = jammer_channels_loss_db


__all__ = [
    "pathloss_matrix",
    "pathloss_between",
    "UAVchannels",
    "Jammerchannels",
    "sample_complex_gaussian",
    "update_fast_fading",
    "renew_channels",
]
