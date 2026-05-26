"""移动层：初始位置采样 + Gauss-Markov 位置/速度更新 + 边界反射 + Stage 3.5 策略派发。

设计原则：

- ``GaussMarkovUAVStrategy`` / ``GaussMarkovJammerStrategy`` 与 baseline
  ``core.py:743-805`` / ``907-971`` 1:1 对应，确保 Stage 3 等价。
- ``PolicyUAVStrategy`` 是 Stage 3.5 扩展：从 ``env._last_mobility_deltas`` 读 3 个 delta，
  按 ``uav_*_delta_max`` 缩放后叠加；**不执行 baseline 末尾的 Gauss-Markov 随机更新**，
  policy action 是 CH 运动状态的唯一增量来源。
- ``UAVGuidedMarkovJammerStrategy`` 在 GaussMarkov 基础上仅修改 ``mean_direction`` /
  ``mean_p``：用线性混合 ``(1-g) * mean_* + g * angle_to_nearest_ch``，引导目标固定为
  **最近 CH**（多个等距时取索引最小），``g == 0`` 严格退化为 GaussMarkov。
- 本模块用 ``TYPE_CHECKING`` 守护 ``Environ`` 引用，防 ``environment → mobility →
  environment`` 形成 import 环（plan locked decision #6）。
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.entities import UAV, Jammer, RP

if TYPE_CHECKING:
    from src.envs.environment import Environ


# ----------------------- 策略 Protocol -----------------------

class UAVMobilityStrategy(Protocol):
    def update_ch_positions(self, env: "Environ") -> None: ...


class JammerMobilityStrategy(Protocol):
    def update_jammer_positions(self, env: "Environ") -> None: ...


# ----------------------- 初始化（不属于策略，复用） -----------------------

def init_uavs(env: "Environ") -> None:
    """簇头 UAV 初始位置/速度/方向采样，与 baseline ``renew_uavs`` 1:1 一致。"""
    for i in range(env.n_ch):
        ch_id = env.ch_list[i]
        start_velocity = float(env._rng.uniform(10.0, 20.0))
        start_direction = float(env._rng.uniform(0.0, 2 * math.pi))
        start_p = float(env._rng.uniform(0.0, 2 * math.pi))

        ch_xpos = float(env._rng.uniform(0.0, env.length))
        ch_ypos = float(env._rng.uniform(0.0, env.width))
        ch_zpos = float(env._rng.uniform(env.low_height, env.high_height))
        start_position = [ch_xpos, ch_ypos, ch_zpos]

        env.uavs[ch_id] = UAV(start_position, start_direction, start_velocity, start_p)
        env.rps[ch_id] = RP(start_position)


def init_uav_clusters(env: "Environ") -> None:
    """簇成员初始位置 + 簇拓扑 (uav_pairs / uav_clusters)，与 baseline 1:1 一致。"""
    cm_list = deepcopy(env.cm_list)
    rp_cm_list = deepcopy(env.rp_cm_list)
    for i in range(env.n_ch):
        ch_id = env.ch_list[i]
        cms = env._sample_without_replacement(cm_list, env.n_cm_for_a_ch)
        rps = cms
        for j in range(env.n_cm_for_a_ch):
            env.uav_clusters[i][j][0] = ch_id
            env.uav_clusters[i][j][1] = cms[j]
            env.uav_pairs[i][j][0] = ch_id
            env.uav_pairs[i][j][1] = cms[j]
            env.uavs[ch_id].connections.append(cms[j])
            env.uavs[ch_id].destinations.append(cms[j])

            ch_pos = [
                env.uavs[ch_id].position[0],
                env.uavs[ch_id].position[1],
                env.uavs[ch_id].position[2],
            ]

            R1 = float(env._rng.uniform(0.0, env.max_distance1))
            d1 = float(env._rng.uniform(0.0, 2 * math.pi))
            p1 = float(env._rng.uniform(0.0, 2 * math.pi))

            rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
            rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
            rp_zpos = ch_pos[2] + R1 * math.sin(p1)
            while ((rp_xpos < 0) or (rp_xpos > env.length) or (rp_ypos < 0) or (rp_ypos > env.width)
                   or (rp_zpos < env.low_height) or (rp_zpos > env.high_height)):
                R1 = float(env._rng.uniform(0.0, R1))
                d1 = float(env._rng.uniform(0.0, 2 * math.pi))
                p1 = float(env._rng.uniform(0.0, 2 * math.pi))
                rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                rp_zpos = ch_pos[2] + R1 * math.sin(p1)

            R2 = float(env._rng.uniform(0.0, env.max_distance2))
            d2 = float(env._rng.uniform(0.0, 2 * math.pi))
            p2 = float(env._rng.uniform(0.0, 2 * math.pi))

            cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
            cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
            cm_zpos = rp_zpos + R2 * math.sin(p2)
            while ((cm_xpos < 0) or (cm_xpos > env.length) or (cm_ypos < 0) or (cm_ypos > env.width)
                   or (cm_zpos < env.low_height) or (cm_zpos > env.high_height)):
                R2 = float(env._rng.uniform(0.0, env.max_distance2))
                d2 = float(env._rng.uniform(0.0, 2 * math.pi))
                p2 = float(env._rng.uniform(0.0, 2 * math.pi))
                cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_zpos + R2 * math.sin(p2)

            start_position = [cm_xpos, cm_ypos, cm_zpos]
            start_position_rp = [rp_xpos, rp_ypos, rp_zpos]
            # baseline 中 CM 的 direction/velocity/p 都被显式设为 None。
            env.uavs[cms[j]] = UAV(start_position, None, None, None)
            env.uavs[cms[j]].connections.append(ch_id)
            env.uavs[cms[j]].destinations.append(ch_id)
            env.rps[rps[j]] = RP(start_position_rp)

        cms_set = set(cms)
        rps_set = set(rps)
        cm_list = [cm_id for cm_id in cm_list if cm_id not in cms_set]
        rp_cm_list = [rp_id for rp_id in rp_cm_list if rp_id not in rps_set]


def init_jammers(env: "Environ") -> None:
    """干扰机初始位置；Stage 3.5 起干扰机始终运动（loader 已保证 is_jammer_moving=True）。"""
    for _ in range(env.n_jammer):
        start_velocity = float(env._rng.uniform(10.0, 20.0))
        start_direction = float(env._rng.uniform(0.0, 2 * math.pi))
        start_p = float(env._rng.uniform(0.0, 2 * math.pi))

        xpos = float(env._rng.uniform(0.0, env.length))
        ypos = float(env._rng.uniform(0.0, env.width))
        zpos = float(env._rng.uniform(env.low_height, env.high_height))
        start_position = [xpos, ypos, zpos]
        env.jammers.append(Jammer(start_position, start_direction, start_velocity, start_p))


# ----------------------- UAV CH 策略 -----------------------

def _gauss_markov_position_step(uav, env: "Environ", delta_seconds: float):
    """通用：依据速度/方向/仰角推位置 + 边界反射，返回 (xyz, delta, oob_amount)。"""
    delta_distance = uav.velocity * delta_seconds
    d = uav.direction
    p = uav.p

    x_delta = delta_distance * math.cos(d) * math.cos(p)
    y_delta = delta_distance * math.sin(d) * math.cos(p)
    z_delta = delta_distance * math.sin(p)

    xpos = uav.position[0] + x_delta
    ypos = uav.position[1] + y_delta
    zpos = uav.position[2] + z_delta
    oob_amount = 0.0

    # X / Y 边界反射：与 baseline 完全一致（注意 X 用 π - d，Y 用 2π - d）
    if xpos < 0:
        oob_amount += float(-xpos)
        uav.direction = math.pi - uav.direction
        xpos = abs(x_delta) - uav.position[0]
    if xpos > env.length:
        oob_amount += float(xpos - env.length)
        uav.direction = math.pi - uav.direction
        xpos = 2 * env.length - abs(x_delta) - uav.position[0]
    if ypos < 0:
        oob_amount += float(-ypos)
        uav.direction = 2 * math.pi - uav.direction
        ypos = abs(y_delta) - uav.position[1]
    if ypos > env.width:
        oob_amount += float(ypos - env.width)
        uav.direction = 2 * math.pi - uav.direction
        ypos = 2 * env.width - abs(y_delta) - uav.position[1]
    if zpos < env.low_height:
        oob_amount += float(env.low_height - zpos)
        uav.p = 2 * math.pi - uav.p
        zpos = 2 * env.low_height - uav.position[2] + abs(z_delta)
    if zpos > env.high_height:
        oob_amount += float(zpos - env.high_height)
        uav.p = 2 * math.pi - uav.p
        zpos = 2 * env.high_height - uav.position[2] - abs(z_delta)

    return [xpos, ypos, zpos], (x_delta, y_delta, z_delta), oob_amount


class GaussMarkovUAVStrategy:
    """与 baseline ``renew_positions_of_chs`` 1:1 等价。"""

    def update_ch_positions(self, env: "Environ") -> None:
        env.xyz_delta_dis = [[0, 0, 0] for _ in range(env.n_ch)]
        env.xyz_oob_dis = [0.0 for _ in range(env.n_ch)]
        noise_scale = (1 - env.k ** 2) ** 0.5
        for ch in range(env.n_ch):
            i = env.ch_list[ch]
            uav = env.uavs[i]
            new_pos, (xd, yd, zd), oob = _gauss_markov_position_step(uav, env, env.timestep)
            env.xyz_delta_dis[ch] = [xd, yd, zd]
            env.xyz_oob_dis[ch] = float(oob)
            uav.position = new_pos

            # Gauss-Markov 随机更新：与 baseline 公式与 RNG 调用次序完全一致。
            uav.velocity = (
                env.k * uav.velocity
                + (1 - env.k) * uav.mean_velocity
                + noise_scale * env._rng.normal(0.0, env.sigma)
            )
            uav.direction = (
                env.k * uav.direction
                + (1 - env.k) * uav.mean_direction
                + noise_scale * env._rng.normal(0.0, env.sigma)
            )
            uav.p = (
                env.k * uav.p
                + (1 - env.k) * uav.mean_p
                + noise_scale * env._rng.normal(0.0, env.sigma)
            )


class PolicyUAVStrategy:
    """Stage 3.5: policy action 是 CH 运动状态的唯一增量来源。

    应用顺序（每 CH）：
    1. 从 ``env._last_mobility_deltas`` 读 3 个 delta（已被 action_space.decompose 缩放）。
    2. ``velocity ← max(0, v + Δv)`` / ``direction ← (d + Δd) mod 2π`` / ``p ← (p + Δp) mod 2π``。
    3. 位置更新 + 边界反射（沿用 baseline 公式）。
    4. **不执行** baseline 末尾的 Gauss-Markov 随机更新。
    """

    def update_ch_positions(self, env: "Environ") -> None:
        env.xyz_delta_dis = [[0, 0, 0] for _ in range(env.n_ch)]
        env.xyz_oob_dis = [0.0 for _ in range(env.n_ch)]
        deltas = getattr(env, "_last_mobility_deltas", None)
        two_pi = 2.0 * math.pi
        for ch in range(env.n_ch):
            i = env.ch_list[ch]
            uav = env.uavs[i]

            if deltas is not None:
                dv, dd, dp = deltas[ch]
                uav.velocity = max(0.0, uav.velocity + float(dv))
                uav.direction = (uav.direction + float(dd)) % two_pi
                uav.p = (uav.p + float(dp)) % two_pi

            new_pos, (xd, yd, zd), oob = _gauss_markov_position_step(uav, env, env.timestep)
            env.xyz_delta_dis[ch] = [xd, yd, zd]
            env.xyz_oob_dis[ch] = float(oob)
            uav.position = new_pos
        # 显式不做 Gauss-Markov 随机 velocity/direction/p 更新。


# ----------------------- CM 跟随（与策略解耦） -----------------------

def update_cm_positions(env: "Environ") -> None:
    """簇成员跟随簇头位移更新，逻辑与 baseline ``renew_positions_of_cms`` 1:1 等价。"""
    for i in range(env.n_ch):
        ch_id = env.ch_list[i]
        cm_id = env.uavs[ch_id].connections
        ch_pos = [
            env.uavs[ch_id].position[0],
            env.uavs[ch_id].position[1],
            env.uavs[ch_id].position[2],
        ]

        if env.xyz_delta_dis[i] == [0, 0, 0]:
            # 簇头本步未移动：重新采样参考点 + 簇成员位置（与 baseline 等价）
            for j in cm_id:
                R1 = float(env._rng.uniform(0.0, env.max_distance1))
                d1 = float(env._rng.uniform(0.0, 2 * math.pi))
                p1 = float(env._rng.uniform(0.0, 2 * math.pi))

                rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                while ((rp_xpos < 0) or (rp_xpos > env.length) or (rp_ypos < 0)
                       or (rp_ypos > env.width) or (rp_zpos < env.low_height)
                       or (rp_zpos > env.high_height)):
                    R1 = float(env._rng.uniform(0.0, R1))
                    d1 = float(env._rng.uniform(0.0, 2 * math.pi))
                    p1 = float(env._rng.uniform(0.0, 2 * math.pi))
                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                env.rps[j].position = rp_pos

                R2 = float(env._rng.uniform(0.0, env.max_distance2))
                d2 = float(env._rng.uniform(0.0, 2 * math.pi))
                p2 = float(env._rng.uniform(0.0, 2 * math.pi))

                cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                env.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

                while ((env.uavs[j].position[0] < 0) or (env.uavs[j].position[0] > env.length)
                       or (env.uavs[j].position[1] < 0) or (env.uavs[j].position[1] > env.width)
                       or (env.uavs[j].position[2] < env.low_height)
                       or (env.uavs[j].position[2] > env.high_height)):
                    R2 = float(env._rng.uniform(0.0, env.max_distance2))
                    d2 = float(env._rng.uniform(0.0, 2 * math.pi))
                    p2 = float(env._rng.uniform(0.0, 2 * math.pi))
                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                    env.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]
        else:
            # 簇头已移动：把同样的位移叠加到 RP，再 RP+R2 偏移得到 CM
            for j in cm_id:
                rp_xpos = env.rps[j].position[0] + env.xyz_delta_dis[i][0]
                rp_ypos = env.rps[j].position[1] + env.xyz_delta_dis[i][1]
                rp_zpos = env.rps[j].position[2] + env.xyz_delta_dis[i][2]

                if rp_xpos < 0:
                    rp_xpos = abs(env.xyz_delta_dis[i][0]) - env.rps[j].position[0]
                if rp_xpos > env.length:
                    rp_xpos = 2 * env.length - abs(env.xyz_delta_dis[i][0]) - env.rps[j].position[0]
                if rp_ypos < 0:
                    rp_ypos = abs(env.xyz_delta_dis[i][1]) - env.rps[j].position[1]
                if rp_ypos > env.width:
                    rp_ypos = 2 * env.width - env.rps[j].position[1] - abs(env.xyz_delta_dis[i][1])
                if rp_zpos < env.low_height:
                    rp_zpos = 2 * env.low_height - env.rps[j].position[2] + abs(env.xyz_delta_dis[i][2])
                if rp_zpos > env.high_height:
                    rp_zpos = 2 * env.high_height - env.rps[j].position[2] - abs(env.xyz_delta_dis[i][2])

                rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                env.rps[j].position = rp_pos

                R2 = float(env._rng.uniform(0.0, env.max_distance2))
                d2 = float(env._rng.uniform(0.0, 2 * math.pi))
                p2 = float(env._rng.uniform(0.0, 2 * math.pi))

                cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_pos[2] + R2 * math.sin(p2)

                while ((cm_xpos < 0) or (cm_xpos > env.length) or (cm_ypos < 0)
                       or (cm_ypos > env.width) or (cm_zpos < env.low_height)
                       or (cm_zpos > env.high_height)):
                    R2 = float(env._rng.uniform(0.0, env.max_distance2))
                    d2 = float(env._rng.uniform(0.0, 2 * math.pi))
                    p2 = float(env._rng.uniform(0.0, 2 * math.pi))
                    cm_xpos = env.rps[j].position[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = env.rps[j].position[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = env.rps[j].position[2] + R2 * math.sin(p2)
                env.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]


# ----------------------- Jammer 策略 -----------------------

def _jammer_position_step(jammer, env: "Environ"):
    """与 ``_gauss_markov_position_step`` 同思路，但用 ``t_collect`` 而非 ``timestep``。"""
    delta_distance = jammer.velocity * env.t_collect
    d = jammer.direction
    p = jammer.p

    x_delta = delta_distance * math.cos(d) * math.cos(p)
    y_delta = delta_distance * math.sin(d) * math.cos(p)
    z_delta = delta_distance * math.sin(p)

    xpos = jammer.position[0] + x_delta
    ypos = jammer.position[1] + y_delta
    zpos = jammer.position[2] + z_delta

    if jammer.position[0] + x_delta < 0:
        jammer.direction = math.pi - jammer.direction
        xpos = abs(x_delta) - jammer.position[0]
    if jammer.position[0] + x_delta > env.length:
        jammer.direction = math.pi - jammer.direction
        xpos = 2 * env.length - abs(x_delta) - jammer.position[0]
    if jammer.position[1] + y_delta < 0:
        jammer.direction = 2 * math.pi - jammer.direction
        ypos = abs(y_delta) - jammer.position[1]
    if jammer.position[1] + y_delta > env.width:
        jammer.direction = 2 * math.pi - jammer.direction
        ypos = 2 * env.width - abs(y_delta) - jammer.position[1]
    if jammer.position[2] + z_delta < env.low_height:
        jammer.p = 2 * math.pi - jammer.p
        zpos = 2 * env.low_height - jammer.position[2] + abs(z_delta)
    if jammer.position[2] + z_delta > env.high_height:
        jammer.p = 2 * math.pi - jammer.p
        zpos = 2 * env.high_height - (jammer.position[2] + abs(z_delta))

    return [xpos, ypos, zpos]


def _gauss_markov_jammer_kinetic_update(jammer, env: "Environ",
                                        mean_direction: float, mean_p: float) -> None:
    """Gauss-Markov 推进 jammer 速度/方向/仰角，``mean_*`` 由策略决定。"""
    noise_scale = (1 - env.k ** 2) ** 0.5
    jammer.velocity = (
        env.k * jammer.velocity
        + (1 - env.k) * jammer.mean_velocity
        + noise_scale * env._rng.normal(0.0, env.sigma)
    )
    jammer.direction = (
        env.k * jammer.direction
        + (1 - env.k) * mean_direction
        + noise_scale * env._rng.normal(0.0, env.sigma)
    )
    jammer.p = (
        env.k * jammer.p
        + (1 - env.k) * mean_p
        + noise_scale * env._rng.normal(0.0, env.sigma)
    )


class GaussMarkovJammerStrategy:
    """与 baseline ``renew_positions_of_jammers`` 1:1 等价。"""

    def update_jammer_positions(self, env: "Environ") -> None:
        for i in range(len(env.jammers)):
            jammer = env.jammers[i]
            jammer.position = _jammer_position_step(jammer, env)
            _gauss_markov_jammer_kinetic_update(
                jammer, env,
                mean_direction=jammer.mean_direction,
                mean_p=jammer.mean_p,
            )


class UAVGuidedMarkovJammerStrategy:
    """Stage 3.5: 在 GaussMarkov 上把方向均值朝最近 CH 偏置。

    `g = env.jammer_guidance_strength ∈ [0, 1]`：
    - ``g == 0`` → mean_direction / mean_p == jammer.mean_*（严格等价 GaussMarkov）。
    - ``g == 1`` → 完全朝向最近 CH。

    引导目标固定为**最近 CH**，多个等距时取**索引最小**者（plan locked decision #1）。
    """

    def _nearest_ch_target(self, env: "Environ", jammer) -> tuple[float, float]:
        """返回 (target_direction, target_p_angle)，单位弧度。"""
        if env.n_ch == 0:
            return jammer.mean_direction, jammer.mean_p

        ch_positions = np.asarray(
            [env.uavs[env.ch_list[i]].position for i in range(env.n_ch)], dtype=np.float64
        )
        jpos = np.asarray(jammer.position, dtype=np.float64)
        diff = ch_positions - jpos  # (n_ch, 3)
        dists = np.linalg.norm(diff, axis=1)
        # 多个等距时取索引最小（np.argmin 行为符合）
        idx = int(np.argmin(dists))
        dx, dy, dz = float(diff[idx, 0]), float(diff[idx, 1]), float(diff[idx, 2])

        horiz = math.hypot(dx, dy)
        target_direction = math.atan2(dy, dx) % (2.0 * math.pi)
        target_p = math.atan2(dz, horiz)  # 仰角，可能为负
        return target_direction, target_p

    def update_jammer_positions(self, env: "Environ") -> None:
        g = float(env.jammer_guidance_strength)
        for i in range(len(env.jammers)):
            jammer = env.jammers[i]
            jammer.position = _jammer_position_step(jammer, env)
            if g == 0.0:
                mean_dir = jammer.mean_direction
                mean_p = jammer.mean_p
            else:
                tgt_dir, tgt_p = self._nearest_ch_target(env, jammer)
                mean_dir = (1.0 - g) * jammer.mean_direction + g * tgt_dir
                mean_p = (1.0 - g) * jammer.mean_p + g * tgt_p
            _gauss_markov_jammer_kinetic_update(jammer, env, mean_direction=mean_dir, mean_p=mean_p)


# ----------------------- 策略派发工厂 -----------------------

def build_uav_mobility_strategy(cfg) -> UAVMobilityStrategy:
    mode = cfg.uav_mobility_control
    if mode == "gauss_markov":
        return GaussMarkovUAVStrategy()
    if mode == "policy":
        return PolicyUAVStrategy()
    raise ValueError(f"unsupported uav_mobility_control={mode!r}")


def build_jammer_mobility_strategy(cfg) -> JammerMobilityStrategy:
    mode = cfg.jammer_mobility_model
    if mode == "gauss_markov":
        return GaussMarkovJammerStrategy()
    if mode == "uav_guided_markov":
        return UAVGuidedMarkovJammerStrategy()
    raise ValueError(f"unsupported jammer_mobility_model={mode!r}")


__all__ = [
    "UAVMobilityStrategy",
    "JammerMobilityStrategy",
    "GaussMarkovUAVStrategy",
    "PolicyUAVStrategy",
    "GaussMarkovJammerStrategy",
    "UAVGuidedMarkovJammerStrategy",
    "init_uavs",
    "init_uav_clusters",
    "init_jammers",
    "update_cm_positions",
    "build_uav_mobility_strategy",
    "build_jammer_mobility_strategy",
]
