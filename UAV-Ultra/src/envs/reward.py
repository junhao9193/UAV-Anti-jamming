"""奖励层：base reward 聚合（baseline 等价）+ Stage 3.5 mobility 惩罚（可分离调用）。

设计原则：

- ``compute_step_reward(env)``：循环调 ``link_budget.compute_link()`` 计算每条链路的
  传输时间与成功标记，聚合为每个 CH 的 ``uav_rewards``；累计 ``rew_suc / rew_energy /
  rew_jump``；fairness 公平性团队惩罚；**同时写入 ``env.last_link_metrics``**（含
  ``delivery / success_flags / transmit_times``，供 ``test_env_contract`` 直接读取）。
- ``apply_mobility_penalty(env, reward)``：Stage 3.5 扩展，按 ``mobility_oob_penalty_weight``
  / ``mobility_energy_weight`` 加权扣分；两个权重默认 0 → 函数无效果 → baseline 等价。
- ``reward_details(env)`` / ``clear_reward(env)`` 与 baseline 同语义。

与 baseline ``UAV-Jammer-RL/envs/core.py:675-734`` 1:1 对应。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.envs.jammer_model import JammerEvent
from src.envs.link_budget import compute_link

if TYPE_CHECKING:
    from src.envs.environment import Environ


def compute_step_reward(env: "Environ") -> np.ndarray:
    """Base reward 聚合，与 baseline ``get_reward`` 等价。

    返回 shape ``(n_ch,)`` 的 ``uav_rewards``；写入 ``env.last_link_metrics`` 包含：
    - ``delivery``: ``(n_ch, n_des)`` 0/1（成功为 1，失败为 0）
    - ``success_flags``: ``(n_ch, n_des)`` baseline 原始标记（成功 +1，失败 -3）
    - ``transmit_times``: ``(n_ch, n_des)`` 实际占用时间
    """
    n_ch = int(env.n_ch)
    n_des = int(env.n_des)

    uav_rewards = np.zeros([n_ch], dtype=float)

    if not env.jammer_events:
        env.jammer_events = [
            JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]),
                        t_start=0.0, t_end=float(env.t_Rx))
            for i in range(env.n_jammer)
        ]

    success_cnt = np.zeros([n_ch], dtype=np.float32)
    delivery = np.zeros((n_ch, n_des), dtype=np.float64)
    success_flags = np.zeros((n_ch, n_des), dtype=np.float64)
    transmit_times = np.zeros((n_ch, n_des), dtype=np.float64)

    tra = 0
    rec = 0
    while tra < n_ch:
        other_channel_list = []
        pairs = []
        for i in range(n_ch):
            for j in range(n_des):
                if i == tra and j == rec:
                    continue
                other_channel_list.append(env.uav_channels[i][j])
                pairs.append([i, j])

        tra_time, suc = compute_link(env, tra, rec, other_channel_list, pairs)
        env.rew_suc += suc
        if suc == 1:
            success_cnt[tra] += 1.0

        # link metrics 公开调试字段（替代 baseline Stage 0 generator 的 monkey-patch）
        success_flags[tra, rec] = float(suc)
        transmit_times[tra, rec] = float(tra_time)
        delivery[tra, rec] = 1.0 if int(suc) == 1 else 0.0

        energy = 10 ** (env.uav_powers[tra][rec] / 10 - 3) * tra_time
        env.rew_energy += energy
        jump = env.uav_jump_count[tra][rec]
        env.rew_jump += jump

        max_energy = 10 ** (env.uav_power_max / 10 - 3) * env.t_Rx + 1e-12
        norm_energy = float(energy / max_energy)
        uav_rewards[tra] += suc - (
            env.reward_energy_weight * norm_energy + env.reward_jump_weight * jump
        )

        rec += 1
        if rec == n_des:
            tra += 1
            rec = 0

    # Fairness 团队惩罚（与 baseline 一致）
    if float(env.fairness_weight) > 0.0 and float(env.fairness_min_success_rate) > 0.0:
        success_rate_per_cluster = success_cnt / float(n_des)
        shortfall = np.maximum(
            0.0, float(env.fairness_min_success_rate) - success_rate_per_cluster
        )
        team_penalty = float(env.fairness_weight) * float(np.mean(shortfall))
        uav_rewards -= team_penalty

    env.jammer_events = []
    env.uav_jump_count = np.zeros([n_ch, n_des], dtype=np.int32)

    env.last_link_metrics = {
        "delivery": delivery,
        "success_flags": success_flags,
        "transmit_times": transmit_times,
    }

    return uav_rewards


def apply_mobility_penalty(env: "Environ", reward: np.ndarray) -> np.ndarray:
    """Stage 3.5: 越界 / 移动能耗惩罚。默认权重 0 → 无效果 → baseline 等价。

    ``env.last_mobility_delta_sq`` / ``env.last_mobility_oob`` 由 ``mobility`` 模块在
    位置更新阶段填入；本函数仅做加权扣分，与 base reward 解耦。
    """
    oob_weight = float(env.mobility_oob_penalty_weight)
    energy_weight = float(env.mobility_energy_weight)
    delta_sq = getattr(env, "last_mobility_delta_sq", None)
    oob = getattr(env, "last_mobility_oob", None)
    delta_arr = (
        np.asarray(delta_sq, dtype=reward.dtype)
        if delta_sq is not None
        else np.zeros_like(reward)
    )
    oob_arr = (
        np.asarray(oob, dtype=reward.dtype)
        if oob is not None
        else np.zeros_like(reward)
    )

    energy_penalty = energy_weight * delta_arr
    oob_penalty = oob_weight * oob_arr
    env.rew_mobility_delta_sq += float(np.sum(delta_arr, dtype=np.float64))
    env.rew_mobility_oob += float(np.sum(oob_arr, dtype=np.float64))
    env.rew_mobility_penalty += float(np.sum(energy_penalty + oob_penalty, dtype=np.float64))

    if oob_weight <= 0.0 and energy_weight <= 0.0:
        return reward  # 默认路径：reward 无效果

    if energy_weight > 0.0:
        reward = reward - energy_penalty
    if oob_weight > 0.0:
        reward = reward - oob_penalty
    return reward


def reward_details(env: "Environ") -> tuple[float, float, float]:
    """与 baseline ``reward_details`` 一致：(avg_energy, avg_jump, avg_suc)。"""
    return (
        env.rew_energy / env.n_ch,
        env.rew_jump / (env.n_ch * env.n_des),
        env.rew_suc / (env.n_ch * env.n_des),
    )


def mobility_reward_details(env: "Environ") -> tuple[float, float, float]:
    """Stage 3.5 独立 mobility 明细：(avg_delta_sq, avg_oob, avg_penalty)。"""
    return (
        env.rew_mobility_delta_sq / env.n_ch,
        env.rew_mobility_oob / env.n_ch,
        env.rew_mobility_penalty / env.n_ch,
    )


def clear_reward(env: "Environ") -> None:
    env.rew_energy = 0
    env.rew_jump = 0
    env.rew_suc = 0
    env.rew_mobility_delta_sq = 0.0
    env.rew_mobility_oob = 0.0
    env.rew_mobility_penalty = 0.0


__all__ = [
    "compute_step_reward",
    "apply_mobility_penalty",
    "reward_details",
    "mobility_reward_details",
    "clear_reward",
]
