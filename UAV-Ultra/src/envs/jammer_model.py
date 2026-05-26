"""干扰机层：联合 Markov 状态空间 + 反应性偏差 + Rx / learn 阶段状态转移。

与 baseline ``UAV-Jammer-RL/envs/jammer_policy.py`` 全文 + ``core.py:319-329 / 1052-1069``
1:1 对应；任何数值偏差都会破坏 ``test_env_contract`` 的 golden master 对齐。

**`generate_p_trans` 强制接收 ``np.random.Generator``**（plan locked decision #4）：
- baseline 默认参数 ``rng=None`` 会触发 ``np.random.default_rng()``，产生跨进程漂移；
  Stage 3 起 ``Environ`` 独立持有 ``self._p_trans_rng``（由 ``cfg.p_trans_seed`` 派生），
  并在内部传入，避免 Stage 0 已修复的不确定性回归。
- 测试可继续通过 ``reset(p_trans=...)`` 注入 golden master 矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import comb, perm

if TYPE_CHECKING:
    from src.envs.environment import Environ


_TIME_EPS = 1e-9


@dataclass(frozen=True)
class JammerEvent:
    """单条 jammer 占用事件：[t_start, t_end] 内 jammer_idx 占用 channel。"""

    jammer_idx: int
    channel: int
    t_start: float
    t_end: float


# ----------------------- 状态枚举 -----------------------

def all_observed_states(env: "Environ") -> None:
    """枚举 jammer 联合状态空间，与 baseline ``core.py:319-329`` 一致。

    写入 ``env`` 的字段：
    - ``all_jammer_states_list``: 所有 ``P(n_channel, n_jammer)`` 个有序排列
    - ``jammer_state_dim``: 上述排列数量
    - ``all_observed_states_list`` / ``observed_state_dim``: 组合（无序）枚举
    - ``observed_state_list``: 旧实现仅声明为空列表，此处保持兼容
    """
    env.observed_state_list = []
    env.all_observed_states_list = []
    env.all_jammer_states_list = []

    env.jammer_state_dim = int(perm(env.n_channel, env.n_jammer))
    env.all_jammer_states_list.extend(list(permutations(env.channel_indexes, env.n_jammer)))

    env.observed_state_dim = int(comb(env.n_channel, env.n_jammer))
    env.all_observed_states_list.extend(list(combinations(env.channel_indexes, env.n_jammer)))


# ----------------------- 工具函数 -----------------------

def _dwell_index(t: float, dwell: float) -> int:
    return int(np.floor((float(t) + _TIME_EPS) / float(dwell)))


def _at_dwell_boundary(t: float, dwell: float) -> bool:
    remainder = float(t) % float(dwell)
    return (remainder < _TIME_EPS) or (abs(remainder - float(dwell)) < _TIME_EPS)


def _jammer_choices(env: "Environ", population, weights=None, k: int = 1):
    """走 ``env._jammer_state_rng``（Python ``random.Random``），保证可复现。"""
    rng = getattr(env, "_jammer_state_rng", None)
    if rng is None:
        raise RuntimeError("Reactive jammer requires env._jammer_state_rng; initialize Environ first.")
    return rng.choices(population, weights=weights, k=k)


def _sample_observed_uav_channel_set(env: "Environ") -> set[int]:
    """采样 jammer 部分观察到的 UAV 信道集合（与 baseline 等价）。"""
    prob = float(getattr(env, "jammer_reactive_observe_prob", 1.0))
    if prob <= 0.0:
        return set()

    channels = [int(x) for x in np.asarray(env.uav_channels, dtype=np.int32).reshape(-1).tolist()]
    true_set = set(channels)
    if prob >= 1.0:
        return true_set

    rng = getattr(env, "_jammer_state_rng", None)
    if rng is None:
        raise RuntimeError("Reactive jammer requires env._jammer_state_rng; initialize Environ first.")
    return {ch for ch in channels if float(rng.random()) < prob}


def record_jammer_observation(env: "Environ") -> None:
    """把当前步的 jammer 部分观察压入历史队列；window=1 或 beta<=0 时跳过。"""
    if float(getattr(env, "jammer_reactive_beta", 0.0)) <= 0.0:
        return
    if int(getattr(env, "jammer_memory_window", 4)) <= 1:
        return
    history = getattr(env, "_jammer_observed_channel_history", None)
    if history is None:
        return
    history.append(_sample_observed_uav_channel_set(env))


def _observed_uav_channel_frequencies(env: "Environ") -> dict[int, float]:
    """聚合历史窗口里观察到的 UAV 信道频率（与 baseline jammer_policy.py:69-91 一致）。"""
    if int(getattr(env, "jammer_memory_window", 4)) <= 1:
        observed_set = _sample_observed_uav_channel_set(env)
        return {ch: 1.0 for ch in observed_set}

    history = getattr(env, "_jammer_observed_channel_history", None)
    if history is None:
        observed_set = _sample_observed_uav_channel_set(env)
        return {ch: 1.0 for ch in observed_set}

    if len(history) == 0:
        history.append(_sample_observed_uav_channel_set(env))

    freq: dict[int, float] = {}
    for observed_set in history:
        for ch in observed_set:
            freq[int(ch)] = freq.get(int(ch), 0.0) + 1.0

    if not freq:
        return {}
    scale = 1.0 / float(len(history))
    return {ch: count * scale for ch, count in freq.items()}


def _reactive_transition_row(env: "Environ", idx: int) -> np.ndarray:
    """根据观察频率给 Markov 行加 ``exp(beta * score)`` 权重，再重归一。"""
    p = np.asarray(env.p_trans[idx], dtype=np.float64)
    beta = float(getattr(env, "jammer_reactive_beta", 0.0))
    if beta <= 0.0:
        return p

    observed_freq = _observed_uav_channel_frequencies(env)
    if not observed_freq:
        return p

    scores = np.asarray(
        [sum(observed_freq.get(int(ch), 0.0) for ch in state) for state in env.all_jammer_states_list],
        dtype=np.float64,
    )
    w = p * np.exp(beta * scores)
    w_sum = float(np.sum(w))
    if np.isfinite(w_sum) and w_sum > 0.0:
        return w / w_sum
    return p


# ----------------------- 状态初始化与转移 -----------------------

def init_jammer_state(env: "Environ") -> None:
    """从所有可能状态中采样一个作为初始 jammer 联合状态。"""
    env.jammer_channels = _jammer_choices(env, env.all_jammer_states_list, k=1)[0]
    env.jammer_events = []


def renew_jammer_channels_after_Rx(env: "Environ") -> None:
    """Rx 阶段后的 Markov 转移，可能跨 dwell 边界（与 baseline 等价）。"""
    env.t_uav += env.t_Rx
    env.t_jammer += env.t_Rx
    prev_dwell = _dwell_index(env.t_jammer - env.t_Rx, env.t_dwell)
    curr_dwell = _dwell_index(env.t_jammer, env.t_dwell)
    if prev_dwell == curr_dwell - 1:
        old_jammer_channels = tuple(env.jammer_channels)
        idx = env.all_jammer_states_list.index(old_jammer_channels)
        p = _reactive_transition_row(env, idx)
        env.jammer_channels = _jammer_choices(
            env, env.all_jammer_states_list, weights=p.tolist(), k=1
        )[0]

        if _at_dwell_boundary(env.t_jammer, env.t_dwell):
            env.jammer_events = [
                JammerEvent(jammer_idx=i, channel=int(old_jammer_channels[i]),
                            t_start=0.0, t_end=float(env.t_Rx))
                for i in range(env.n_jammer)
            ]
        else:
            change_point = curr_dwell * env.t_dwell
            t_switch = float(env.t_Rx - (env.t_jammer - change_point))
            env.jammer_events = []
            for i in range(env.n_jammer):
                env.jammer_events.append(
                    JammerEvent(jammer_idx=i, channel=int(old_jammer_channels[i]),
                                t_start=0.0, t_end=t_switch)
                )
                env.jammer_events.append(
                    JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]),
                                t_start=t_switch, t_end=float(env.t_Rx))
                )


def renew_jammer_channels_after_learn(env: "Environ") -> None:
    """学习阶段后的 Markov 转移（与 baseline 等价）。"""
    env.t_uav += env.timestep
    env.t_jammer += env.timestep
    if (
        _dwell_index(env.t_jammer - env.timestep, env.t_dwell)
        == _dwell_index(env.t_jammer, env.t_dwell) - 1
    ):
        idx = env.all_jammer_states_list.index(tuple(env.jammer_channels))
        p = _reactive_transition_row(env, idx)
        env.jammer_channels = _jammer_choices(
            env, env.all_jammer_states_list, weights=p.tolist(), k=1
        )[0]
        env.jammer_events = [
            JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]),
                        t_start=0.0, t_end=float(env.t_Rx))
            for i in range(env.n_jammer)
        ]


# ----------------------- p_trans 生成 -----------------------

def generate_p_trans(
    jammer_state_dim: int,
    *,
    rng: np.random.Generator,
    preferred_next_states: int = 2,
    preference_strength: float = 0.5,
) -> np.ndarray:
    """生成 jammer 联合状态 Markov 转移矩阵。

    与 baseline ``jammer_policy.py:170-195`` 等价，但**强制接收** ``rng`` 参数
    （baseline 允许 ``rng=None`` → 跨进程不确定，Stage 3 起禁止）。
    """
    if rng is None:
        raise TypeError(
            "generate_p_trans requires an explicit np.random.Generator (rng=None is forbidden "
            "since Stage 3 to eliminate cross-process drift)."
        )
    p_trans = rng.uniform(0.0, 1.0, [jammer_state_dim, jammer_state_dim])
    preferred_next_states = int(preferred_next_states)
    preference_strength = float(preference_strength)
    if preferred_next_states > 0 and preference_strength > 0.0:
        preferred_next_states = min(preferred_next_states, jammer_state_dim)
        p_trans_sum = np.sum(p_trans, axis=1)
        preferred_idx = np.asarray(
            [
                rng.choice(jammer_state_dim, size=preferred_next_states, replace=False)
                for _ in range(jammer_state_dim)
            ],
            dtype=np.int64,
        )
        row_idx = np.arange(jammer_state_dim, dtype=np.int64)[:, None]
        np.add.at(p_trans, (row_idx, preferred_idx), p_trans_sum[:, None] * preference_strength)

    p_trans_sum = np.sum(p_trans, axis=1, keepdims=True)
    return p_trans / p_trans_sum


__all__ = [
    "JammerEvent",
    "all_observed_states",
    "init_jammer_state",
    "record_jammer_observation",
    "renew_jammer_channels_after_Rx",
    "renew_jammer_channels_after_learn",
    "generate_p_trans",
]
