import random
from dataclasses import dataclass
from typing import Any

import numpy as np


_TIME_EPS = 1e-9


@dataclass(frozen=True)
class JammerEvent:
    jammer_idx: int
    channel: int
    t_start: float
    t_end: float


def _dwell_index(t: float, dwell: float) -> int:
    return int(np.floor((float(t) + _TIME_EPS) / float(dwell)))


def _at_dwell_boundary(t: float, dwell: float) -> bool:
    remainder = float(t) % float(dwell)
    return (remainder < _TIME_EPS) or (abs(remainder - float(dwell)) < _TIME_EPS)


def _jammer_choices(env: Any, population, weights=None, k: int = 1):
    rng = getattr(env, "_jammer_state_rng", None)
    if rng is None:
        return random.choices(population, weights=weights, k=k)
    return rng.choices(population, weights=weights, k=k)


def _observed_uav_channel_set(env: Any) -> set[int]:
    prob = float(getattr(env, "jammer_reactive_observe_prob", 1.0))
    if prob <= 0.0:
        return set()

    channels = [int(x) for x in np.asarray(env.uav_channels, dtype=np.int32).reshape(-1).tolist()]
    true_set = set(channels)
    if prob >= 1.0:
        return true_set

    rng = getattr(env, "_jammer_state_rng", None)
    if rng is None:
        rng = random

    return {ch for ch in channels if float(rng.random()) < prob}


def _reactive_transition_row(env: Any, idx: int) -> np.ndarray:
    p = np.asarray(env.p_trans[idx], dtype=np.float64)
    beta = float(getattr(env, "jammer_reactive_beta", 0.0))
    if beta <= 0.0:
        return p

    observed_set = _observed_uav_channel_set(env)
    if not observed_set:
        return p

    scores = np.asarray(
        [sum(1 for ch in state if ch in observed_set) for state in env.all_jammer_states_list],
        dtype=np.float64,
    )
    w = p * np.exp(beta * scores)
    w_sum = float(np.sum(w))
    if np.isfinite(w_sum) and w_sum > 0.0:
        return w / w_sum
    return p


def init_jammer_state(env: Any) -> None:
    env.jammer_channels = _jammer_choices(env, env.all_jammer_states_list, k=1)[0]
    env.jammer_events = []


def renew_jammer_channels_after_Rx(env: Any) -> None:
    env.t_uav += env.t_Rx
    env.t_jammer += env.t_Rx
    prev_dwell = _dwell_index(env.t_jammer - env.t_Rx, env.t_dwell)
    curr_dwell = _dwell_index(env.t_jammer, env.t_dwell)
    if prev_dwell == curr_dwell - 1:
        # （干扰机时间-传输时间0.98）/干扰机扫频停留时间2.28 == 干扰机时间/干扰机扫频停留时间 - 1
        old_jammer_channels = tuple(env.jammer_channels)
        idx = env.all_jammer_states_list.index(old_jammer_channels)
        p = _reactive_transition_row(env, idx)
        env.jammer_channels = _jammer_choices(env, env.all_jammer_states_list, weights=p.tolist(), k=1)[0]

        if _at_dwell_boundary(env.t_jammer, env.t_dwell):  # 传输完成后切换干扰信道
            env.jammer_events = [
                JammerEvent(jammer_idx=i, channel=int(old_jammer_channels[i]), t_start=0.0, t_end=float(env.t_Rx))
                for i in range(env.n_jammer)
            ]

        else:  # 传输中切换干扰信道
            change_point = curr_dwell * env.t_dwell
            t_switch = float(env.t_Rx - (env.t_jammer - change_point))
            env.jammer_events = []
            for i in range(env.n_jammer):
                env.jammer_events.append(
                    JammerEvent(jammer_idx=i, channel=int(old_jammer_channels[i]), t_start=0.0, t_end=t_switch)
                )
                env.jammer_events.append(
                    JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]), t_start=t_switch, t_end=float(env.t_Rx))
                )

def renew_jammer_channels_after_learn(env: Any) -> None:
    env.t_uav += env.timestep
    env.t_jammer += env.timestep
    if (
        _dwell_index(env.t_jammer - env.timestep, env.t_dwell)
        == _dwell_index(env.t_jammer, env.t_dwell) - 1
    ):  # 这里是什么意思
        idx = env.all_jammer_states_list.index(tuple(env.jammer_channels))
        p = _reactive_transition_row(env, idx)
        env.jammer_channels = _jammer_choices(env, env.all_jammer_states_list, weights=p.tolist(), k=1)[0]

        # The new jammer state is already active before the next Rx interval.
        env.jammer_events = [
            JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]), t_start=0.0, t_end=float(env.t_Rx))
            for i in range(env.n_jammer)
        ]

        # print("change_channels", self.jammer_channels)


def generate_p_trans(jammer_state_dim: int, mode: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    # 不使用uniform, 因为从统计上感觉很好学, 差异性不大
    if rng is None:
        rng = np.random.default_rng()
    p_trans = rng.uniform(0, 1, [jammer_state_dim, jammer_state_dim])  # 从[0,1)均匀分布随机取数
    p_trans_sum = np.sum(p_trans, axis=1)  # 每一行的数相加得到列向量
    if mode == 1:
        for i in range(jammer_state_dim):
            temp = rng.integers(low=0, high=jammer_state_dim)
            p_trans[i][temp] += p_trans_sum[i] / 2
            while rng.random() > 0.5:
                temp = rng.integers(low=0, high=jammer_state_dim)
                p_trans[i][temp] += p_trans_sum[i] / 3
    elif mode == 2:
        for i in range(jammer_state_dim):
            while rng.random() > 0.7:
                temp = rng.integers(low=0, high=jammer_state_dim)
                p_trans[i][temp] += p_trans_sum[i] / 2
    elif mode == 3:
        pass
    elif mode == 4:
        for i in range(jammer_state_dim):
            temp = rng.integers(low=0, high=jammer_state_dim)
            p_trans[i][temp] += p_trans_sum[i]

    p_trans_sum = np.sum(p_trans, axis=1)
    for i in range(jammer_state_dim):
        for j in range(jammer_state_dim):
            p_trans[i][j] = p_trans[i][j] / p_trans_sum[i]  # 每行归一化
    return p_trans
