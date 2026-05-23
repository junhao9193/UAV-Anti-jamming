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
        raise RuntimeError("Reactive jammer requires env._jammer_state_rng; initialize Environ first.")
    return rng.choices(population, weights=weights, k=k)


def _sample_observed_uav_channel_set(env: Any) -> set[int]:
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


def record_jammer_observation(env: Any) -> None:
    """Push this step's partial UAV-channel observation into jammer memory.

    Skip when reactive jamming is disabled or when memory is disabled
    (window=1). The window=1 path samples directly inside the reactive row
    calculation to preserve the legacy single-step behavior.
    """
    if float(getattr(env, "jammer_reactive_beta", 0.0)) <= 0.0:
        return
    if int(getattr(env, "jammer_memory_window", 4)) <= 1:
        return

    history = getattr(env, "_jammer_observed_channel_history", None)
    if history is None:
        return
    history.append(_sample_observed_uav_channel_set(env))


def _observed_uav_channel_frequencies(env: Any) -> dict[int, float]:
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


def _reactive_transition_row(env: Any, idx: int) -> np.ndarray:
    p = np.asarray(env.p_trans[idx], dtype=np.float64)
    beta = float(getattr(env, "jammer_reactive_beta", 0.0))
    if beta <= 0.0:
        return p

    observed_freq = _observed_uav_channel_frequencies(env)
    if not observed_freq:
        return p

    state_channel_counts = getattr(env, "_jammer_state_channel_counts", None)
    if state_channel_counts is not None:
        freq_vec = np.zeros(int(env.n_channel), dtype=np.float64)
        for ch, freq in observed_freq.items():
            ch = int(ch)
            if 0 <= ch < int(env.n_channel):
                freq_vec[ch] = float(freq)
        scores = np.asarray(state_channel_counts, dtype=np.float64) @ freq_vec
    else:
        scores = np.asarray(
            [sum(observed_freq.get(int(ch), 0.0) for ch in state) for state in env.all_jammer_states_list],
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
        # Detect whether the Rx interval crossed exactly one jammer dwell boundary.
        old_jammer_channels = tuple(env.jammer_channels)
        idx = getattr(env, "_jammer_state_to_index", {}).get(old_jammer_channels)
        if idx is None:
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
        jammer_channels = tuple(env.jammer_channels)
        idx = getattr(env, "_jammer_state_to_index", {}).get(jammer_channels)
        if idx is None:
            idx = env.all_jammer_states_list.index(jammer_channels)
        p = _reactive_transition_row(env, idx)
        env.jammer_channels = _jammer_choices(env, env.all_jammer_states_list, weights=p.tolist(), k=1)[0]

        # The new jammer state is already active before the next Rx interval.
        env.jammer_events = [
            JammerEvent(jammer_idx=i, channel=int(env.jammer_channels[i]), t_start=0.0, t_end=float(env.t_Rx))
            for i in range(env.n_jammer)
        ]

        # print("change_channels", self.jammer_channels)


def generate_p_trans(
    jammer_state_dim: int,
    rng: np.random.Generator | None = None,
    preferred_next_states: int = 2,
    preference_strength: float = 0.5,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
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
