"""链路预算：单 (cluster_head, dest) 链路的 SINR / 传输时间 / 成功判定。

与 baseline ``UAV-Jammer-RL/envs/core.py:600-673`` (``compute_reward``) 1:1 对应。
本模块**不**做任何奖励聚合（那是 ``reward.py`` 的职责），也**不**写
``env.last_link_metrics``（在 ``reward.compute_step_reward`` 里统一写入），仅返回
``(transmit_time, success_flag)``。

success_flag 语义沿用 baseline：成功 ``+1``，失败 ``-3``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.envs.environment import Environ


_TIME_EPS = 1e-9


def compute_link(env: "Environ", i: int, j: int, other_channel_list, pairs) -> tuple[float, int]:
    """单链路 (i, j) 的传输时间与成功标记。

    ``other_channel_list`` 是除当前 (i, j) 外其他所有 (i', j') 的信道列表；
    ``pairs`` 是与之配对的 [(i', j'), ...]，二者长度相同。
    """
    uav_uav_interference = 0.0

    transmitter_idx = env.uav_pairs[i][j][0]
    receiver_idx = env.uav_pairs[i][j][1]
    target_channel = int(env.uav_channels[i][j])

    uav_signal = 10 ** (
        (
            env.uav_powers[i][j]
            - env.UAVchannels_loss_db[transmitter_idx, receiver_idx, target_channel]
            + 2 * env.uavAntGain
            - env.uavNoiseFigure
        )
        / 10
    )

    other_channel_arr = np.asarray(other_channel_list, dtype=np.int32)
    if target_channel in other_channel_list:
        index = np.where(other_channel_arr == target_channel)
        for k in range(len(index[0])):
            ii, jj = pairs[index[0][k]]
            interferer_tx_idx = env.uav_pairs[ii][jj][0]
            uav_uav_interference += 10 ** (
                (
                    env.uav_powers[ii][jj]
                    - env.UAVchannels_loss_db[interferer_tx_idx, receiver_idx, target_channel]
                    + 2 * env.uavAntGain
                    - env.uavNoiseFigure
                )
                / 10
            )

    events = [
        event
        for event in env.jammer_events
        if int(event.channel) == target_channel and float(event.t_end) > float(event.t_start)
    ]
    boundaries = [0.0, float(env.t_Rx)]
    for event in events:
        boundaries.append(float(np.clip(event.t_start, 0.0, env.t_Rx)))
        boundaries.append(float(np.clip(event.t_end, 0.0, env.t_Rx)))
    boundaries = sorted(set(boundaries))

    remaining_data = float(env.data_size)
    transmit_time = float(env.t_Rx)
    interference_scale = float(max(0.0, env.uav_interference_scale))

    for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
        duration = float(segment_end - segment_start)
        if duration <= _TIME_EPS:
            continue

        jammer_interference = 0.0
        for event in events:
            if (float(event.t_start) <= segment_start + _TIME_EPS
                    and float(event.t_end) >= segment_end - _TIME_EPS):
                jammer_idx = int(event.jammer_idx)
                jammer_interference += 10 ** (
                    (
                        env.jammer_power
                        - env.Jammerchannels_loss_db[jammer_idx, receiver_idx, target_channel]
                        + env.jammerAntGain
                        + env.uavAntGain
                        - env.uavNoiseFigure
                    )
                    / 10
                )

        denom = (
            interference_scale * float(uav_uav_interference)
            + float(jammer_interference)
            + float(env.sig2)
        )
        uav_rate = np.log2(1 + np.divide(uav_signal, denom))
        uav_rate *= env.bandwidth
        deliverable = float(uav_rate) * duration
        if deliverable + _TIME_EPS >= remaining_data:
            transmit_time = float(segment_start + remaining_data / float(uav_rate))
            break
        remaining_data -= deliverable

    if transmit_time < env.t_Rx:
        return float(transmit_time), 1
    return float(env.t_Rx), -3


__all__ = ["compute_link"]
