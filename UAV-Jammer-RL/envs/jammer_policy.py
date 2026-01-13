from __future__ import division

import random
from typing import Any

import numpy as np


def init_jammer_state(env: Any) -> None:
    if env.type_of_interference == "saopin":
        # self.jammer_channels = random.sample(range(0, self.n_channel), k=self.n_jammer)  #不包括 stop
        env.jammer_channels = random.choices(range(env.n_channel), k=env.n_jammer)
    elif env.type_of_interference == "markov":
        env.jammer_channels = random.choices(env.all_jammer_states_list, k=1)[0]
    else:
        raise ValueError(f"Unknown type_of_interference: {env.type_of_interference!r}")

    env.jammer_channels_list = []
    env.jammer_index_list = []
    # 如果传输阶段先后干扰两个信道,0是后半段 改变后的信道，1是前半段 改变前的信道
    env.jammer_time = np.zeros([2])  # 每个干扰机在传输阶段最多先后干扰两个信道，目前假设各个干扰机时间线相同


def renew_jammer_channels_after_Rx(env: Any) -> None:
    env.t_uav += env.t_Rx
    env.t_jammer += env.t_Rx
    # self.jammer_channels_list = []
    if np.floor_divide((env.t_jammer - env.t_Rx), env.t_dwell) == np.floor_divide(env.t_jammer, env.t_dwell) - 1:
        # （干扰机时间-传输时间0.98）/干扰机扫频停留时间2.28 == 干扰机时间/干扰机扫频停留时间 - 1
        if env.type_of_interference == "saopin":
            for i in range(env.n_jammer):
                env.jammer_channels[i] += env.step_forward
                env.jammer_channels[i] = int(env.jammer_channels[i] % env.n_channel)

            if env.t_jammer % env.t_dwell == 0:
                for i in range(env.n_jammer):
                    env.jammer_channels_list.append((env.jammer_channels[i] + env.n_channel - 1) % env.n_channel)
                    env.jammer_index_list.append(i)
                env.jammer_time[0] = env.t_Rx

            else:  # 正好在Rx中间切换干扰信道
                for i in range(env.n_jammer):
                    env.jammer_channels_list.append(env.jammer_channels[i])  # 后半段
                    env.jammer_index_list.append(i)
                    env.jammer_channels_list.append(
                        (env.jammer_channels[i] + env.n_channel - 1) % env.n_channel
                    )  # jammer_channels[i]-1
                    env.jammer_index_list.append(i)
                change_times = np.floor_divide(env.t_jammer, env.t_dwell)
                change_point = change_times * env.t_dwell

                env.jammer_time[0] = env.t_jammer - change_point  # 0对应传输后半段的干扰时间
                env.jammer_time[1] = env.t_Rx - env.jammer_time[0]

        elif env.type_of_interference == "markov":
            old_jammer_channels = env.jammer_channels
            env.jammer_channels = tuple(env.jammer_channels)
            idx = env.all_jammer_states_list.index(env.jammer_channels)
            p = env.p_trans[idx]
            env.jammer_channels = random.choices(env.all_jammer_states_list, weights=p, k=1)[0]

            if env.t_jammer % env.t_dwell == 0:  # 传输完成后切换干扰信道
                for i in range(env.n_jammer):
                    env.jammer_channels_list.append(old_jammer_channels[i])
                    env.jammer_index_list.append(i)
                env.jammer_time[0] = env.t_Rx

            else:  # 传输中切换干扰信道
                for i in range(env.n_jammer):
                    env.jammer_channels_list.append(env.jammer_channels[i])  # 后半段
                    env.jammer_index_list.append(i)
                    env.jammer_channels_list.append(old_jammer_channels[i])  # jammer_channels[i]-1
                    env.jammer_index_list.append(i)
                change_times = np.floor_divide(env.t_jammer, env.t_dwell)
                change_point = change_times * env.t_dwell

                env.jammer_time[0] = env.t_jammer - change_point  # 0对应传输后半段的干扰时间
                env.jammer_time[1] = env.t_Rx - env.jammer_time[0]

        # print("jammer_channels", self.jammer_channels_list)


def renew_jammer_channels_after_learn(env: Any) -> None:
    env.t_uav += env.timestep
    env.t_jammer += env.timestep
    if (
        np.floor_divide((env.t_jammer - env.timestep), env.t_dwell)
        == np.floor_divide(env.t_jammer, env.t_dwell) - 1
    ):  # 这里是什么意思
        if env.type_of_interference == "saopin":
            for i in range(env.n_jammer):
                env.jammer_channels[i] += env.step_forward
                env.jammer_channels[i] = int(env.jammer_channels[i] % env.n_channel)

                env.jammer_channels_list.append(env.jammer_channels[i])
                env.jammer_index_list.append(i)
            env.jammer_time[0] = env.t_Rx

        elif env.type_of_interference == "markov":
            idx = env.all_jammer_states_list.index(env.jammer_channels)
            p = env.p_trans[idx]
            env.jammer_channels = random.choices(env.all_jammer_states_list, weights=p, k=1)[0]

            # if self.t_jammer % self.t_dwell == 0:  传输开始前切换干扰信道
            for i in range(env.n_jammer):
                env.jammer_channels_list.append(env.jammer_channels[i])
                env.jammer_index_list.append(i)
            env.jammer_time[0] = env.t_Rx

        # print("change_channels", self.jammer_channels)


def generate_p_trans(jammer_state_dim: int, mode: int = 1) -> np.ndarray:
    # 不使用uniform, 因为从统计上感觉很好学, 差异性不大
    p_trans = np.random.uniform(0, 1, [jammer_state_dim, jammer_state_dim])  # 从[0,1)均匀分布随机取数
    p_trans_sum = np.sum(p_trans, axis=1)  # 每一行的数相加得到列向量
    if mode == 1:
        for i in range(jammer_state_dim):
            temp = np.random.randint(low=0, high=jammer_state_dim)
            p_trans[i][temp] += p_trans_sum[i] / 2
            while np.random.random() > 0.5:
                temp = np.random.randint(low=0, high=jammer_state_dim)
                p_trans[i][temp] += p_trans_sum[i] / 3
    elif mode == 2:
        for i in range(jammer_state_dim):
            while np.random.random() > 0.7:
                temp = np.random.randint(low=0, high=jammer_state_dim)
                p_trans[i][temp] += p_trans_sum[i] / 2
    elif mode == 3:
        pass
    elif mode == 4:
        for i in range(jammer_state_dim):
            temp = np.random.randint(low=0, high=jammer_state_dim)
            p_trans[i][temp] += p_trans_sum[i]

    p_trans_sum = np.sum(p_trans, axis=1)
    for i in range(jammer_state_dim):
        for j in range(jammer_state_dim):
            p_trans[i][j] = p_trans[i][j] / p_trans_sum[i]  # 每行归一化
    return p_trans

