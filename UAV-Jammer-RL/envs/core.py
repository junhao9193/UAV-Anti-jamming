from __future__ import division

import math
import random
from copy import deepcopy

try:
    import gym
    from gym import spaces
except ModuleNotFoundError:  # gym is unmaintained; prefer gymnasium
    import gymnasium as gym
    from gymnasium import spaces

import numpy as np

from scipy.special import comb, perm
from itertools import combinations, permutations

import matplotlib.pyplot as plt
import seaborn as sns

from envs.channels import UAVchannels, Jammerchannels
from envs.config import load_env_config
from envs.entities import UAV, Jammer, RP
from envs.jammer_policy import (
    generate_p_trans as generate_jammer_p_trans,
    init_jammer_state,
    renew_jammer_channels_after_Rx as jammer_channels_after_Rx,
    renew_jammer_channels_after_learn as jammer_channels_after_learn,
)

class Environ(gym.Env):
    def __init__(self, config=None, config_path=None):
        cfg = load_env_config(config=config, config_path=config_path)

        self.length = cfg["length"]  # 1000
        self.width = cfg["width"]  # 500
        self.low_height = cfg["low_height"]
        self.high_height = cfg["high_height"]
        self.BS_position = cfg.get(
            "BS_position",
            [self.length / 2, self.width / 2, (self.low_height + self.high_height) / 2],  # Suppose the BS is in the center
        )
        self.k = cfg["k"]
        self.sigma = cfg["sigma"]

        #无人机、干扰机各种参数
        uav_power_list = cfg.get("uav_power_list")
        if uav_power_list is not None:
            self.uav_power_min = float(min(uav_power_list))
            self.uav_power_max = float(max(uav_power_list))
        else:
            self.uav_power_min = float(cfg["uav_power_min"])
            self.uav_power_max = float(cfg["uav_power_max"])
        self.jammer_power = cfg["jammer_power"]  # dBm
        self.sig2_dB = cfg["sig2_dB"]  # dBm       Noise power
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.uavAntGain = cfg["uavAntGain"]  # dBi       uav antenna gain
        self.uavNoiseFigure = cfg["uavNoiseFigure"]  # dB    uav receiver noise figure
        self.jammerAntGain = cfg["jammerAntGain"]  # dBi       jammer antenna gain
        self.bandwidth = cfg["bandwidth"]  # Hz
        # Difficulty knob: scale UAV-UAV interference term in SINR denominator.
        self.uav_interference_scale = float(cfg.get("uav_interference_scale", 1.0))

        #数据传输过程的参数
        self.data_size = cfg["data_size"]
        self.t_Rx = cfg["t_Rx"]  # 传输时间,单位都是s
        self.t_collect = cfg["t_collect"]  # 收集数据
        self.timestep = cfg["timestep"]  # 频谱感知，选动作 + ACK + 学习
        self.timeslot = self.t_Rx + self.timestep  # 时隙
        self.t_uav = 0.00
        # self.t_uav = Decimal(self.t_uav).quantize(Decimal("0.00"))
        self.jammer_start = cfg["jammer_start"]  # 干扰机开始干扰时间
        self.t_dwell = cfg["t_dwell"]  # 干扰机扫频停留时间
        self.t_jammer = 0.00
        # self.t_jammer = Decimal(self.t_jammer).quantize(Decimal("0.00"))

        #参考点群移动模型
        self.n_ch = cfg["n_ch"]  # TODO UAV簇头个数
        self.n_cm_for_a_ch = cfg["n_cm_for_a_ch"]  # 每个ch的簇成员个数
        self.n_cm = self.n_ch * self.n_cm_for_a_ch  # UAV簇成员个数
        self.n_uav = self.n_ch + self.n_cm  # number of UAVs
        self.n_rp_ch = self.n_ch
        self.n_rp_cm = self.n_cm
        self.n_rp = self.n_uav  # 簇成员的参考节点个数
        self.n_des = self.n_cm_for_a_ch  # 每个ch的通信目标数
        self.n_uav_pair = self.n_ch * self.n_des  # 一共6个通信对
        self.n_jammer = cfg["n_jammer"]  # number of jammers
        self.n_channel = cfg["n_channel"]  # int(self.n_ch+self.n_jammer-1)  # number of channels
        self.channel_indexes = np.arange(self.n_channel)
        self.channels = np.zeros([self.n_channel], dtype=np.int32)
        self.states_observed = cfg["states_observed"]  # 信道被干扰或未被干扰

        self.p_md = cfg["p_md"]  # 漏警概率
        self.p_fa = cfg["p_fa"]  # 虚警概率
        self.pn0 = cfg["pn0"]  # 数据包长度
        self.sensing_w_jammer = float(cfg.get("sensing_w_jammer", 1.0))
        self.sensing_w_uav = float(cfg.get("sensing_w_uav", 1.0))
        self.sensing_noise_std = float(cfg.get("sensing_noise_std", 0.0))
        self.sensing_jammer_range = float(cfg.get("sensing_jammer_range", 1e9))
        self.sensing_uav_range = float(cfg.get("sensing_uav_range", 1e9))

        channel_loss_cfg = cfg.get("channel_loss_db", None)
        if channel_loss_cfg is None:
            self.channel_loss_db = np.zeros((int(self.n_channel),), dtype=np.float32)
        else:
            self.channel_loss_db = np.asarray(channel_loss_cfg, dtype=np.float32).reshape(-1)
            if self.channel_loss_db.size != int(self.n_channel):
                raise ValueError(
                    f"channel_loss_db length must equal n_channel ({self.n_channel}), got {self.channel_loss_db.size}"
                )

        # Link-level frequency selectivity (static across the whole run when seed is fixed).
        # This is NOT fast fading (no per-step randomness), but makes different links prefer different channels.
        self.channel_selectivity_std_db = float(cfg.get("channel_selectivity_std_db", 0.0))
        self.channel_selectivity_seed = int(cfg.get("channel_selectivity_seed", 0))
        if self.channel_selectivity_std_db <= 0.0:
            self.uav_channel_selectivity_db = np.zeros(
                (int(self.n_uav), int(self.n_uav), int(self.n_channel)), dtype=np.float32
            )
            self.jammer_channel_selectivity_db = np.zeros(
                (int(self.n_jammer), int(self.n_uav), int(self.n_channel)), dtype=np.float32
            )
        else:
            rng = np.random.default_rng(self.channel_selectivity_seed)
            self.uav_channel_selectivity_db = rng.normal(
                0.0,
                self.channel_selectivity_std_db,
                size=(int(self.n_uav), int(self.n_uav), int(self.n_channel)),
            ).astype(np.float32)
            self.jammer_channel_selectivity_db = rng.normal(
                0.0,
                self.channel_selectivity_std_db,
                size=(int(self.n_jammer), int(self.n_uav), int(self.n_channel)),
            ).astype(np.float32)

        self.max_distance1 = cfg["max_distance1"]
        self.max_distance2 = cfg["max_distance2"]

        self.is_jammer_moving = cfg["is_jammer_moving"]
        self.type_of_interference = cfg["type_of_interference"]
        self.step_forward = cfg["step_forward"]
        self.p_trans_mode = cfg["p_trans_mode"]
        self.reward_energy_weight = cfg["reward_energy_weight"]
        self.reward_jump_weight = cfg["reward_jump_weight"]
        self.fairness_min_success_rate = float(cfg.get("fairness_min_success_rate", 0.0))
        self.fairness_weight = float(cfg.get("fairness_weight", 0.0))
        self.csi_pathloss_offset = float(cfg.get("csi_pathloss_offset", 80.0))
        self.csi_pathloss_scale = float(cfg.get("csi_pathloss_scale", 60.0))
        self.csi_clip = bool(cfg.get("csi_clip", True))
        # "markov"首先干扰机通过检测智能体的主要变化,
        # 识别agent的工作模式并且建立工作模式状态转移的马尔可夫链,
        # 然后利用合适的算法对建立的agent工作模式转移马尔可夫链计算转移概率,
        # 最后将agent工作模式转移概率转化为矩阵形式就对agent下一个工作模式进行预测,
        # 从而使得干扰机能够最大限度的对agent进行干扰
        self.policy = None  # 对应算法
        self.training = True
        self.jammer_channels = [0 for _ in range(self.n_jammer)]
        self.jammer_channels_list = []
        self.jammer_index_list = []
        self.jammer_time = np.zeros([2], dtype=np.float32)

        self.uav_list = list(np.arange(self.n_uav))
        self.ch_list = random.sample(self.uav_list, k=self.n_ch)#由于随机数种子的原因，每次都选择2、7、3作为簇头
        self.cm_list = list(set(self.uav_list) - set(self.ch_list))
        self.rp_list = self.uav_list
        self.rp_ch_list = self.ch_list
        self.rp_cm_list = self.cm_list
        self.uav_pairs = np.zeros([self.n_ch, self.n_des, 2], dtype=np.int32)
        self.uav_clusters = np.zeros([self.n_ch, self.n_cm_for_a_ch, 2], dtype=np.int32)

        # MP-DQN style parameterized action:
        # - Discrete action selects a channel assignment for all destinations (n_channel ** n_des)
        # - Continuous parameters provide power for each destination, for each discrete action
        self.action_dim = int(self.n_channel ** self.n_des)
        self.param_dim_per_action = int(self.n_des)
        self.total_param_dim = int(self.action_dim * self.param_dim_per_action)
        self.action_space = [
            spaces.Tuple(
                (
                    spaces.Discrete(self.action_dim),
                    spaces.Box(low=0.0, high=1.0, shape=(self.total_param_dim,), dtype=np.float32),
                )
            )
            for _ in range(self.n_ch)
        ]

        #与奖励相关参数
        self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)
        self.rew_energy = 0
        self.rew_jump = 0
        self.rew_suc = 0

        self.n_step = 0

        # 初始化观察状态和环境
        self.all_observed_states()
        self.reset(self.generate_p_trans(mode=self.p_trans_mode))
        self.state_dim = len(self.get_state()[0])
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.state_dim,)) for _ in range(self.n_ch)]

    def all_observed_states(self):
        self.observed_state_list = []
        observed_state = 0
        self.all_observed_states_list = []

        self.all_jammer_states_list = []
        self.jammer_state_dim = int(perm(self.n_channel, self.n_jammer)) # perm()全排列
        self.all_jammer_states_list.extend(list(permutations(self.channel_indexes, self.n_jammer))) # permutations给定一个数组集合，返回所有可能的排列。
        
        self.observed_state_dim = int(comb(self.n_channel, self.n_jammer)) # comb返回从 n_channel 种可能性中选择n_jammer个无序结果的方式数量，无重复，也称为组合。
        self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))

    def renew_uavs(self):
        for i in range(self.n_ch):
            # 更新簇头无人机的位置
            ch_id = self.ch_list[i]

            start_velocity = random.uniform(10, 20)
            start_direction = random.uniform(0, 2 * math.pi)
            start_p = random.uniform(0, 2 * math.pi)

            ch_xpos = random.uniform(0.0, self.length)
            ch_ypos = random.uniform(0.0, self.width)
            ch_zpos = random.uniform(self.low_height, self.high_height)
            start_position = [ch_xpos, ch_ypos, ch_zpos]

            self.uavs[ch_id] = UAV(start_position, start_direction, start_velocity, start_p)
            self.rps[ch_id] = RP(start_position)

            self.uavs[ch_id].uav_velocity.append(start_velocity)
            self.uavs[ch_id].uav_direction.append(start_direction)
            self.uavs[ch_id].uav_p.append(start_p)

    def renew_uav_clusters(self):
        cm_list = deepcopy(self.cm_list)
        rp_cm_list = deepcopy(self.rp_cm_list)
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cms = random.sample(cm_list, k=self.n_cm_for_a_ch)
            rps = cms
            for j in range(self.n_cm_for_a_ch):
                self.uav_clusters[i][j][0] = ch_id
                self.uav_clusters[i][j][1] = cms[j]
                self.uav_pairs[i][j][0] = ch_id
                self.uav_pairs[i][j][1] = cms[j]
                self.uavs[ch_id].connections.append(cms[j])
                self.uavs[ch_id].destinations.append(cms[j])

                ch_pos = [self.uavs[ch_id].position[0], self.uavs[ch_id].position[1], self.uavs[ch_id].position[2]]

                # 参考节点的位置设定
                R1 = random.uniform(0.0, self.max_distance1)
                d1 = random.uniform(0.0, 2 * math.pi)
                p1 = random.uniform(0.0, 2 * math.pi)

                rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) or (rp_ypos > self.width) or (
                        rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                    R1 = random.uniform(0.0, R1)
                    d1 = random.uniform(0.0, 2 * math.pi)
                    p1 = random.uniform(0.0, 2 * math.pi)

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                # 簇内节点的位置设定
                R2 = random.uniform(0.0, self.max_distance2)
                d2 = random.uniform(0.0, 2 * math.pi)
                p2 = random.uniform(0.0, 2 * math.pi)

                cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_zpos + R2 * math.sin(p2)

                while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (
                        cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                    # 簇内节点的位置设定
                    R2 = random.uniform(0.0, self.max_distance2)
                    d2 = random.uniform(0.0, 2 * math.pi)
                    p2 = random.uniform(0.0, 2 * math.pi)

                    cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_zpos + R2 * math.sin(p2)

                start_position = [cm_xpos, cm_ypos, cm_zpos]
                start_position_rp = [rp_xpos, rp_ypos, rp_zpos]
                start_direction = None
                start_velocity = None
                start_p = None

                self.uavs[cms[j]] = UAV(start_position, start_direction, start_velocity, start_p)
                self.uavs[cms[j]].connections.append(ch_id)
                self.uavs[cms[j]].destinations.append(ch_id)
                self.rps[rps[j]] = RP(start_position_rp)

            cm_list = list(set(cm_list) - set(cms))
            rp_cm_list = list(set(rp_cm_list) - set(rps))

        # print(self.uav_clusters)
        # print(self.uav_pairs)

    def renew_jammers(self):
        if self.is_jammer_moving:
            for i in range(self.n_jammer):
                start_velocity = random.uniform(10.0, 20.0)
                start_direction = random.uniform(0, 2 * math.pi)
                start_p = random.uniform(0, 2 * math.pi)

                xpos = random.uniform(0.0, self.length)
                ypos = random.uniform(0.0, self.width)
                zpos = random.uniform(self.low_height, self.high_height)
                start_position = [xpos, ypos, zpos]

                self.jammers.append(Jammer(start_position, start_direction, start_velocity, start_p))
                self.jammers[i].jammer_velocity.append(start_velocity)
                self.jammers[i].jammer_direction.append(start_direction)
                self.jammers[i].jammer_p.append(start_p)

    def new_random_game(self):
        # self.all_observed_states()
        self.t_uav = 0.0
        self.t_jammer = 0.0
        # 一个发送机若有多个通信目标，每个元素是智能体为每个通信目标分配的信道，假设各不相同
        self.uav_channels = np.zeros([self.n_ch, self.n_des], dtype=np.int32)   # 每个智能体观察到的全局动作（假设智能体可以观察到其他智能体已经完成的动作）
        self.uav_powers = np.zeros([self.n_ch, self.n_des], dtype=np.float32)
        self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)
        for i in range(self.n_ch):
            for j in range(self.n_des):
                self.uav_channels[i][j] = random.randint(0, self.n_channel - 1)  #包括上下限
                self.uav_powers[i][j] = random.uniform(self.uav_power_min, self.uav_power_max)  # dBm
        init_jammer_state(self)

        # print("jammer_channels", self.jammer_channels)

        self.uavs = [None] * self.n_uav
        self.rps = [None] * self.n_rp
        self.jammers = []
        self.renew_uavs()       #更新簇头无人机的位置、方向和速度
        self.renew_uav_clusters()       #更新簇头和簇内无人机的位置 并且保证簇内成员与簇头在通信范围内
        self.renew_jammers()        #更新干扰机的位置

        self.UAVchannels = UAVchannels(self.n_uav, self.n_channel, self.BS_position)
        self.Jammerchannels = Jammerchannels(self.n_jammer, self.n_uav, self.n_channel, self.BS_position)
        self.renew_channels()

    def get_state(self):
        if self.policy == "Q_learning":
            raise NotImplementedError("Q_learning policy is not supported in continuous-power (MP-DQN) mode.")

        elif self.policy == "Sensing_Based_Method":
            if not isinstance(self.jammer_channels, list):
                jammer_channels = list(self.jammer_channels)
            else:
                jammer_channels = self.jammer_channels
            if self.p_md == 0 and self.p_fa == 0:
                channels_observed = np.zeros([self.n_channel], dtype=np.int32)
                channels_observed[jammer_channels] = 1
            else:
                channels_observed = np.zeros([self.n_channel], dtype=np.int32)
                for i in range(self.n_channel):
                    if i in jammer_channels:
                        if random.random() < self.p_md:
                            channels_observed[i] = 0  # 漏警
                        else:
                            channels_observed[i] = 1  # 发现干扰
                    else:
                        if random.random() < self.p_fa:
                            channels_observed[i] = 1  # 虚警
                        else:
                            channels_observed[i] = 0  # 发现未干扰
            return channels_observed

        else:
            joint_state = []
            # CSI：每条链路在每个信道上的大尺度 CSI（路径损耗 + 信道固定差异/选择性），不引入快衰落。
            # 形状: (n_ch, n_des, n_channel)
            csi = np.zeros([self.n_ch, self.n_des, self.n_channel], dtype=np.float32)

            for i in range(self.n_ch):
                for j in range(self.n_des):
                    tra_id = self.uav_pairs[i][j][0]        # 发射机
                    rec_id = self.uav_pairs[i][j][1]        # 接收机
                    pathloss_vec = self.UAVchannels_loss_db[tra_id, rec_id, :].astype(np.float32)  # (n_channel,)
                    csi_ij = (pathloss_vec - self.csi_pathloss_offset) / self.csi_pathloss_scale
                    if self.csi_clip:
                        csi_ij = np.clip(csi_ij, -1.0, 1.0)
                    csi[i, j, :] = csi_ij

            # 频谱感知：连续的“信道能量图”作为观测（不采样成 0/1）
            # z_i(c) = w_J * I[c in C^J] + w_U * sum_{k!=i} I[c in C^k] + noise
            # 再做 z-score 标准化并 clip 到 [-1,1]，训练更稳定。
            if not isinstance(self.jammer_channels, list):
                jammer_ch_list = list(self.jammer_channels)
            else:
                jammer_ch_list = self.jammer_channels

            other_used_sets = [set(map(int, self.uav_channels[k].reshape(-1).tolist())) for k in range(self.n_ch)]

            # Range clipping: each cluster head only "sees" nearby jammers / nearby other cluster heads.
            ch_tx_ids = np.asarray([int(self.uav_pairs[k][0][0]) for k in range(self.n_ch)], dtype=np.int32)
            ch_positions = np.asarray([self.uavs[idx].position for idx in ch_tx_ids], dtype=np.float32)  # (n_ch,3)
            jammer_positions = (
                np.asarray([j.position for j in self.jammers], dtype=np.float32) if len(self.jammers) > 0 else None
            )

            for i in range(self.n_ch):
                z = np.zeros([self.n_channel], dtype=np.float32)

                # jammer 占用（仅统计探测范围内的 jammer）
                jammer_set_i = set()
                if jammer_positions is None:
                    jammer_set_i = set(map(int, jammer_ch_list))
                else:
                    d_j = np.linalg.norm(jammer_positions - ch_positions[i], axis=1)  # (n_jammer,)
                    for jammer_idx, ch in enumerate(jammer_ch_list):
                        if jammer_idx < d_j.shape[0] and float(d_j[jammer_idx]) <= float(self.sensing_jammer_range):
                            jammer_set_i.add(int(ch))

                if jammer_set_i:
                    z[np.asarray(list(jammer_set_i), dtype=np.int32)] += float(self.sensing_w_jammer)

                # 其他簇头占用（按簇头计数，不按 link 计数）
                d_ch = np.linalg.norm(ch_positions - ch_positions[i], axis=1)  # (n_ch,)
                for k in range(self.n_ch):
                    if k == i:
                        continue
                    if float(d_ch[k]) > float(self.sensing_uav_range):
                        continue
                    used = other_used_sets[k]
                    if used:
                        z[np.asarray(list(used), dtype=np.int32)] += float(self.sensing_w_uav)

                # 可选：感知噪声（默认 0，不引入随机性）
                if self.sensing_noise_std > 0.0:
                    z += np.random.normal(0.0, self.sensing_noise_std, size=self.n_channel).astype(np.float32)

                mu = float(np.mean(z))
                std = float(np.std(z))
                if std < 1e-6:
                    z_norm = np.zeros_like(z, dtype=np.float32)
                else:
                    z_norm = (z - mu) / (std + 1e-12)
                channel_sensing = np.clip(z_norm, -1.0, 1.0).astype(np.float32)

                # 观测 = [CSI(n_des*n_channel), 频谱感知(n_channel)]
                obs_i = np.concatenate([
                    csi[i].reshape(-1),         # CSI: (n_des*n_channel,)
                    channel_sensing,           # 频谱感知: n_channel
                ]).astype(np.float32)
                joint_state.append(obs_i)
            return joint_state

    def compute_reward(self, i, j, other_channel_list, pairs):
        uav_uav_interference = 0.0   # interference from other UAV transmitters (linear mW)
        jammer_interference_from_jammer0 = 0.0    #后半段干扰机干扰
        jammer_interference_from_jammer1 = 0.0   #前半段干扰机干扰

        transmitter_idx = self.uav_pairs[i][j][0]
        receiver_idx = self.uav_pairs[i][j][1]
        uav_signal = 10 ** ((self.uav_powers[i][j] - self.UAVchannels_loss_db[transmitter_idx, receiver_idx, self.uav_channels[i][j]] +
                             2 * self.uavAntGain - self.uavNoiseFigure) / 10)
        other_channel_arr = np.asarray(other_channel_list, dtype=np.int32)
        if self.uav_channels[i][j] in other_channel_list:
            index = np.where(other_channel_arr == self.uav_channels[i][j])
            for k in range(len(index[0])):
                ii, jj = pairs[index[0][k]]
                interferer_tx_idx = self.uav_pairs[ii][jj][0]
                uav_uav_interference += 10 ** (
                    (self.uav_powers[ii][jj] - self.UAVchannels_loss_db[interferer_tx_idx, receiver_idx, self.uav_channels[i][j]]
                     + 2 * self.uavAntGain - self.uavNoiseFigure) / 10
                )     #无人机内部干扰

        jam_arr = np.asarray(self.jammer_channels_list, dtype=np.int32)
        idx = np.where(jam_arr == self.uav_channels[i][j])[0]
        if idx.size > 0:
            time_eps = 1e-9
            jammer_switched_during_rx = float(self.jammer_time[1]) > time_eps
            if not jammer_switched_during_rx:     # 传输时间干扰机没换信道
                jammer_interference = 0.0
                for m in idx:
                    jammer_idx = self.jammer_index_list[m]
                    jammer_interference += 10 ** (
                        (self.jammer_power - self.Jammerchannels_loss_db[jammer_idx, receiver_idx, self.uav_channels[i][j]]
                         + self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10
                    )
                denom = float(max(0.0, self.uav_interference_scale)) * float(uav_uav_interference) + float(jammer_interference) + float(self.sig2)
                uav_rate = np.log2(1 + np.divide(uav_signal, denom))
                uav_rate *= self.bandwidth
                transmit_time = self.data_size / uav_rate


            else:    # 传输时间干扰机换了信道，判断干扰了前半段还是后半段
                for m in idx:
                    jammer_idx = self.jammer_index_list[m]
                    if m % 2 == 0:   # 后半段(self.jammer_channels_list先存入的后半段干扰信道序号）
                        jammer_interference_from_jammer0 += 10 ** ((self.jammer_power - self.Jammerchannels_loss_db[jammer_idx, receiver_idx, self.uav_channels[i][j]] +
                                                                 self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)

                denom0 = float(max(0.0, self.uav_interference_scale)) * float(uav_uav_interference) + float(jammer_interference_from_jammer0) + float(self.sig2)
                uav_rate = np.log2(1 + np.divide(uav_signal, denom0))
                uav_rate *= self.bandwidth
                transmit_time1 = self.data_size / uav_rate

                for m in idx:
                    jammer_idx = self.jammer_index_list[m]
                    if m % 2 == 1:   # 前半段
                        jammer_interference_from_jammer1 += 10 ** ((self.jammer_power - self.Jammerchannels_loss_db[jammer_idx, receiver_idx, self.uav_channels[i][j]] +
                                                                 self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)
                denom1 = float(max(0.0, self.uav_interference_scale)) * float(uav_uav_interference) + float(jammer_interference_from_jammer1) + float(self.sig2)
                uav_rate = np.log2(1 + np.divide(uav_signal, denom1))
                uav_rate *= self.bandwidth
                transmit_time2 = self.data_size / uav_rate

                if transmit_time2 > self.jammer_time[1]:
                    transmit_time1 = (self.data_size - uav_rate * self.jammer_time[1]) / (self.data_size / transmit_time1)
                    transmit_time = transmit_time1 + self.jammer_time[1]
                else:
                    transmit_time = transmit_time2

        else:
            denom = float(max(0.0, self.uav_interference_scale)) * float(uav_uav_interference) + float(self.sig2)
            uav_rate = np.log2(1 + np.divide(uav_signal, denom))
            uav_rate *= self.bandwidth
            transmit_time = self.data_size / uav_rate
        suc = 0
        time = 0
        if transmit_time < self.t_Rx:
            suc = 1
            time = transmit_time
        else:
            suc = -3
            time = self.t_Rx

        return time, suc

    def get_reward(self):
        uav_rewards = np.zeros([self.n_ch], dtype=float)

        if self.jammer_channels_list == []:
            for i in range(self.n_jammer):
                self.jammer_channels_list.append(self.jammer_channels[i])
                self.jammer_index_list.append(i)
            self.jammer_time[0] = self.t_Rx
            # print("jammer_channels", self.jammer_channels_list)

        tra = 0
        rec = 0
        success_cnt = np.zeros([self.n_ch], dtype=np.float32)
        
        while tra < self.n_ch:
            other_channel_list = []
            pairs = []
            for i in range(self.n_ch):
                for j in range(self.n_des):
                    if i==tra and j==rec:
                        continue
                    other_channel_list.append(self.uav_channels[i][j])      #排除自己通信信道的其他信道
                    pairs.append([i, j])

            tra_time, suc = self.compute_reward(tra, rec, other_channel_list, pairs)  # 传输时间
            self.rew_suc += suc
            if suc == 1:
                success_cnt[tra] += 1.0
            energy = 10 ** (self.uav_powers[tra][rec] / 10 - 3) * tra_time      # 能量奖励
            self.rew_energy += energy
            jump = self.uav_jump_count[tra] # 跳频开销
            self.rew_jump += jump
            # uav_rewards[tra] += (0.5 * energy - 0.5 * jump)
            max_energy = 10 ** (self.uav_power_max / 10 - 3) * self.t_Rx + 1e-12
            norm_energy = float(energy / max_energy)
            uav_rewards[tra] += suc - (self.reward_energy_weight * norm_energy + self.reward_jump_weight * jump)
            # print(energy, jump, suc)
            # 保留两位小数
            rec += 1
            if rec == self.n_des:
                tra += 1
                rec = 0

        # Fairness: penalize the whole team if any cluster falls below a minimum success rate.
        if float(self.fairness_weight) > 0.0 and float(self.fairness_min_success_rate) > 0.0:
            success_rate_per_cluster = success_cnt / float(self.n_des)
            shortfall = np.maximum(0.0, float(self.fairness_min_success_rate) - success_rate_per_cluster)
            team_penalty = float(self.fairness_weight) * float(np.mean(shortfall))
            uav_rewards -= team_penalty

        self.jammer_channels_list = []
        self.jammer_index_list = []
        self.jammer_time = np.zeros([2])
        self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)
        return uav_rewards

    def reward_details(self):
        return self.rew_energy / self.n_ch, self.rew_jump / (self.n_ch * self.n_des), self.rew_suc / (self.n_ch * self.n_des)

    def clear_reward(self):
        self.rew_energy = 0
        self.rew_jump = 0
        self.rew_suc = 0

    def renew_jammer_channels_after_Rx(self):
        jammer_channels_after_Rx(self)

    def renew_jammer_channels_after_learn(self):
        jammer_channels_after_learn(self)

    # 更新簇头的位置，在无人机获知网络状态信息阶段，簇头无人机根据方向，delta距离来更新其位置
    def renew_positions_of_chs(self):
        # ========================================================
        # This function update the position of each ch
        # ===========================================================
        self.xyz_delta_dis = [[0, 0, 0] for _ in range(self.n_ch)]  # 拷贝成[[0,0],[0,0],[0,0],[0,0]]
        for ch in range(self.n_ch):
            i = self.ch_list[ch]
            delta_distance = self.uavs[i].velocity * self.timestep
            d = self.uavs[i].direction
            p = self.uavs[i].p

            x_delta_distance = delta_distance * math.cos(d) * math.cos(p)
            y_delta_distance = delta_distance * math.sin(d) * math.cos(p)
            z_delta_distance = delta_distance * math.sin(p)

            xpos = self.uavs[i].position[0] + x_delta_distance
            ypos = self.uavs[i].position[1] + y_delta_distance
            zpos = self.uavs[i].position[2] + z_delta_distance

            if (xpos < 0):
                self.uavs[i].direction = math.pi - self.uavs[i].direction
                xpos = abs(x_delta_distance) - self.uavs[i].position[0]

            if (xpos > self.length):
                self.uavs[i].direction = math.pi - self.uavs[i].direction
                xpos = 2 * self.length - abs(x_delta_distance) - self.uavs[i].position[0]

            if (ypos < 0):
                self.uavs[i].direction = 2 * math.pi - self.uavs[i].direction
                ypos = abs(y_delta_distance) - self.uavs[i].position[1]

            if (ypos > self.width):
                self.uavs[i].direction = 2 * math.pi - self.uavs[i].direction
                ypos = 2 * self.width - abs(y_delta_distance) - self.uavs[i].position[1]

            if (zpos < self.low_height):
                self.uavs[i].p = 2 * math.pi - self.uavs[i].p
                zpos = 2 * self.low_height - self.uavs[i].position[2] + abs(z_delta_distance)

            if (zpos > self.high_height):
                self.uavs[i].p = 2 * math.pi - self.uavs[i].p
                zpos = 2 * self.high_height - self.uavs[i].position[2] - abs(z_delta_distance)

            self.xyz_delta_dis[ch] = [x_delta_distance, y_delta_distance, z_delta_distance]
            self.uavs[i].position = [xpos, ypos, zpos]
            # self.rps[i].position = self.uavs[i].position

            self.uavs[i].velocity = self.k * self.uavs[i].velocity + (1 - self.k) * np.average(
                self.uavs[i].uav_velocity) + (
                                            1 - self.k ** 2) ** 0.5 * np.random.normal(0, self.sigma)
            self.uavs[i].direction = self.k * self.uavs[i].direction + (1 - self.k) * np.average(
                self.uavs[i].uav_direction) + (
                                             1 - self.k ** 2) ** 0.5 * np.random.normal(0, self.sigma)
            self.uavs[i].p = self.k * self.uavs[i].p + (1 - self.k) * np.average(self.uavs[i].uav_p) + (
                        1 - self.k ** 2) ** 0.5 * np.random.normal(0, self.sigma)

            self.uavs[i].uav_velocity.append(self.uavs[i].velocity)
            self.uavs[i].uav_direction.append(self.uavs[i].direction)
            self.uavs[i].uav_p.append(self.uavs[i].p)

    def renew_positions_of_cms(self):
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cm_id = self.uavs[ch_id].connections
            ch_pos = [self.uavs[ch_id].position[0], self.uavs[ch_id].position[1], self.uavs[ch_id].position[2]]
            # 簇头位置没变化时，即最开始的时候
            if self.xyz_delta_dis[i] == [0, 0, 0]:
                for j in cm_id:
                    # 更新参考点的位置
                    R1 = random.uniform(0, self.max_distance1)
                    d1 = random.uniform(0, 2 * math.pi)
                    p1 = random.uniform(0, 2 * math.pi)

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                    while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) \
                           or (rp_ypos > self.width) or (rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                        R1 = random.uniform(0, R1)
                        d1 = random.uniform(0, 2 * math.pi)
                        p1 = random.uniform(0, 2 * math.pi)

                        rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                        rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                        rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                    rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                    self.rps[j].position = rp_pos

                    # 更新簇内节点的位置
                    R2 = random.uniform(0, self.max_distance2)
                    d2 = random.uniform(0, 2 * math.pi)
                    p2 = random.uniform(0, 2 * math.pi)

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                    self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

                    while ((self.uavs[j].position[0] < 0) or (self.uavs[j].position[0] > self.length) or (
                            self.uavs[j].position[1] < 0) \
                           or (self.uavs[j].position[1] > self.width) or (
                                   self.uavs[j].position[2] < self.low_height) or (
                                   self.uavs[j].position[2] > self.high_height)):
                        R2 = random.uniform(0, self.max_distance2)
                        d2 = random.uniform(0, 2 * math.pi)
                        p2 = random.uniform(0, 2 * math.pi)

                        cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                        cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                        cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                        self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]
            else:
                for j in cm_id:
                    # 更新参考点的位置
                    rp_xpos = self.rps[j].position[0] + self.xyz_delta_dis[i][0]
                    rp_ypos = self.rps[j].position[1] + self.xyz_delta_dis[i][1]
                    rp_zpos = self.rps[j].position[2] + self.xyz_delta_dis[i][2]

                    if (rp_xpos < 0):
                        rp_xpos = abs(self.xyz_delta_dis[i][0]) - self.rps[j].position[0]

                    if (rp_xpos > self.length):
                        rp_xpos = 2 * self.length - abs(self.xyz_delta_dis[i][0]) - self.rps[j].position[0]

                    if (rp_ypos < 0):
                        rp_ypos = abs(self.xyz_delta_dis[i][1]) - self.rps[j].position[1]

                    if (rp_ypos > self.width):
                        rp_ypos = 2 * self.width - self.rps[j].position[1] - abs(self.xyz_delta_dis[i][1])

                    if (rp_zpos < self.low_height):
                        rp_zpos = 2 * self.low_height - self.rps[j].position[2] + abs(self.xyz_delta_dis[i][2])

                    if (rp_zpos > self.high_height):
                        rp_zpos = 2 * self.high_height - self.rps[j].position[2] - abs(self.xyz_delta_dis[i][2])

                    rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                    self.rps[j].position = rp_pos

                    # 更新簇内节点的位置
                    R2 = random.uniform(0, self.max_distance2)
                    d2 = random.uniform(0, 2 * math.pi)
                    p2 = random.uniform(0, 2 * math.pi)

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)

                    while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (
                            cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                        R2 = random.uniform(0, self.max_distance2)
                        d2 = random.uniform(0, 2 * math.pi)
                        p2 = random.uniform(0, 2 * math.pi)

                        cm_xpos = self.rps[j].position[0] + R2 * math.cos(d2) * math.cos(p2)
                        cm_ypos = self.rps[j].position[1] + R2 * math.sin(d2) * math.cos(p2)
                        cm_zpos = self.rps[j].position[2] + R2 * math.sin(p2)
                    self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

    def renew_positions_of_jammers(self):
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================

        i = 0
        # for i in range(len(self.position)):
        while (i < len(self.jammers)):
            # print ('start iteration ', i)
            # print(self.position, len(self.position), self.direction)
            delta_distance = self.jammers[i].velocity * self.t_collect
            d = self.jammers[i].direction
            p = self.jammers[i].p

            x_delta_distance = delta_distance * math.cos(d) * math.cos(p)
            y_delta_distance = delta_distance * math.sin(d) * math.cos(p)
            z_delta_distance = delta_distance * math.sin(p)

            xpos = self.jammers[i].position[0] + x_delta_distance
            ypos = self.jammers[i].position[1] + y_delta_distance
            zpos = self.jammers[i].position[2] + z_delta_distance

            if (self.jammers[i].position[0] + x_delta_distance < 0):
                self.jammers[i].direction = math.pi - self.jammers[i].direction
                xpos = abs(x_delta_distance) - self.jammers[i].position[0]

            if (self.jammers[i].position[0] + x_delta_distance > self.length):
                self.jammers[i].direction = math.pi - self.jammers[i].direction
                xpos = 2 * self.length - abs(x_delta_distance) - self.jammers[i].position[0]

            if (self.jammers[i].position[1] + y_delta_distance < 0):
                self.jammers[i].direction = 2 * math.pi - self.jammers[i].direction
                ypos = abs(y_delta_distance) - self.jammers[i].position[1]

            if (self.jammers[i].position[1] + y_delta_distance > self.width):
                self.jammers[i].direction = 2 * math.pi - self.jammers[i].direction
                ypos = 2 * self.width - abs(y_delta_distance) - self.jammers[i].position[1]

            if (self.jammers[i].position[2] + z_delta_distance < self.low_height):
                self.jammers[i].p = 2 * math.pi - self.jammers[i].p
                zpos = 2 * self.low_height - self.jammers[i].position[2] + abs(z_delta_distance)

            if (self.jammers[i].position[2] + z_delta_distance > self.high_height):
                self.jammers[i].p = 2 * math.pi - self.jammers[i].p
                zpos = 2 * self.high_height - (self.jammers[i].position[2] + abs(z_delta_distance))

            self.jammers[i].position = [xpos, ypos, zpos]

            self.jammers[i].velocity = self.k * self.jammers[i].velocity + (1 - self.k) * np.average(
                self.jammers[i].jammer_velocity) + (1 - self.k) ** 0.5 * np.random.normal(0, self.sigma)
            self.jammers[i].direction = self.k * self.jammers[i].direction + (1 - self.k) * np.average(
                self.jammers[i].jammer_direction) + (1 - self.k) ** 0.5 * np.random.normal(0, self.sigma)
            self.jammers[i].p = self.k * self.jammers[i].p + (1 - self.k) * np.average(self.jammers[i].jammer_p) + (
                        1 - self.k) ** 0.5 * np.random.normal(0, self.sigma)

            self.jammers[i].jammer_velocity.append(self.jammers[i].velocity)
            self.jammers[i].jammer_direction.append(self.jammers[i].direction)
            self.jammers[i].jammer_p.append(self.jammers[i].p)
            i += 1

    def renew_channels(self):
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        uavs_ = [u.position for u in self.uavs]
        uav_positions = uavs_
        jammer_positions = [j.position for j in self.jammers]
        self.Jammerchannels.update_positions(jammer_positions, uav_positions)
        self.UAVchannels.update_positions(uav_positions)
        self.Jammerchannels.update_pathloss()
        self.UAVchannels.update_pathloss()
        uav_channels_loss_db = np.repeat(self.UAVchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        uav_channels_loss_db = (
            uav_channels_loss_db + self.channel_loss_db.reshape(1, 1, -1) + self.uav_channel_selectivity_db
        )
        self.UAVchannels_loss_db = uav_channels_loss_db
        jammer_channels_loss_db = np.repeat(self.Jammerchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        jammer_channels_loss_db = (
            jammer_channels_loss_db
            + self.channel_loss_db.reshape(1, 1, -1)
            + self.jammer_channel_selectivity_db
        )
        self.Jammerchannels_loss_db = jammer_channels_loss_db

    def act(self):
        self.renew_jammer_channels_after_Rx()
        reward = self.get_reward()
        self.renew_positions_of_chs()
        self.renew_positions_of_cms()
        if self.is_jammer_moving:
            self.renew_positions_of_jammers()
        self.renew_channels()
        return reward

        # 分解动作值的操作，由簇头到每个选择的通信信道数

    def decomposition_action(self, action):
        for i in range(self.n_ch):
            discrete_action, all_action_params = action[i]
            discrete_action = int(discrete_action)

            all_action_params = np.asarray(all_action_params, dtype=np.float32).reshape(-1)
            if all_action_params.size != self.total_param_dim:
                raise ValueError(
                    f"Invalid action_params size: got {all_action_params.size}, expected {self.total_param_dim}"
                )

            # Power parameters for the chosen discrete action (normalized [0, 1])
            start = discrete_action * self.param_dim_per_action
            end = start + self.param_dim_per_action
            power_norm = np.clip(all_action_params[start:end], 0.0, 1.0)

            decoded = discrete_action
            for j in range(self.n_des):
                channel_last = self.uav_channels[i][j]
                self.uav_channels[i][j] = int(decoded % self.n_channel)
                self.uav_powers[i][j] = float(
                    self.uav_power_min + float(power_norm[j]) * (self.uav_power_max - self.uav_power_min)
                )
                if self.uav_channels[i][j] != channel_last:
                    self.uav_jump_count[i] += 1
                decoded = int(decoded / self.n_channel)

    def generate_p_trans(self, mode = 1):
        return generate_jammer_p_trans(self.jammer_state_dim, mode=mode)

    def set_p(self, p_trans):
        self.p_trans = p_trans

    def reset(self, p_trans):
        self.set_p(p_trans)
        self.new_random_game()
        state = self.get_state()
        return state

    def step(self, a):
        action = deepcopy(a)
        self.decomposition_action(action)
        reward = self.act()
        state_next = self.get_state()  # 得到新的状态
        self.renew_jammer_channels_after_learn()
        return state_next, reward, False, {}

    def smooth(self, data, sm=1):
        smooth_data = []
        # if sm > 1:
        #     for d in data:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        smooth_data.append(d)
        return smooth_data

    def plot(self, cost_list):
        y_data = self.smooth(cost_list, 19)
        x_data = np.arange(len(cost_list))
        # sns.set(style="darkgrid", font_scale=1.5)
        # sns.tsplot(time=x_data, data=y_data, color='b', linestyle='-')
        np.savetxt('DRQN_po.txt', y_data[0], fmt='%f')
        np.save('DRQN.npy',  y_data[0])


        plt.plot(x_data, y_data[0])
        plt.ylabel('DRQN__reward')
        plt.xlabel('training Episode')
        # plt.ylim(0.5, 1.0)
        plt.show()

        plt.plot(x_data, cost_list)
        plt.ylabel('reward')
        plt.xlabel('training Episode')
        # plt.ylim(0.5, 1.0)
        plt.show()
