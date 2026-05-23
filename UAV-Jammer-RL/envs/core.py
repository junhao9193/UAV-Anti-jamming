import math
import random
from collections import deque
from copy import deepcopy

from gymnasium import spaces

import numpy as np

from scipy.special import comb, perm
from itertools import combinations, permutations

from envs.channels import UAVchannels, Jammerchannels
from envs.config import load_env_config
from envs.entities import UAV, Jammer, RP
from envs.jammer_policy import (
    JammerEvent,
    generate_p_trans as generate_jammer_p_trans,
    init_jammer_state,
    record_jammer_observation,
    renew_jammer_channels_after_Rx as jammer_channels_after_Rx,
    renew_jammer_channels_after_learn as jammer_channels_after_learn,
)

class Environ:
    """Project-specific multi-agent environment.

    This class intentionally keeps the repository's legacy multi-agent API:
    `reset(p_trans=None) -> state` and `step(actions) -> (state, reward, done, info)`.
    It does not inherit from `gymnasium.Env` because its spaces are per-agent
    lists and its reset/step signatures are not the Gymnasium single-agent API.
    """

    def __init__(self, config=None, config_path=None):
        cfg = load_env_config(config=config, config_path=config_path)

        self.env_seed = cfg.get("env_seed", None)
        seed_sequence = np.random.SeedSequence(None if self.env_seed is None else int(self.env_seed))
        env_seed_seq, fast_fading_seed_seq, jammer_seed_seq = seed_sequence.spawn(3)
        self._rng = np.random.default_rng(env_seed_seq)

        def _seed_int_from_sequence(seed_seq):
            return int(seed_seq.generate_state(1, dtype=np.uint32)[0])

        self.length = cfg["length"]
        self.width = cfg["width"]
        self.low_height = cfg["low_height"]
        self.high_height = cfg["high_height"]
        self.BS_position = cfg.get(
            "BS_position",
            [self.length / 2, self.width / 2, (self.low_height + self.high_height) / 2],  # Suppose the BS is in the center
        )
        self.k = float(cfg["k"])
        if not (0.0 <= self.k <= 1.0):
            raise ValueError(f"k must be in [0, 1] for the Gauss-Markov mobility model, got {self.k}")
        self.sigma = float(cfg["sigma"])
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative for the Gauss-Markov mobility model, got {self.sigma}")

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
        self.max_episode_steps = int(cfg.get("max_episode_steps", 1000))
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
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
        self.n_uav_pair = self.n_ch * self.n_des  # total communication links
        self.n_jammer = cfg["n_jammer"]  # number of jammers
        self.n_channel = cfg["n_channel"]  # int(self.n_ch+self.n_jammer-1)  # number of channels
        self.channel_indexes = np.arange(self.n_channel)
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

        # Fast fading (small-scale), temporally correlated Rayleigh (AR(1) on complex coefficients).
        # We model per-link, per-channel complex fading h_t:
        #   h_t = rho * h_{t-1} + sqrt(1-rho^2) * w_t,   w_t ~ CN(0,1)
        # The power-gain in dB is 20*log10(|h_t|). Effective loss (dB) becomes:
        #   loss_db = base_loss_db - fast_fading_db
        # so received power: P_rx_dbm = P_tx_dbm - loss_db + gains - NF.
        self.enable_fast_fading = bool(cfg.get("enable_fast_fading", True))
        self.fast_fading_rho = float(cfg.get("fast_fading_rho", 0.95))
        if not (0.0 <= self.fast_fading_rho < 1.0):
            raise ValueError(f"fast_fading_rho must be in [0,1), got {self.fast_fading_rho}")
        self.fast_fading_eps = float(cfg.get("fast_fading_eps", 1e-12))
        self.fast_fading_db_clip_low = cfg.get("fast_fading_db_clip_low", None)
        self.fast_fading_db_clip_high = cfg.get("fast_fading_db_clip_high", None)
        if self.fast_fading_db_clip_low is not None:
            self.fast_fading_db_clip_low = float(self.fast_fading_db_clip_low)
        if self.fast_fading_db_clip_high is not None:
            self.fast_fading_db_clip_high = float(self.fast_fading_db_clip_high)
        if (self.fast_fading_db_clip_low is None) != (self.fast_fading_db_clip_high is None):
            raise ValueError("fast_fading_db_clip_low/high must both be set or both be null")
        if (
            self.fast_fading_db_clip_low is not None
            and self.fast_fading_db_clip_high is not None
            and not (self.fast_fading_db_clip_low < self.fast_fading_db_clip_high)
        ):
            raise ValueError(
                "fast_fading_db_clip_low must be < fast_fading_db_clip_high, got "
                f"{self.fast_fading_db_clip_low} vs {self.fast_fading_db_clip_high}"
            )
        fast_fading_seed = cfg.get("fast_fading_seed", None)
        if fast_fading_seed is None:
            self._fast_fading_rng = np.random.default_rng(fast_fading_seed_seq)
        else:
            self._fast_fading_rng = np.random.default_rng(int(fast_fading_seed))
        # E[20*log10(|h|)] for Rayleigh (CN(0,1)/sqrt(2)) is approx -2.507 dB.
        # Subtract this mean so that fast fading fluctuates around 0 dB.
        # Derivation: E[ln|h|] = -gamma/2 for Rayleigh(sigma=1/sqrt(2)),
        # so E[20*log10|h|] = (20/ln10)*(-gamma/2) = -10*gamma/ln10.
        self._rayleigh_mean_db = float(-10.0 * np.euler_gamma / np.log(10.0))
        self._uav_fast_h = None  # complex64, (n_uav,n_uav,n_channel)
        self._jammer_fast_h = None  # complex64, (n_jammer,n_uav,n_channel)

        self.max_distance1 = cfg["max_distance1"]
        self.max_distance2 = cfg["max_distance2"]

        self.is_jammer_moving = cfg["is_jammer_moving"]
        self.p_trans_seed = int(cfg.get("p_trans_seed", 0))
        jammer_state_dim_local = int(perm(self.n_channel, self.n_jammer))
        self.p_trans_preferred_next_states = int(cfg.get("p_trans_preferred_next_states", 2))
        if not (0 <= self.p_trans_preferred_next_states <= jammer_state_dim_local):
            raise ValueError(
                "p_trans_preferred_next_states must be in [0, jammer_state_dim], got "
                f"{self.p_trans_preferred_next_states}"
            )
        self.p_trans_preference_strength = float(cfg.get("p_trans_preference_strength", 0.5))
        if self.p_trans_preference_strength < 0.0:
            raise ValueError(
                f"p_trans_preference_strength must be non-negative, got {self.p_trans_preference_strength}"
            )
        self.jammer_reactive_beta = float(cfg.get("jammer_reactive_beta", 0.0))
        self.jammer_memory_window = int(cfg.get("jammer_memory_window", 4))
        if self.jammer_memory_window < 1:
            raise ValueError(f"jammer_memory_window must be >= 1, got {self.jammer_memory_window}")
        self._jammer_observed_channel_history = deque(maxlen=self.jammer_memory_window)
        self.jammer_reactive_observe_prob = float(cfg.get("jammer_reactive_observe_prob", 1.0))
        if not (0.0 <= self.jammer_reactive_observe_prob <= 1.0):
            raise ValueError(
                "jammer_reactive_observe_prob must be in [0,1], got "
                f"{self.jammer_reactive_observe_prob}"
            )
        jammer_seed = cfg.get("jammer_seed", None)
        if jammer_seed is None:
            jammer_seed = _seed_int_from_sequence(jammer_seed_seq)
        # Jammer state sampling and partial observation use a private RNG so they
        # remain reproducible without depending on the process-global `random` state.
        self._jammer_state_rng = random.Random(int(jammer_seed))
        self.reward_energy_weight = cfg["reward_energy_weight"]
        self.reward_jump_weight = cfg["reward_jump_weight"]
        self.fairness_min_success_rate = float(cfg.get("fairness_min_success_rate", 0.0))
        self.fairness_weight = float(cfg.get("fairness_weight", 0.0))
        self.csi_pathloss_offset = float(cfg.get("csi_pathloss_offset", 80.0))
        self.csi_pathloss_scale = float(cfg.get("csi_pathloss_scale", 60.0))
        self.csi_noise_std = float(cfg.get("csi_noise_std", 0.0))
        if self.csi_noise_std < 0.0:
            raise ValueError(f"csi_noise_std must be non-negative, got {self.csi_noise_std}")
        self.csi_clip = bool(cfg.get("csi_clip", True))
        # Jammer behavior: Markov transition matrix plus optional reactive bias
        # from partially observed UAV channel choices.
        self.policy = None  # 对应算法
        self.jammer_channels = [0 for _ in range(self.n_jammer)]
        self.jammer_events = []

        self.uav_list = list(range(int(self.n_uav)))
        self.ch_list = self._sample_without_replacement(self.uav_list, self.n_ch)
        ch_set = set(self.ch_list)
        self.cm_list = [uav_id for uav_id in self.uav_list if uav_id not in ch_set]
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
        self.agent_action_spaces = [
            spaces.Tuple(
                (
                    spaces.Discrete(self.action_dim),
                    spaces.Box(low=0.0, high=1.0, shape=(self.total_param_dim,), dtype=np.float32),
                )
            )
            for _ in range(self.n_ch)
        ]
        self.action_space = self.agent_action_spaces  # Legacy alias used by existing training code.

        #与奖励相关参数
        self.uav_jump_count = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        self.rew_energy = 0
        self.rew_jump = 0
        self.rew_suc = 0

        self.episode_step = 0
        self._episode_initialized = False

        # 初始化静态观察/干扰状态空间。Episode state is created lazily by reset().
        self.all_observed_states()
        self.set_p(self.generate_p_trans())
        self.state_dim = self._compute_state_dim()
        self.agent_observation_spaces = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.state_dim,))
            for _ in range(self.n_ch)
        ]
        self.observation_space = self.agent_observation_spaces  # Legacy alias used by existing training code.

    def _sample_complex_gaussian(self, shape):
        real = self._fast_fading_rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        imag = self._fast_fading_rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)

    def _init_fast_fading(self) -> None:
        if not self.enable_fast_fading:
            self._uav_fast_h = None
            self._jammer_fast_h = None
            return
        self._uav_fast_h = self._sample_complex_gaussian((int(self.n_uav), int(self.n_uav), int(self.n_channel)))
        self._jammer_fast_h = self._sample_complex_gaussian(
            (int(self.n_jammer), int(self.n_uav), int(self.n_channel))
        )

    def _update_fast_fading(self) -> None:
        if not self.enable_fast_fading:
            return
        if self._uav_fast_h is None or self._jammer_fast_h is None:
            self._init_fast_fading()
            return
        rho = float(self.fast_fading_rho)
        sigma = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        self._uav_fast_h = (rho * self._uav_fast_h + sigma * self._sample_complex_gaussian(self._uav_fast_h.shape)).astype(
            np.complex64
        )
        self._jammer_fast_h = (
            rho * self._jammer_fast_h + sigma * self._sample_complex_gaussian(self._jammer_fast_h.shape)
        ).astype(np.complex64)

    def all_observed_states(self):
        self.observed_state_list = []
        observed_state = 0
        self.all_observed_states_list = []

        self.all_jammer_states_list = []
        self.jammer_state_dim = int(perm(self.n_channel, self.n_jammer)) # perm()全排列
        self.all_jammer_states_list.extend(list(permutations(self.channel_indexes, self.n_jammer))) # permutations给定一个数组集合，返回所有可能的排列。
        self._jammer_state_to_index = {
            tuple(int(ch) for ch in state): idx
            for idx, state in enumerate(self.all_jammer_states_list)
        }
        self._jammer_state_channel_counts = np.zeros(
            (int(self.jammer_state_dim), int(self.n_channel)),
            dtype=np.float64,
        )
        for idx, state in enumerate(self.all_jammer_states_list):
            self._jammer_state_channel_counts[idx, np.asarray(state, dtype=np.int32)] += 1.0
        
        self.observed_state_dim = int(comb(self.n_channel, self.n_jammer)) # comb返回从 n_channel 种可能性中选择n_jammer个无序结果的方式数量，无重复，也称为组合。
        self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))

    def _compute_state_dim(self):
        if self.policy == "Sensing_Based_Method":
            return int(self.n_channel)
        if self.policy == "Q_learning":
            raise NotImplementedError("Q_learning policy is not supported in continuous-power (MP-DQN) mode.")
        return int((self.n_des + 1) * self.n_channel)

    def _sample_without_replacement(self, population, k):
        population = list(population)
        if int(k) > len(population):
            raise ValueError(f"Cannot sample {k} items from population of size {len(population)}")
        indices = self._rng.choice(len(population), size=int(k), replace=False)
        return [population[int(idx)] for idx in np.asarray(indices).reshape(-1)]

    def renew_uavs(self):
        for i in range(self.n_ch):
            # 更新簇头无人机的位置
            ch_id = self.ch_list[i]

            start_velocity = float(self._rng.uniform(10.0, 20.0))
            start_direction = float(self._rng.uniform(0.0, 2 * math.pi))
            start_p = float(self._rng.uniform(0.0, 2 * math.pi))

            ch_xpos = float(self._rng.uniform(0.0, self.length))
            ch_ypos = float(self._rng.uniform(0.0, self.width))
            ch_zpos = float(self._rng.uniform(self.low_height, self.high_height))
            start_position = [ch_xpos, ch_ypos, ch_zpos]

            self.uavs[ch_id] = UAV(start_position, start_direction, start_velocity, start_p)
            self.rps[ch_id] = RP(start_position)

    def renew_uav_clusters(self):
        cm_list = deepcopy(self.cm_list)
        rp_cm_list = deepcopy(self.rp_cm_list)
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cms = self._sample_without_replacement(cm_list, self.n_cm_for_a_ch)
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
                R1 = float(self._rng.uniform(0.0, self.max_distance1))
                d1 = float(self._rng.uniform(0.0, 2 * math.pi))
                p1 = float(self._rng.uniform(0.0, 2 * math.pi))

                rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) or (rp_ypos > self.width) or (
                        rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                    R1 = float(self._rng.uniform(0.0, R1))
                    d1 = float(self._rng.uniform(0.0, 2 * math.pi))
                    p1 = float(self._rng.uniform(0.0, 2 * math.pi))

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                # 簇内节点的位置设定
                R2 = float(self._rng.uniform(0.0, self.max_distance2))
                d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                p2 = float(self._rng.uniform(0.0, 2 * math.pi))

                cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_zpos + R2 * math.sin(p2)

                while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (
                        cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                    # 簇内节点的位置设定
                    R2 = float(self._rng.uniform(0.0, self.max_distance2))
                    d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                    p2 = float(self._rng.uniform(0.0, 2 * math.pi))

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

            cms_set = set(cms)
            rps_set = set(rps)
            cm_list = [cm_id for cm_id in cm_list if cm_id not in cms_set]
            rp_cm_list = [rp_id for rp_id in rp_cm_list if rp_id not in rps_set]

        # print(self.uav_clusters)
        # print(self.uav_pairs)

    def renew_jammers(self):
        if self.is_jammer_moving:
            for i in range(self.n_jammer):
                start_velocity = float(self._rng.uniform(10.0, 20.0))
                start_direction = float(self._rng.uniform(0.0, 2 * math.pi))
                start_p = float(self._rng.uniform(0.0, 2 * math.pi))

                xpos = float(self._rng.uniform(0.0, self.length))
                ypos = float(self._rng.uniform(0.0, self.width))
                zpos = float(self._rng.uniform(self.low_height, self.high_height))
                start_position = [xpos, ypos, zpos]

                self.jammers.append(Jammer(start_position, start_direction, start_velocity, start_p))

    def new_random_game(self):
        # self.all_observed_states()
        self.t_uav = 0.0
        self.t_jammer = 0.0
        self.episode_step = 0
        self._episode_initialized = True
        self._jammer_observed_channel_history.clear()
        # 一个发送机若有多个通信目标，每个元素是智能体为每个通信目标分配的信道，假设各不相同
        self.uav_channels = np.zeros([self.n_ch, self.n_des], dtype=np.int32)   # 每个智能体观察到的全局动作（假设智能体可以观察到其他智能体已经完成的动作）
        self.uav_powers = np.zeros([self.n_ch, self.n_des], dtype=np.float32)
        self.uav_jump_count = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        for i in range(self.n_ch):
            for j in range(self.n_des):
                self.uav_channels[i][j] = int(self._rng.integers(0, self.n_channel))
                self.uav_powers[i][j] = float(self._rng.uniform(self.uav_power_min, self.uav_power_max))  # dBm
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
        # Reset fast fading between episodes. The first `renew_channels()` will init h_0
        # (without applying an AR update), so reset state aligns with h_0 rather than h_1.
        self._uav_fast_h = None
        self._jammer_fast_h = None
        self.renew_channels()

    def get_state(self):
        if not self._episode_initialized:
            raise RuntimeError("Environment episode state is not initialized. Call reset() before get_state().")

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
                        if float(self._rng.random()) < self.p_md:
                            channels_observed[i] = 0  # 漏警
                        else:
                            channels_observed[i] = 1  # 发现干扰
                    else:
                        if float(self._rng.random()) < self.p_fa:
                            channels_observed[i] = 1  # 虚警
                        else:
                            channels_observed[i] = 0  # 发现未干扰
            return channels_observed

        else:
            # CSI：每条链路在每个信道上的 CSI（路径损耗 + 信道固定差异/选择性 + 可选快衰落）。
            # 形状: (n_ch, n_des, n_channel)
            tx_ids = self.uav_pairs[:, :, 0]
            rx_ids = self.uav_pairs[:, :, 1]
            csi = (
                self.UAVchannels_loss_db[tx_ids, rx_ids, :].astype(np.float32)
                - float(self.csi_pathloss_offset)
            ) / float(self.csi_pathloss_scale)
            if self.csi_noise_std > 0.0:
                csi = csi + self._rng.normal(0.0, self.csi_noise_std, size=csi.shape).astype(np.float32)
            if self.csi_clip:
                csi = np.clip(csi, -1.0, 1.0)

            # 频谱感知：连续的“信道能量图”作为观测（不采样成 0/1）
            # z_i(c) = w_J * I[c in C^J] + w_U * sum_{k!=i} I[c in C^k] + noise
            # 再做 z-score 标准化并 clip 到 [-1,1]，训练更稳定。
            if not isinstance(self.jammer_channels, list):
                jammer_ch_list = list(self.jammer_channels)
            else:
                jammer_ch_list = self.jammer_channels

            uav_used = np.zeros((self.n_ch, self.n_channel), dtype=np.float32)
            uav_used[
                np.arange(self.n_ch, dtype=np.int32)[:, None],
                np.asarray(self.uav_channels, dtype=np.int32),
            ] = 1.0

            # Range clipping: each cluster head only "sees" nearby jammers / nearby other cluster heads.
            ch_tx_ids = np.asarray(self.uav_pairs[:, 0, 0], dtype=np.int32)
            ch_positions = np.asarray([self.uavs[idx].position for idx in ch_tx_ids], dtype=np.float32)  # (n_ch,3)
            jammer_positions = (
                np.asarray([j.position for j in self.jammers], dtype=np.float32) if len(self.jammers) > 0 else None
            )

            z = np.zeros((self.n_ch, self.n_channel), dtype=np.float32)

            # jammer 占用（仅统计探测范围内的 jammer）
            jammer_ch_arr = np.asarray(jammer_ch_list, dtype=np.int32)
            jammer_seen = np.zeros((self.n_ch, self.n_channel), dtype=bool)
            if jammer_positions is None:
                jammer_seen[:, jammer_ch_arr] = True
            elif jammer_ch_arr.size > 0:
                d_j = np.linalg.norm(ch_positions[:, None, :] - jammer_positions[None, :, :], axis=2)
                visible_jammer = d_j <= float(self.sensing_jammer_range)
                row_idx = np.repeat(np.arange(self.n_ch, dtype=np.int32), jammer_ch_arr.size)
                ch_idx = np.tile(jammer_ch_arr, self.n_ch)
                visible_flat = visible_jammer.reshape(-1)
                jammer_seen[row_idx[visible_flat], ch_idx[visible_flat]] = True
            z += jammer_seen.astype(np.float32) * float(self.sensing_w_jammer)

            # 其他簇头占用（按簇头计数，不按 link 计数）
            d_ch = np.linalg.norm(ch_positions[:, None, :] - ch_positions[None, :, :], axis=2)
            visible_ch = d_ch <= float(self.sensing_uav_range)
            np.fill_diagonal(visible_ch, False)
            z += (visible_ch.astype(np.float32) @ uav_used) * float(self.sensing_w_uav)

            # 可选：感知噪声（默认 0，不引入随机性）
            if self.sensing_noise_std > 0.0:
                z += self._rng.normal(0.0, self.sensing_noise_std, size=z.shape).astype(np.float32)

            mu = np.mean(z, axis=1, keepdims=True)
            std = np.std(z, axis=1, keepdims=True)
            z_norm = np.divide(z - mu, std + 1e-12, out=np.zeros_like(z, dtype=np.float32), where=std >= 1e-6)
            channel_sensing = np.clip(z_norm, -1.0, 1.0).astype(np.float32)

            obs = np.concatenate([csi.reshape(self.n_ch, -1), channel_sensing], axis=1).astype(np.float32)
            return [obs[i] for i in range(self.n_ch)]

    def compute_reward(self, i, j, other_channel_list, pairs):
        uav_uav_interference = 0.0   # interference from other UAV transmitters (linear mW)

        transmitter_idx = self.uav_pairs[i][j][0]
        receiver_idx = self.uav_pairs[i][j][1]
        target_channel = int(self.uav_channels[i][j])
        uav_signal = 10 ** ((self.uav_powers[i][j] - self.UAVchannels_loss_db[transmitter_idx, receiver_idx, target_channel] +
                             2 * self.uavAntGain - self.uavNoiseFigure) / 10)
        other_channel_arr = np.asarray(other_channel_list, dtype=np.int32)
        if target_channel in other_channel_list:
            index = np.where(other_channel_arr == target_channel)
            for k in range(len(index[0])):
                ii, jj = pairs[index[0][k]]
                interferer_tx_idx = self.uav_pairs[ii][jj][0]
                uav_uav_interference += 10 ** (
                    (self.uav_powers[ii][jj] - self.UAVchannels_loss_db[interferer_tx_idx, receiver_idx, target_channel]
                     + 2 * self.uavAntGain - self.uavNoiseFigure) / 10
                )     #无人机内部干扰

        events = [
            event
            for event in self.jammer_events
            if int(event.channel) == target_channel and float(event.t_end) > float(event.t_start)
        ]
        boundaries = [0.0, float(self.t_Rx)]
        for event in events:
            boundaries.append(float(np.clip(event.t_start, 0.0, self.t_Rx)))
            boundaries.append(float(np.clip(event.t_end, 0.0, self.t_Rx)))
        boundaries = sorted(set(boundaries))

        remaining_data = float(self.data_size)
        transmit_time = float(self.t_Rx)
        time_eps = 1e-9
        interference_scale = float(max(0.0, self.uav_interference_scale))

        for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
            duration = float(segment_end - segment_start)
            if duration <= time_eps:
                continue

            jammer_interference = 0.0
            for event in events:
                if float(event.t_start) <= segment_start + time_eps and float(event.t_end) >= segment_end - time_eps:
                    jammer_idx = int(event.jammer_idx)
                    jammer_interference += 10 ** (
                        (
                            self.jammer_power
                            - self.Jammerchannels_loss_db[jammer_idx, receiver_idx, target_channel]
                            + self.jammerAntGain
                            + self.uavAntGain
                            - self.uavNoiseFigure
                        )
                        / 10
                    )

            denom = interference_scale * float(uav_uav_interference) + float(jammer_interference) + float(self.sig2)
            uav_rate = np.log2(1 + np.divide(uav_signal, denom))
            uav_rate *= self.bandwidth
            deliverable = float(uav_rate) * duration
            if deliverable + time_eps >= remaining_data:
                transmit_time = float(segment_start + remaining_data / float(uav_rate))
                break
            remaining_data -= deliverable

        suc = 0
        time = 0
        if transmit_time < self.t_Rx:
            suc = 1
            time = transmit_time
        else:
            suc = -3
            time = self.t_Rx

        return time, suc

    def _compute_link_delivery(self, receiver_idx, target_channel, uav_signal, uav_uav_interference, event_data):
        boundaries = [0.0, float(self.t_Rx)]
        for event_start, event_end, _ in event_data:
            boundaries.append(event_start)
            boundaries.append(event_end)
        boundaries = sorted(set(boundaries))

        remaining_data = float(self.data_size)
        transmit_time = float(self.t_Rx)
        time_eps = 1e-9
        interference_scale = float(max(0.0, self.uav_interference_scale))

        for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:]):
            duration = float(segment_end - segment_start)
            if duration <= time_eps:
                continue

            jammer_interference = 0.0
            for event_start, event_end, jammer_idx in event_data:
                if event_start <= float(segment_start) + time_eps and event_end >= float(segment_end) - time_eps:
                    jammer_interference += 10 ** (
                        (
                            self.jammer_power
                            - self.Jammerchannels_loss_db[jammer_idx, receiver_idx, target_channel]
                            + self.jammerAntGain
                            + self.uavAntGain
                            - self.uavNoiseFigure
                        )
                        / 10
                    )

            denom = interference_scale * float(uav_uav_interference) + float(jammer_interference) + float(self.sig2)
            uav_rate = np.log2(1 + np.divide(uav_signal, denom))
            uav_rate *= self.bandwidth
            deliverable = float(uav_rate) * duration
            if deliverable + time_eps >= remaining_data:
                transmit_time = float(segment_start + remaining_data / float(uav_rate))
                break
            remaining_data -= deliverable

        if transmit_time < self.t_Rx:
            return transmit_time, 1
        return float(self.t_Rx), -3

    def get_reward(self):
        if not self.jammer_events:
            self.jammer_events = [
                JammerEvent(jammer_idx=i, channel=int(self.jammer_channels[i]), t_start=0.0, t_end=float(self.t_Rx))
                for i in range(self.n_jammer)
            ]

        n_links = int(self.n_ch * self.n_des)
        pairs_flat = self.uav_pairs.reshape(n_links, 2)
        tx_idx = pairs_flat[:, 0].astype(np.int32, copy=False)
        rx_idx = pairs_flat[:, 1].astype(np.int32, copy=False)
        channels = np.asarray(self.uav_channels, dtype=np.int32).reshape(n_links)
        powers = np.asarray(self.uav_powers, dtype=np.float32).reshape(n_links)
        jumps = np.asarray(self.uav_jump_count, dtype=np.float32).reshape(n_links)

        signal_loss = self.UAVchannels_loss_db[tx_idx, rx_idx, channels]
        uav_signal = 10 ** ((powers - signal_loss + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)

        loss_to_receivers = self.UAVchannels_loss_db[tx_idx[None, :], rx_idx[:, None], channels[:, None]]
        received_from_links = 10 ** (
            (powers[None, :] - loss_to_receivers + 2 * self.uavAntGain - self.uavNoiseFigure) / 10
        )
        same_channel = channels[None, :] == channels[:, None]
        np.fill_diagonal(same_channel, False)
        uav_uav_interference = np.sum(
            np.where(same_channel, received_from_links, 0.0),
            axis=1,
            dtype=np.float64,
        )

        events_by_channel = [[] for _ in range(self.n_channel)]
        for event in self.jammer_events:
            if float(event.t_end) > float(event.t_start):
                ch = int(event.channel)
                if 0 <= ch < self.n_channel:
                    events_by_channel[ch].append(
                        (
                            float(np.clip(event.t_start, 0.0, self.t_Rx)),
                            float(np.clip(event.t_end, 0.0, self.t_Rx)),
                            int(event.jammer_idx),
                        )
                    )

        transmit_time = np.empty(n_links, dtype=np.float64)
        suc_arr = np.empty(n_links, dtype=np.float32)
        for link_idx in range(n_links):
            tra_time, suc = self._compute_link_delivery(
                receiver_idx=int(rx_idx[link_idx]),
                target_channel=int(channels[link_idx]),
                uav_signal=float(uav_signal[link_idx]),
                uav_uav_interference=float(uav_uav_interference[link_idx]),
                event_data=events_by_channel[int(channels[link_idx])],
            )
            transmit_time[link_idx] = float(tra_time)
            suc_arr[link_idx] = float(suc)

        energy = (10 ** (powers.astype(np.float64) / 10 - 3)) * transmit_time
        self.rew_suc += float(np.sum(suc_arr, dtype=np.float64))
        self.rew_energy += float(np.sum(energy, dtype=np.float64))
        self.rew_jump += float(np.sum(jumps, dtype=np.float64))

        max_energy = 10 ** (self.uav_power_max / 10 - 3) * self.t_Rx + 1e-12
        norm_energy = energy / max_energy
        link_rewards = suc_arr.astype(np.float64) - (
            float(self.reward_energy_weight) * norm_energy + float(self.reward_jump_weight) * jumps
        )
        uav_rewards = np.sum(link_rewards.reshape(self.n_ch, self.n_des), axis=1)
        success_cnt = np.sum((suc_arr.reshape(self.n_ch, self.n_des) == 1.0), axis=1).astype(np.float32)

        # Fairness: penalize the whole team if any cluster falls below a minimum success rate.
        if float(self.fairness_weight) > 0.0 and float(self.fairness_min_success_rate) > 0.0:
            success_rate_per_cluster = success_cnt / float(self.n_des)
            shortfall = np.maximum(0.0, float(self.fairness_min_success_rate) - success_rate_per_cluster)
            team_penalty = float(self.fairness_weight) * float(np.mean(shortfall))
            uav_rewards -= team_penalty

        self.jammer_events = []
        self.uav_jump_count = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
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

            noise_scale = (1 - self.k ** 2) ** 0.5
            self.uavs[i].velocity = (
                self.k * self.uavs[i].velocity
                + (1 - self.k) * self.uavs[i].mean_velocity
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )
            self.uavs[i].direction = (
                self.k * self.uavs[i].direction
                + (1 - self.k) * self.uavs[i].mean_direction
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )
            self.uavs[i].p = (
                self.k * self.uavs[i].p
                + (1 - self.k) * self.uavs[i].mean_p
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )

    def renew_positions_of_cms(self):
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cm_id = self.uavs[ch_id].connections
            ch_pos = [self.uavs[ch_id].position[0], self.uavs[ch_id].position[1], self.uavs[ch_id].position[2]]
            # 簇头位置没变化时，即最开始的时候
            if self.xyz_delta_dis[i] == [0, 0, 0]:
                for j in cm_id:
                    # 更新参考点的位置
                    R1 = float(self._rng.uniform(0.0, self.max_distance1))
                    d1 = float(self._rng.uniform(0.0, 2 * math.pi))
                    p1 = float(self._rng.uniform(0.0, 2 * math.pi))

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                    while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) \
                           or (rp_ypos > self.width) or (rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                        R1 = float(self._rng.uniform(0.0, R1))
                        d1 = float(self._rng.uniform(0.0, 2 * math.pi))
                        p1 = float(self._rng.uniform(0.0, 2 * math.pi))

                        rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                        rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                        rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                    rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                    self.rps[j].position = rp_pos

                    # 更新簇内节点的位置
                    R2 = float(self._rng.uniform(0.0, self.max_distance2))
                    d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                    p2 = float(self._rng.uniform(0.0, 2 * math.pi))

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                    self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

                    while ((self.uavs[j].position[0] < 0) or (self.uavs[j].position[0] > self.length) or (
                            self.uavs[j].position[1] < 0) \
                           or (self.uavs[j].position[1] > self.width) or (
                                   self.uavs[j].position[2] < self.low_height) or (
                                   self.uavs[j].position[2] > self.high_height)):
                        R2 = float(self._rng.uniform(0.0, self.max_distance2))
                        d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                        p2 = float(self._rng.uniform(0.0, 2 * math.pi))

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
                    R2 = float(self._rng.uniform(0.0, self.max_distance2))
                    d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                    p2 = float(self._rng.uniform(0.0, 2 * math.pi))

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)

                    while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (
                            cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                        R2 = float(self._rng.uniform(0.0, self.max_distance2))
                        d2 = float(self._rng.uniform(0.0, 2 * math.pi))
                        p2 = float(self._rng.uniform(0.0, 2 * math.pi))

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

            noise_scale = (1 - self.k ** 2) ** 0.5
            self.jammers[i].velocity = (
                self.k * self.jammers[i].velocity
                + (1 - self.k) * self.jammers[i].mean_velocity
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )
            self.jammers[i].direction = (
                self.k * self.jammers[i].direction
                + (1 - self.k) * self.jammers[i].mean_direction
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )
            self.jammers[i].p = (
                self.k * self.jammers[i].p
                + (1 - self.k) * self.jammers[i].mean_p
                + noise_scale * self._rng.normal(0.0, self.sigma)
            )
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
        self._update_fast_fading()
        uav_channels_loss_db = np.repeat(self.UAVchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        uav_channels_loss_db = (
            uav_channels_loss_db + self.channel_loss_db.reshape(1, 1, -1) + self.uav_channel_selectivity_db
        )
        if self.enable_fast_fading and self._uav_fast_h is not None:
            uav_fast_db = 20.0 * np.log10(np.abs(self._uav_fast_h) + float(self.fast_fading_eps)) - self._rayleigh_mean_db
            if self.fast_fading_db_clip_low is not None and self.fast_fading_db_clip_high is not None:
                uav_fast_db = np.clip(uav_fast_db, self.fast_fading_db_clip_low, self.fast_fading_db_clip_high)
            self.UAVchannels_loss_db = uav_channels_loss_db - uav_fast_db.astype(np.float32)
        else:
            self.UAVchannels_loss_db = uav_channels_loss_db
        jammer_channels_loss_db = np.repeat(self.Jammerchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        jammer_channels_loss_db = (
            jammer_channels_loss_db
            + self.channel_loss_db.reshape(1, 1, -1)
            + self.jammer_channel_selectivity_db
        )
        if self.enable_fast_fading and self._jammer_fast_h is not None:
            jammer_fast_db = 20.0 * np.log10(np.abs(self._jammer_fast_h) + float(self.fast_fading_eps)) - self._rayleigh_mean_db
            if self.fast_fading_db_clip_low is not None and self.fast_fading_db_clip_high is not None:
                jammer_fast_db = np.clip(
                    jammer_fast_db, self.fast_fading_db_clip_low, self.fast_fading_db_clip_high
                )
            self.Jammerchannels_loss_db = jammer_channels_loss_db - jammer_fast_db.astype(np.float32)
        else:
            self.Jammerchannels_loss_db = jammer_channels_loss_db

    def act(self):
        record_jammer_observation(self)
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
        if (
            isinstance(action, tuple)
            and len(action) == 2
            and isinstance(action[0], np.ndarray)
            and isinstance(action[1], np.ndarray)
        ):
            action_discrete_arr = np.asarray(action[0], dtype=np.int64).reshape(-1)
            action_params_arr = np.asarray(action[1], dtype=np.float32)
            if action_discrete_arr.size != self.n_ch:
                raise ValueError(f"Invalid action_discrete size: got {action_discrete_arr.size}, expected {self.n_ch}")
            if action_params_arr.shape != (self.n_ch, self.total_param_dim):
                raise ValueError(
                    "Invalid action_params shape: got "
                    f"{action_params_arr.shape}, expected ({self.n_ch}, {self.total_param_dim})"
                )
            action_iter = ((int(action_discrete_arr[i]), action_params_arr[i]) for i in range(self.n_ch))
        else:
            action_iter = (action[i] for i in range(self.n_ch))

        for i, (discrete_action, all_action_params) in enumerate(action_iter):
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
                    self.uav_jump_count[i][j] += 1
                decoded = int(decoded / self.n_channel)

    def generate_p_trans(self, rng=None):
        return generate_jammer_p_trans(
            self.jammer_state_dim,
            rng=rng,
            preferred_next_states=self.p_trans_preferred_next_states,
            preference_strength=self.p_trans_preference_strength,
        )

    def set_p(self, p_trans):
        p_arr = np.asarray(p_trans, dtype=np.float32)
        if p_arr.ndim != 2:
            raise ValueError(f"Invalid p_trans ndim={p_arr.ndim}. Expected a single 2-D Markov matrix.")
        if p_arr.shape != (int(self.jammer_state_dim), int(self.jammer_state_dim)):
            raise ValueError(
                "Invalid p_trans shape. Expected (D, D) with D=jammer_state_dim, got "
                f"{p_arr.shape} (D={self.jammer_state_dim})"
            )
        self.p_trans = p_arr

    def reset(self, p_trans=None):
        if p_trans is not None:
            self.set_p(p_trans)
        self.new_random_game()
        state = self.get_state()
        return state

    def step(self, a, return_info: bool = True):
        if not self._episode_initialized:
            raise RuntimeError("Environment episode state is not initialized. Call reset() before step().")

        # NOTE: `a` comes from the caller (and in our training it is freshly created each step).
        # `decomposition_action()` does not mutate the action container, so `deepcopy` is unnecessary
        # and can become a major overhead when running many env steps in parallel workers.
        self.decomposition_action(a)
        # Capture the start-of-slot jammer (J_k_start) BEFORE act(). act() internally calls
        # renew_jammer_channels_after_Rx() and then get_reward(), so by the time it returns
        # self.jammer_channels reflects the post-Rx-boundary state, not what the agent decided
        # against. The BCE label must be the start-of-slot jammer to match the predictor's
        # intended semantic at action-selection time.
        if return_info:
            jammer_channels_current = [int(ch) for ch in list(self.jammer_channels)]
            jammer_channels_current_multi_hot = np.zeros([self.n_channel], dtype=np.float32)
            jammer_channels_current_multi_hot[np.asarray(jammer_channels_current, dtype=np.int32)] = 1.0
        reward = self.act()
        state_next = self.get_state()  # 得到新的状态
        self.renew_jammer_channels_after_learn()
        self.episode_step += 1
        done = self.episode_step >= self.max_episode_steps
        if return_info:
            jammer_channels_next = [int(ch) for ch in list(self.jammer_channels)]
            jammer_channels_next_multi_hot = np.zeros([self.n_channel], dtype=np.float32)
            jammer_channels_next_multi_hot[np.asarray(jammer_channels_next, dtype=np.int32)] = 1.0
            info = {
                "episode_step": int(self.episode_step),
                "max_episode_steps": int(self.max_episode_steps),
                "jammer_channels_current": jammer_channels_current,
                "jammer_channels_current_multi_hot": jammer_channels_current_multi_hot.tolist(),
                "jammer_channels_next": jammer_channels_next,
                "jammer_channels_next_multi_hot": jammer_channels_next_multi_hot.tolist(),
                "jammer_obs_history": [
                    sorted(int(ch) for ch in observed_set)
                    for observed_set in self._jammer_observed_channel_history
                ],
            }
            if done:
                info["terminal_reason"] = "time_limit"
                info["TimeLimit.truncated"] = True
        else:
            info = {}
        return state_next, reward, done, info

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
        import matplotlib.pyplot as plt

        y_data = self.smooth(cost_list, 19)
        x_data = np.arange(len(cost_list))
        np.savetxt('DRQN_po.txt', y_data[0], fmt='%f')
        np.save('DRQN.npy',  y_data[0])

        plt.plot(x_data, y_data[0])
        plt.ylabel('DRQN__reward')
        plt.xlabel('training Episode')
        plt.show()

        plt.plot(x_data, cost_list)
        plt.ylabel('reward')
        plt.xlabel('training Episode')
        plt.show()
