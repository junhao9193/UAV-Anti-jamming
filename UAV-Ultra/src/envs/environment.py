"""``Environ`` 编排器：组装 8 个子模块 + 4 个私有 RNG + last_link_metrics 公开字段。

设计原则：

- 本类**只做编排**，不持业务逻辑：``reset`` / ``step`` 顺序调用子模块函数 / 策略。
- 4 个私有 RNG（plan locked decision #8）：
  - ``self._rng``：通用（位置、感知噪声、CSI 噪声等），从 ``env_seed`` 第 0 个 spawn 派生。
  - ``self._fast_fading_rng``：快衰落 AR(1)，从 ``env_seed`` 第 1 个 spawn 或显式
    ``fast_fading_seed`` 派生。
  - ``self._jammer_state_rng``（Python ``random.Random``）：jammer 状态采样 + 部分观察。
  - ``self._p_trans_rng``（**Stage 3 新增**）：``generate_p_trans`` 使用，独立于
    jammer_state_rng，消除 baseline 跨进程漂移。
- ``self.last_link_metrics`` 是公开调试字段（plan §A）：``{"delivery", "success_flags",
  "transmit_times"}`` 三个 ``(n_ch, n_des)`` numpy 数组，``test_env_contract.py`` 直接读取。
- step() 调用链严格按 plan §A 顺序：base reward 在 mobility 前算，``apply_mobility_penalty``
  在 mobility 后调用；默认权重 0 → baseline 等价。
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import perm

from src.config.loader import load_env_config
from src.envs import channel as channel_module
from src.envs import jammer_model
from src.envs import mobility as mobility_module
from src.envs import observation as observation_module
from src.envs import reward as reward_module
from src.envs.action_space import decompose as decompose_action


class Environ:
    """Stage 3 拆分后的多智能体环境。

    API 与 baseline 兼容：``reset(p_trans=None) -> state``、``step(actions) ->
    (state, reward, done, info)``。``state`` 为 ``list[np.ndarray]``，长度 ``n_ch``。
    """

    # ----------------------- 初始化 -----------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Any] = None) -> None:
        cfg = load_env_config(yaml_path=config_path, overrides=config)
        self._cfg = cfg

        # ----- RNG 初始化（4 个流） -----
        self.env_seed = cfg.env_seed
        seed_sequence = np.random.SeedSequence(None if self.env_seed is None else int(self.env_seed))
        # spawn(4) 的前 3 个 child 与 baseline spawn(3) 完全相同（spawn_key 仅依赖索引），
        # 所以新增 p_trans_rng 不破坏 baseline 等价。
        env_seed_seq, fast_fading_seed_seq, jammer_seed_seq, p_trans_seed_seq = seed_sequence.spawn(4)
        self._rng = np.random.default_rng(env_seed_seq)

        def _seed_int_from_sequence(seq) -> int:
            return int(seq.generate_state(1, dtype=np.uint32)[0])

        # ----- 场景几何 -----
        self.length = cfg.length
        self.width = cfg.width
        self.low_height = cfg.low_height
        self.high_height = cfg.high_height
        # baseline 用 cfg.get("BS_position", [center])；UAV-Ultra schema 不含此字段，
        # 直接用 baseline 默认（取中心）。pathloss 公式不依赖 BS_position，仅作签名占位。
        self.BS_position = [
            self.length / 2,
            self.width / 2,
            (self.low_height + self.high_height) / 2,
        ]

        # ----- 运动模型参数 -----
        self.k = float(cfg.k)
        if not (0.0 <= self.k <= 1.0):
            raise ValueError(f"k must be in [0, 1] for the Gauss-Markov mobility model, got {self.k}")
        self.sigma = float(cfg.sigma)
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative for the Gauss-Markov mobility model, got {self.sigma}")

        # ----- 功率 / 天线 -----
        self.uav_power_min = float(cfg.uav_power_min)
        self.uav_power_max = float(cfg.uav_power_max)
        self.jammer_power = cfg.jammer_power
        self.sig2_dB = cfg.sig2_dB
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.uavAntGain = cfg.uavAntGain
        self.uavNoiseFigure = cfg.uavNoiseFigure
        self.jammerAntGain = cfg.jammerAntGain
        self.bandwidth = cfg.bandwidth
        self.uav_interference_scale = float(cfg.uav_interference_scale)

        # ----- 时间预算 -----
        self.data_size = cfg.data_size
        self.t_Rx = cfg.t_Rx
        self.t_collect = cfg.t_collect
        self.timestep = cfg.timestep
        self.timeslot = self.t_Rx + self.timestep
        self.max_episode_steps = int(cfg.max_episode_steps)
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        self.t_uav = 0.0
        self.jammer_start = cfg.jammer_start
        self.t_dwell = cfg.t_dwell
        self.t_jammer = 0.0

        # ----- 拓扑规模 -----
        self.n_ch = cfg.n_ch
        self.n_cm_for_a_ch = cfg.n_cm_for_a_ch
        self.n_cm = self.n_ch * self.n_cm_for_a_ch
        self.n_uav = self.n_ch + self.n_cm
        self.n_rp_ch = self.n_ch
        self.n_rp_cm = self.n_cm
        self.n_rp = self.n_uav
        self.n_des = self.n_cm_for_a_ch
        self.n_uav_pair = self.n_ch * self.n_des
        self.n_jammer = cfg.n_jammer
        self.n_channel = cfg.n_channel
        self.channel_indexes = np.arange(self.n_channel)
        self.states_observed = cfg.states_observed

        # ----- 感知 -----
        self.p_md = cfg.p_md
        self.p_fa = cfg.p_fa
        self.pn0 = cfg.pn0
        self.sensing_w_jammer = float(cfg.sensing_w_jammer)
        self.sensing_w_uav = float(cfg.sensing_w_uav)
        self.sensing_noise_std = float(cfg.sensing_noise_std)
        self.sensing_jammer_range = float(cfg.sensing_jammer_range)
        self.sensing_uav_range = float(cfg.sensing_uav_range)

        # ----- 信道损耗 + 频率选择性 -----
        self.channel_loss_db = np.asarray(cfg.channel_loss_db, dtype=np.float32).reshape(-1)
        if self.channel_loss_db.size != int(self.n_channel):
            raise ValueError(
                f"channel_loss_db length must equal n_channel ({self.n_channel}), got {self.channel_loss_db.size}"
            )

        self.channel_selectivity_std_db = float(cfg.channel_selectivity_std_db)
        self.channel_selectivity_seed = int(cfg.channel_selectivity_seed)
        if self.channel_selectivity_std_db <= 0.0:
            self.uav_channel_selectivity_db = np.zeros(
                (int(self.n_uav), int(self.n_uav), int(self.n_channel)), dtype=np.float32
            )
            self.jammer_channel_selectivity_db = np.zeros(
                (int(self.n_jammer), int(self.n_uav), int(self.n_channel)), dtype=np.float32
            )
        else:
            sel_rng = np.random.default_rng(self.channel_selectivity_seed)
            self.uav_channel_selectivity_db = sel_rng.normal(
                0.0, self.channel_selectivity_std_db,
                size=(int(self.n_uav), int(self.n_uav), int(self.n_channel)),
            ).astype(np.float32)
            self.jammer_channel_selectivity_db = sel_rng.normal(
                0.0, self.channel_selectivity_std_db,
                size=(int(self.n_jammer), int(self.n_uav), int(self.n_channel)),
            ).astype(np.float32)

        # ----- 快衰落 -----
        self.enable_fast_fading = bool(cfg.enable_fast_fading)
        self.fast_fading_rho = float(cfg.fast_fading_rho)
        if not (0.0 <= self.fast_fading_rho < 1.0):
            raise ValueError(f"fast_fading_rho must be in [0,1), got {self.fast_fading_rho}")
        self.fast_fading_eps = float(cfg.fast_fading_eps)
        self.fast_fading_db_clip_low = cfg.fast_fading_db_clip_low
        self.fast_fading_db_clip_high = cfg.fast_fading_db_clip_high
        if self.fast_fading_db_clip_low is not None:
            self.fast_fading_db_clip_low = float(self.fast_fading_db_clip_low)
        if self.fast_fading_db_clip_high is not None:
            self.fast_fading_db_clip_high = float(self.fast_fading_db_clip_high)

        if cfg.fast_fading_seed is None:
            self._fast_fading_rng = np.random.default_rng(fast_fading_seed_seq)
        else:
            self._fast_fading_rng = np.random.default_rng(int(cfg.fast_fading_seed))

        # Rayleigh 均值（baseline 行 189）
        self._rayleigh_mean_db = float(-10.0 * np.euler_gamma / np.log(10.0))
        self._uav_fast_h = None
        self._jammer_fast_h = None

        self.max_distance1 = cfg.max_distance1
        self.max_distance2 = cfg.max_distance2

        # ----- Jammer 参数 -----
        # is_jammer_moving 在 loader 中已强制 True；保留字段供向后兼容
        self.is_jammer_moving = cfg.is_jammer_moving
        self.p_trans_seed = int(cfg.p_trans_seed)
        self.p_trans_preferred_next_states = int(cfg.p_trans_preferred_next_states)
        self.p_trans_preference_strength = float(cfg.p_trans_preference_strength)
        self.jammer_reactive_beta = float(cfg.jammer_reactive_beta)
        self.jammer_memory_window = int(cfg.jammer_memory_window)
        if self.jammer_memory_window < 1:
            raise ValueError(f"jammer_memory_window must be >= 1, got {self.jammer_memory_window}")
        self._jammer_observed_channel_history = deque(maxlen=self.jammer_memory_window)
        self.jammer_reactive_observe_prob = float(cfg.jammer_reactive_observe_prob)
        if not (0.0 <= self.jammer_reactive_observe_prob <= 1.0):
            raise ValueError(
                f"jammer_reactive_observe_prob must be in [0,1], got {self.jammer_reactive_observe_prob}"
            )

        if cfg.jammer_seed is None:
            jammer_seed = _seed_int_from_sequence(jammer_seed_seq)
        else:
            jammer_seed = int(cfg.jammer_seed)
        self._jammer_state_rng = random.Random(int(jammer_seed))

        # p_trans_rng：独立流，由 cfg.p_trans_seed 派生（plan locked decision #4/#8）
        self._p_trans_rng = np.random.default_rng(self.p_trans_seed)

        # ----- 奖励参数 -----
        self.reward_energy_weight = cfg.reward_energy_weight
        self.reward_jump_weight = cfg.reward_jump_weight
        self.fairness_min_success_rate = float(cfg.fairness_min_success_rate)
        self.fairness_weight = float(cfg.fairness_weight)
        self.csi_pathloss_offset = float(cfg.csi_pathloss_offset)
        self.csi_pathloss_scale = float(cfg.csi_pathloss_scale)
        self.csi_noise_std = float(cfg.csi_noise_std)
        if self.csi_noise_std < 0.0:
            raise ValueError(f"csi_noise_std must be non-negative, got {self.csi_noise_std}")
        self.csi_clip = bool(cfg.csi_clip)

        # ----- Stage 3.5 mobility 配置（供 action_space / mobility / reward 读取） -----
        self.uav_mobility_control = cfg.uav_mobility_control
        self.jammer_mobility_model = cfg.jammer_mobility_model
        self.uav_velocity_delta_max = float(cfg.uav_velocity_delta_max)
        self.uav_direction_delta_max = float(cfg.uav_direction_delta_max)
        self.uav_p_delta_max = float(cfg.uav_p_delta_max)
        self.jammer_guidance_strength = float(cfg.jammer_guidance_strength)
        self.observation_include_mobility = bool(cfg.observation_include_mobility)
        self.mobility_oob_penalty_weight = float(cfg.mobility_oob_penalty_weight)
        self.mobility_energy_weight = float(cfg.mobility_energy_weight)
        # 由 action_space.decompose 填入；GaussMarkov 模式下保持 None
        self._last_mobility_deltas = None
        self.last_mobility_delta_sq = None
        self.last_mobility_oob = None
        self.rew_mobility_delta_sq = 0.0
        self.rew_mobility_oob = 0.0
        self.rew_mobility_penalty = 0.0

        # 策略派发
        self._uav_mobility_strategy = mobility_module.build_uav_mobility_strategy(cfg)
        self._jammer_mobility_strategy = mobility_module.build_jammer_mobility_strategy(cfg)

        # ----- 状态字段初始化 -----
        self.policy = None  # 兼容 baseline；UAV-Ultra 只走默认路径
        self.jammer_channels = [0 for _ in range(self.n_jammer)]
        self.jammer_events: list = []

        self.uav_list = list(range(int(self.n_uav)))
        self.ch_list = self._sample_without_replacement(self.uav_list, self.n_ch)
        ch_set = set(self.ch_list)
        self.cm_list = [uav_id for uav_id in self.uav_list if uav_id not in ch_set]
        self.rp_list = self.uav_list
        self.rp_ch_list = self.ch_list
        self.rp_cm_list = self.cm_list
        self.uav_pairs = np.zeros([self.n_ch, self.n_des, 2], dtype=np.int32)
        self.uav_clusters = np.zeros([self.n_ch, self.n_cm_for_a_ch, 2], dtype=np.int32)

        # 动作空间维度（baseline 沿用）
        self.action_dim = int(self.n_channel ** self.n_des)
        self.param_dim_per_action = int(self.n_des)
        self.total_param_dim = int(self.action_dim * self.param_dim_per_action)

        # 奖励累积
        self.uav_jump_count = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        self.rew_energy = 0
        self.rew_jump = 0
        self.rew_suc = 0

        # 回合管理
        self.episode_step = 0
        self._episode_initialized = False

        # 公开 link metrics（替代 Stage 0 generator 的 monkey-patch）
        self.last_link_metrics: dict = {}

        # 初始化静态状态空间 + 默认 p_trans
        jammer_model.all_observed_states(self)
        self.set_p(self.generate_p_trans())
        self.state_dim = (self.n_des + 1) * self.n_channel + (
            6 if self.observation_include_mobility else 0
        )

    # ----------------------- 工具 -----------------------

    def _sample_without_replacement(self, population, k):
        population = list(population)
        if int(k) > len(population):
            raise ValueError(f"Cannot sample {k} items from population of size {len(population)}")
        indices = self._rng.choice(len(population), size=int(k), replace=False)
        return [population[int(idx)] for idx in np.asarray(indices).reshape(-1)]

    # ----------------------- jammer p_trans -----------------------

    def generate_p_trans(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """生成 Markov 转移矩阵。默认使用 ``self._p_trans_rng``，可显式覆盖。"""
        use_rng = rng if rng is not None else self._p_trans_rng
        return jammer_model.generate_p_trans(
            self.jammer_state_dim,
            rng=use_rng,
            preferred_next_states=self.p_trans_preferred_next_states,
            preference_strength=self.p_trans_preference_strength,
        )

    def set_p(self, p_trans) -> None:
        p_arr = np.asarray(p_trans, dtype=np.float32)
        if p_arr.ndim != 2:
            raise ValueError(f"Invalid p_trans ndim={p_arr.ndim}. Expected a single 2-D Markov matrix.")
        if p_arr.shape != (int(self.jammer_state_dim), int(self.jammer_state_dim)):
            raise ValueError(
                "Invalid p_trans shape. Expected (D, D) with D=jammer_state_dim, got "
                f"{p_arr.shape} (D={self.jammer_state_dim})"
            )
        self.p_trans = p_arr

    # ----------------------- reset / step -----------------------

    def _new_random_game(self) -> None:
        self.t_uav = 0.0
        self.t_jammer = 0.0
        self.episode_step = 0
        self._episode_initialized = True
        self._jammer_observed_channel_history.clear()

        self.uav_channels = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        self.uav_powers = np.zeros([self.n_ch, self.n_des], dtype=np.float32)
        self.uav_jump_count = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        for i in range(self.n_ch):
            for j in range(self.n_des):
                self.uav_channels[i][j] = int(self._rng.integers(0, self.n_channel))
                self.uav_powers[i][j] = float(
                    self._rng.uniform(self.uav_power_min, self.uav_power_max)
                )
        jammer_model.init_jammer_state(self)

        self.uavs = [None] * self.n_uav
        self.rps = [None] * self.n_rp
        self.jammers = []
        mobility_module.init_uavs(self)
        mobility_module.init_uav_clusters(self)
        mobility_module.init_jammers(self)

        self.UAVchannels = channel_module.UAVchannels(self.n_uav, self.n_channel, self.BS_position)
        self.Jammerchannels = channel_module.Jammerchannels(
            self.n_jammer, self.n_uav, self.n_channel, self.BS_position
        )
        # 重置快衰落 h 状态，下一次 renew_channels 重新初始化 h_0
        self._uav_fast_h = None
        self._jammer_fast_h = None
        channel_module.renew_channels(self)

    def reset(self, p_trans=None):
        if p_trans is not None:
            self.set_p(p_trans)
        self._new_random_game()
        return observation_module.get_state(self)

    def step(self, a, return_info: bool = True):
        if not self._episode_initialized:
            raise RuntimeError("Environment episode state is not initialized. Call reset() before step().")

        # 1. action 分解（gauss_markov 或 policy 模式）
        decompose_action(self, a)

        # 2. 快照 jammer_channels_current（在 jammer 转移之前）
        if return_info:
            jammer_channels_current = [int(ch) for ch in list(self.jammer_channels)]
            jammer_channels_current_multi_hot = np.zeros([self.n_channel], dtype=np.float32)
            jammer_channels_current_multi_hot[
                np.asarray(jammer_channels_current, dtype=np.int32)
            ] = 1.0

        # 3-4. jammer 观察记录 + Rx 后转移
        jammer_model.record_jammer_observation(self)
        jammer_model.renew_jammer_channels_after_Rx(self)

        # 5. base reward + 写 last_link_metrics
        reward = reward_module.compute_step_reward(self)

        # 6. mobility 更新（CH 策略 / CM 跟随 / Jammer 策略）
        self._uav_mobility_strategy.update_ch_positions(self)
        mobility_module.update_cm_positions(self)
        self._jammer_mobility_strategy.update_jammer_positions(self)

        # 6.5 记录 mobility 调试量（供 reward.apply_mobility_penalty 使用；默认权重 0 时不影响）
        if self.last_mobility_delta_sq is None:
            self.last_mobility_delta_sq = np.zeros((self.n_ch,), dtype=np.float64)
        else:
            self.last_mobility_delta_sq.fill(0.0)
        if self.last_mobility_oob is None:
            self.last_mobility_oob = np.zeros((self.n_ch,), dtype=np.float64)
        else:
            self.last_mobility_oob.fill(0.0)
        for ch in range(self.n_ch):
            dx, dy, dz = self.xyz_delta_dis[ch]
            self.last_mobility_delta_sq[ch] = float(dx * dx + dy * dy + dz * dz)
        oob_values = getattr(self, "xyz_oob_dis", [0.0 for _ in range(self.n_ch)])
        for ch in range(self.n_ch):
            self.last_mobility_oob[ch] = float(oob_values[ch])

        # 7. 信道刷新（路损 + 快衰落）
        channel_module.renew_channels(self)

        # 8. 移动惩罚（默认权重 0 = 等价 baseline）
        reward = reward_module.apply_mobility_penalty(self, reward)

        # 9. 新观测
        state_next = observation_module.get_state(self)

        # 10. learn 阶段 jammer 转移
        jammer_model.renew_jammer_channels_after_learn(self)

        # 11. 计步 + info
        self.episode_step += 1
        done = self.episode_step >= self.max_episode_steps
        if return_info:
            jammer_channels_next = [int(ch) for ch in list(self.jammer_channels)]
            jammer_channels_next_multi_hot = np.zeros([self.n_channel], dtype=np.float32)
            jammer_channels_next_multi_hot[
                np.asarray(jammer_channels_next, dtype=np.int32)
            ] = 1.0
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

    # ----------------------- 兼容 baseline 暴露的方法 -----------------------

    def get_state(self):
        return observation_module.get_state(self)

    def reward_details(self):
        return reward_module.reward_details(self)

    def mobility_reward_details(self):
        return reward_module.mobility_reward_details(self)

    def clear_reward(self) -> None:
        reward_module.clear_reward(self)


__all__ = ["Environ"]
