"""类型化配置 dataclass。

设计约定：
- YAML 是默认值的唯一真相源，dataclass **不**承担业务默认值。
- 项目内只通过 ``loader.load_*`` 构造配置对象；dataclass 技术上仍可直接构造，
  但不允许走这条路径，避免 Python 默认值与 YAML 漂移成第二真相源。
- ``@dataclass(frozen=True)`` 防止运行时被改写；类型校验由 loader 在实例化
  **之前**对 merged dict 完成，dataclass 本身不会拒绝 ``"1.0"`` 这种字符串。

所有字段与 baseline 提交 ``2360ab92`` 的 ``UAV-Jammer-RL/configs/env.yaml`` 一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class EnvConfig:
    """与 baseline env.yaml 严格 1:1 的环境配置。

    字段总数 63，与 ``yaml.safe_load(baseline env.yaml)`` 的 key 集合相同。
    BS_position / uav_power_list 等旧 ``cfg.get()`` 兼容字段不在此处，等 Stage 3
    迁环境时再决定是否纳入 schema。
    """

    # ----- 场景几何 -----
    length: int
    width: int
    low_height: int
    high_height: int

    # ----- 全局 RNG 种子 -----
    # YAML 中允许为 null：null 时 env 内部从全局 SeedSequence 抽取或重置。
    env_seed: Optional[int]

    # ----- Gauss-Markov 运动模型 -----
    k: float
    sigma: float

    # ----- UAV / 干扰机功率与天线 -----
    uav_power_min: float
    uav_power_max: float
    jammer_power: int
    sig2_dB: int
    uavAntGain: int
    uavNoiseFigure: int
    jammerAntGain: int
    bandwidth: float
    uav_interference_scale: float

    # ----- 通信工作量与时间预算 -----
    data_size: float
    t_Rx: float
    t_collect: float
    timestep: float
    max_episode_steps: int
    jammer_start: float
    t_dwell: float

    # ----- 拓扑规模 -----
    n_ch: int
    n_cm_for_a_ch: int
    n_jammer: int
    n_channel: int
    states_observed: int

    # ----- 感知误差 -----
    p_md: int
    p_fa: int
    pn0: int

    # ----- 信道损耗与频率选择性 -----
    # channel_loss_db 长度必须 == n_channel，由 loader 显式断言。
    channel_loss_db: list[float]
    channel_selectivity_std_db: float
    channel_selectivity_seed: int

    # ----- 快衰落（AR(1)）-----
    enable_fast_fading: bool
    fast_fading_rho: float
    fast_fading_seed: Optional[int]
    fast_fading_eps: float
    # 用户可覆盖为 null 来关闭剪切；baseline YAML 中为浮点。
    fast_fading_db_clip_low: Optional[float]
    fast_fading_db_clip_high: Optional[float]

    # ----- 频谱感知能量图 -----
    sensing_w_jammer: float
    sensing_w_uav: float
    sensing_noise_std: float
    sensing_jammer_range: float
    sensing_uav_range: float

    # ----- 簇拓扑距离约束 -----
    max_distance1: int
    max_distance2: int

    # ----- 干扰机 Markov 行为 -----
    is_jammer_moving: bool
    p_trans_seed: int
    p_trans_preferred_next_states: int
    p_trans_preference_strength: float
    jammer_reactive_beta: float
    jammer_memory_window: int
    jammer_reactive_observe_prob: float
    jammer_seed: Optional[int]

    # ----- 奖励权重 -----
    reward_energy_weight: float
    reward_jump_weight: float
    fairness_min_success_rate: float
    fairness_weight: float

    # ----- CSI 观测 -----
    csi_pathloss_offset: float
    csi_pathloss_scale: float
    csi_noise_std: float
    csi_clip: bool

    # ----- Stage 3.5 移动控制扩展 -----
    # 默认值保持 baseline 行为：gauss_markov + 所有 delta_max/strength/penalty 权重为 0。
    # 详见 REFACTOR.md「四、后续构想」与 plan 文件。
    uav_mobility_control: str
    jammer_mobility_model: str
    uav_velocity_delta_max: float
    uav_direction_delta_max: float
    uav_p_delta_max: float
    jammer_guidance_strength: float
    observation_include_mobility: bool
    mobility_oob_penalty_weight: float
    mobility_energy_weight: float


@dataclass(frozen=True)
class TrainConfig:
    """训练循环共享配置。DQN 族（IQL/QMIX/VDN/QPLEX）通过 loader 自动叠加。"""

    n_episode: int
    # n_steps=None 表示从 EnvConfig.max_episode_steps 继承；用户可覆盖为整数显式截短。
    n_steps: Optional[int]
    num_envs: int
    batch_size: int
    buffer_capacity: int
    learn_every: int
    updates_per_learn: int
    seed: int
    device: str
    use_amp: bool
    start_method: str
    loss_log_every: int
    gamma: float
    target_update_interval: int
    lr_actor: float
    lr_q: float
    max_grad_norm: float


@dataclass(frozen=True)
class IQLConfig:
    """独立 Q 学习基线。直接组合 TrainConfig 字段 + epsilon 调度。"""

    # ----- 训练循环（来自 train/default.yaml）-----
    n_episode: int
    n_steps: Optional[int]
    num_envs: int
    batch_size: int
    buffer_capacity: int
    learn_every: int
    updates_per_learn: int
    seed: int
    device: str
    use_amp: bool
    start_method: str
    loss_log_every: int
    gamma: float
    target_update_interval: int
    lr_actor: float
    lr_q: float
    max_grad_norm: float

    # ----- IQL 专有 -----
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float


@dataclass(frozen=True)
class VDNConfig:
    """VDN：sum mixer 无可学习参数，因此**不**含 lr_mixer 字段。"""

    # ----- 训练循环 -----
    n_episode: int
    n_steps: Optional[int]
    num_envs: int
    batch_size: int
    buffer_capacity: int
    learn_every: int
    updates_per_learn: int
    seed: int
    device: str
    use_amp: bool
    start_method: str
    loss_log_every: int
    gamma: float
    target_update_interval: int
    lr_actor: float
    lr_q: float
    max_grad_norm: float

    # ----- VDN 专有 -----
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    value_target_clip: float


@dataclass(frozen=True)
class QMIXConfig:
    """QMIX：可学习 mixer + 超网络。lr_mixer=None 在 loader 中落定为 lr_q。"""

    # ----- 训练循环 -----
    n_episode: int
    n_steps: Optional[int]
    num_envs: int
    batch_size: int
    buffer_capacity: int
    learn_every: int
    updates_per_learn: int
    seed: int
    device: str
    use_amp: bool
    start_method: str
    loss_log_every: int
    gamma: float
    target_update_interval: int
    lr_actor: float
    lr_q: float
    max_grad_norm: float

    # ----- QMIX 专有 -----
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    # 注：YAML 中可写 null，loader 会替换为 lr_q；返回的实例字段不再为 None。
    lr_mixer: Optional[float]
    mixing_hidden_dim: int
    hypernet_hidden_dim: int
    value_target_clip: float
    callbacks: list[str]
    value_expansion_alpha_model: float
    value_expansion_seq_len: int
    value_expansion_td_lambda: float
    value_expansion_rollout_k: int
    # Stage 7：WM concurrent / block-alternating + L_VC
    wm_block_qmix_episodes: int
    wm_block_wm_episodes: int
    wm_batch_size: int
    wm_updates_per_learn: int
    wm_vc_eta_max: float
    wm_vc_warmup_ep: int
    wm_vc_ramp_end_ep: int
    wm_buffer_capacity: int
    wm_hidden_dim: int
    wm_n_layers: int
    wm_stochastic_dim: int
    wm_kl_beta: float
    wm_free_nats: float
    wm_lr: float
    wm_max_grad_norm: float
    # Stage 8：JP learning head
    jammer_history_len: int
    jammer_pred_hidden_dim: int
    jammer_aux_weight: float
    # lr_jammer: null → loader settle 为 lr_q
    lr_jammer: Optional[float]
    jammer_warmup_episodes: int
    use_jammer_feature: bool
    critic_stable_tau: float
    critic_stable_lr_scale: float


@dataclass(frozen=True)
class QPLEXConfig:
    """QPLEX：QMIX 结构 + 多头 attention。lr_mixer=None 在 loader 落定为 lr_q。"""

    # ----- 训练循环 -----
    n_episode: int
    n_steps: Optional[int]
    num_envs: int
    batch_size: int
    buffer_capacity: int
    learn_every: int
    updates_per_learn: int
    seed: int
    device: str
    use_amp: bool
    start_method: str
    loss_log_every: int
    gamma: float
    target_update_interval: int
    lr_actor: float
    lr_q: float
    max_grad_norm: float

    # ----- QPLEX 专有 -----
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    lr_mixer: Optional[float]
    mixing_hidden_dim: int
    hypernet_hidden_dim: int
    value_target_clip: float
    n_heads: int


@dataclass(frozen=True)
class MAPPOConfig:
    """MAPPO：PPO 与 DQN 族训练循环结构不同，本类**不**继承 TrainConfig。

    loader 在加载 mappo 时不合并 train/default.yaml，避免 PPO 拿到 DQN 的字段。
    """

    n_episode: int
    n_steps: Optional[int]
    seed: int
    device: str
    lr: float
    max_grad_norm: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    update_epochs: int
    minibatch_size: int


@dataclass(frozen=True)
class ExperimentPreset:
    """Packaged or file-backed baseline experiment preset.

    Presets are complete experiment overlays. ``env`` and ``algo`` stay as raw
    dictionaries so the normal loader/type validation remains the single gate.
    """

    algorithm: str
    description: str
    source: str
    env: dict
    algo: dict
    path: Path
    sha256: str


# 名字 -> 算法配置类的注册表。Stage 4/5 的 Runner 通过它根据 CLI 参数选类型。
AlgoConfig = Union[IQLConfig, VDNConfig, QMIXConfig, QPLEXConfig, MAPPOConfig]

ALGO_CONFIG_TYPES: dict[str, type] = {
    "iql": IQLConfig,
    "qmix": QMIXConfig,
    "vdn": VDNConfig,
    "qplex": QPLEXConfig,
    "mappo": MAPPOConfig,
}


__all__ = [
    "EnvConfig",
    "TrainConfig",
    "IQLConfig",
    "VDNConfig",
    "QMIXConfig",
    "QPLEXConfig",
    "MAPPOConfig",
    "ExperimentPreset",
    "AlgoConfig",
    "ALGO_CONFIG_TYPES",
]
