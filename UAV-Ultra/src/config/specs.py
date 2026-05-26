"""纯函数维度规格。

设计约束（详见 REFACTOR.md Constraint 2）：

- 本模块**只**可 import ``math.perm`` 与 ``src.config.schema.EnvConfig``。
- 禁止 import ``src.envs.*`` / ``src.algorithms.*`` / ``torch`` —— 维度信息
  属于「配置派生出的接口契约」，``algorithms/`` 必须可以独立拿到。
- 公式与 baseline ``2360ab9`` 的 ``UAV-Jammer-RL/envs/core.py`` 1:1 对应：

  ====================== ===================================== ==================
  字段                     公式                                   旧代码行号
  ====================== ===================================== ==================
  n_des                  cfg.n_cm_for_a_ch                     envs/core.py:102
  n_uav_pair             cfg.n_ch * n_des(cfg)                 envs/core.py:103
  state_dim              (n_des(cfg) + 1) * cfg.n_channel      envs/core.py:341-346
  action_dim             cfg.n_channel ** n_des(cfg)           envs/core.py:256
  param_dim_per_action   n_des(cfg)                            envs/core.py:257
  total_param_dim        action_dim(cfg)*param_dim_per_action  envs/core.py:258
  jammer_state_dim       perm(cfg.n_channel, cfg.n_jammer)     envs/core.py:325
  ====================== ===================================== ==================

观测向量布局的唯一真相源是 ``envs/observation.py``（Stage 3 落地）；
本模块通过测试 ``len(env.get_state()) == specs.state_dim(cfg)`` 与之强制同步。
"""

from __future__ import annotations

from math import perm

from src.config.schema import EnvConfig


def n_des(cfg: EnvConfig) -> int:
    """每个簇头服务的目的节点数。直接来自 YAML 字段，不是派生量。"""
    return int(cfg.n_cm_for_a_ch)


def n_uav_pair(cfg: EnvConfig) -> int:
    """总通信链路数 = n_ch × n_des。"""
    return int(cfg.n_ch) * n_des(cfg)


def state_dim(cfg: EnvConfig) -> int:
    """状态向量最后一维。

    基线项：CSI 部分 ``n_des * n_channel`` + 频谱感知部分 ``1 * n_channel``。
    Stage 3.5 扩展：``observation_include_mobility=True`` 时再加
    ``mobility_obs_dim_per_ch(cfg) = 6``（CH 自身的归一化位置 / 速度 / 方向 / 仰角）。
    默认配置下 = ``(n_des + 1) * n_channel``（与 baseline 一致）。
    """
    return (n_des(cfg) + 1) * int(cfg.n_channel) + mobility_obs_dim_per_ch(cfg)


def action_dim(cfg: EnvConfig) -> int:
    """离散动作空间大小：n_channel 进制下表示 n_des 个目的的信道分配。"""
    return int(cfg.n_channel) ** n_des(cfg)


def param_dim_per_action(cfg: EnvConfig) -> int:
    """每个离散动作槽的连续功率参数数量 = n_des。"""
    return n_des(cfg)


def total_param_dim(cfg: EnvConfig) -> int:
    """全部连续参数维度 = action_dim × param_dim_per_action（仅信道分配部分）。"""
    return action_dim(cfg) * param_dim_per_action(cfg)


def jammer_state_dim(cfg: EnvConfig) -> int:
    """干扰机联合状态空间大小 = P(n_channel, n_jammer)（有序选择）。"""
    return int(perm(int(cfg.n_channel), int(cfg.n_jammer)))


def mobility_action_dim_per_ch(cfg: EnvConfig) -> int:
    """Stage 3.5: policy 模式下每个 CH 额外的连续 mobility delta 维数。

    - ``uav_mobility_control == "gauss_markov"`` → 0（baseline 等价）
    - ``uav_mobility_control == "policy"`` → 3（velocity_delta / direction_delta / p_delta）
    """
    return 3 if cfg.uav_mobility_control == "policy" else 0


def mobility_obs_dim_per_ch(cfg: EnvConfig) -> int:
    """Stage 3.5: ``observation_include_mobility`` 开启时每个 CH 追加的观测维数。

    6 维 = (position_xyz_normalized: 3) + (velocity_norm: 1) + (direction_norm: 1)
       + (p_norm: 1)；默认 0，与 baseline 等价。
    """
    return 6 if cfg.observation_include_mobility else 0


def per_ch_param_dim(cfg: EnvConfig) -> int:
    """单个 CH 的连续参数总维度 = total_param_dim + mobility_action_dim_per_ch。"""
    return total_param_dim(cfg) + mobility_action_dim_per_ch(cfg)


__all__ = [
    "n_des",
    "n_uav_pair",
    "state_dim",
    "action_dim",
    "param_dim_per_action",
    "total_param_dim",
    "jammer_state_dim",
    "mobility_action_dim_per_ch",
    "mobility_obs_dim_per_ch",
    "per_ch_param_dim",
]
