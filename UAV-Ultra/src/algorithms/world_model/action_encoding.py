"""联合动作执行期特征编码（baseline ``world_model/action_encoding.py`` 1:1 端口）。

Per-agent feature: ``n_des * onehot(n_channel) + param_dim``。
Joint feature: ``n_agents * per_agent_dim``。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def exec_action_dim(*, n_agents: int, n_des: int, n_channel: int, param_dim: int) -> int:
    """Execution-time action feature dimension。"""
    per_agent = int(n_des) * int(n_channel) + int(param_dim)
    return int(n_agents) * per_agent


def _decode_channels_base_n(
    discrete_action: torch.Tensor, *, n_des: int, n_channel: int
) -> torch.Tensor:
    if discrete_action.ndim != 1:
        raise ValueError(f"discrete_action must be (B,), got {tuple(discrete_action.shape)}")
    decoded = discrete_action
    channels = []
    for _ in range(int(n_des)):
        channels.append(decoded.remainder(int(n_channel)))
        decoded = torch.div(decoded, int(n_channel), rounding_mode="floor")
    return torch.stack(channels, dim=1).to(torch.long)


def _gather_power_params_for_chosen_action(
    action_discrete: torch.Tensor,
    action_params_flat: torch.Tensor,
    *,
    n_actions: int,
    param_dim: int,
) -> torch.Tensor:
    if action_params_flat.ndim != 2:
        raise ValueError(
            f"action_params_flat must be (B, n_actions*param_dim), got {tuple(action_params_flat.shape)}"
        )
    bsz = int(action_params_flat.shape[0])
    params_all = action_params_flat.view(bsz, int(n_actions), int(param_dim))
    idx = action_discrete.view(bsz, 1, 1).expand(bsz, 1, int(param_dim))
    chosen = torch.gather(params_all, dim=1, index=idx).squeeze(1)
    return chosen


def encode_agent_action_exec(
    action_discrete: torch.Tensor,
    action_params_flat: torch.Tensor,
    *,
    n_channel: int,
    n_des: int,
    n_actions: int,
    param_dim: int,
    power_min_dbm: float | None = None,
    power_max_dbm: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单 agent execution feature：channel one-hot + power scalars。"""
    action_discrete = action_discrete.to(torch.long).view(-1)
    action_params_flat = action_params_flat.to(torch.float32).view(action_discrete.shape[0], -1)

    channels_idx = _decode_channels_base_n(action_discrete, n_des=int(n_des), n_channel=int(n_channel))
    channels_oh = F.one_hot(channels_idx, num_classes=int(n_channel)).to(torch.float32)
    channels_oh = channels_oh.view(action_discrete.shape[0], -1)

    power_params = _gather_power_params_for_chosen_action(
        action_discrete,
        action_params_flat,
        n_actions=int(n_actions),
        param_dim=int(param_dim),
    ).to(torch.float32)

    # 物理一致性：把归一化 power 映射到 watts 再除以 max watts，提升 reward/transition 学习
    if power_min_dbm is not None and power_max_dbm is not None:
        pmin = float(power_min_dbm)
        pmax = float(power_max_dbm)
        p_dbm = pmin + power_params * (pmax - pmin)
        p_watts = torch.pow(10.0, p_dbm / 10.0 - 3.0)
        p_watts_max = float(10.0 ** (pmax / 10.0 - 3.0))
        power_params = p_watts / (p_watts_max + 1e-12)
        power_params = torch.clamp(power_params, 0.0, 1.0)

    feat = torch.cat([channels_oh, power_params], dim=1)
    return feat, channels_idx, power_params


def encode_joint_action_exec(
    action_discrete: torch.Tensor,
    action_params_flat: torch.Tensor,
    *,
    n_agents: int,
    n_channel: int,
    n_des: int,
    n_actions: int,
    param_dim: int,
    power_min_dbm: float | None = None,
    power_max_dbm: float | None = None,
) -> torch.Tensor:
    """所有 agent 串联：``(B, N*(n_des*n_channel + param_dim))``。"""
    if action_discrete.ndim != 2:
        raise ValueError(f"action_discrete must be (B,N), got {tuple(action_discrete.shape)}")
    if action_params_flat.ndim != 3:
        raise ValueError(f"action_params_flat must be (B,N,AP), got {tuple(action_params_flat.shape)}")
    if int(action_discrete.shape[1]) != int(n_agents):
        raise ValueError(f"Expected N={int(n_agents)} agents, got {int(action_discrete.shape[1])}")

    feats = []
    for i in range(int(n_agents)):
        feat_i, _, _ = encode_agent_action_exec(
            action_discrete[:, i],
            action_params_flat[:, i, :],
            n_channel=int(n_channel),
            n_des=int(n_des),
            n_actions=int(n_actions),
            param_dim=int(param_dim),
            power_min_dbm=power_min_dbm,
            power_max_dbm=power_max_dbm,
        )
        feats.append(feat_i)
    return torch.cat(feats, dim=1)


__all__ = ["exec_action_dim", "encode_agent_action_exec", "encode_joint_action_exec"]
