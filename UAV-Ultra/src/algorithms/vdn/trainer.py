"""VDN trainer：``ValueDecompTrainerBase`` 的 sum-mixer 具化。

与 baseline ``algorithms/mpdqn/vdn/trainer.py`` 等价：
- Mixer 无可学参（``VDNMixer`` 仅做 ``sum(qs, dim=1, keepdim=True)``）。
- ``_mixer_parameters()`` 返回 ``()`` → 基类不构造 mixer_opt。
- ``_collect_*_extras()`` 全部返回 ``{}``。
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from src.algorithms.common.networks.mixers import VDNMixer
from src.algorithms.common.value_decomp import ValueDecompTrainerBase
from src.config import specs
from src.config.schema import EnvConfig, VDNConfig


class VDNTrainer(ValueDecompTrainerBase):
    """VDN（无可学参 mixer）+ MP-DQN agents + joint replay。"""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: VDNConfig,
        device: Optional[str] = None,
    ):
        n_agents = int(env_cfg.n_ch)
        state_dim = int(specs.state_dim(env_cfg))
        n_actions = int(specs.action_dim(env_cfg))
        param_dim = int(specs.param_dim_per_action(env_cfg))
        global_state_dim = n_agents * state_dim

        if device is None:
            chosen = algo_cfg.device
            resolved_device = "cuda" if (chosen == "auto" and torch.cuda.is_available()) else (
                "cpu" if chosen == "auto" else chosen
            )
        else:
            resolved_device = device

        super().__init__(
            n_agents=n_agents,
            state_dim=state_dim,
            n_actions=n_actions,
            param_dim=param_dim,
            global_state_dim=global_state_dim,
            buffer_capacity=int(algo_cfg.buffer_capacity),
            batch_size=int(algo_cfg.batch_size),
            gamma=float(algo_cfg.gamma),
            lr_actor=float(algo_cfg.lr_actor),
            lr_q=float(algo_cfg.lr_q),
            lr_mixer=None,  # VDN: no mixer optimizer
            target_update_interval=int(algo_cfg.target_update_interval),
            use_amp=bool(algo_cfg.use_amp),
            max_grad_norm=float(algo_cfg.max_grad_norm),
            value_target_clip=float(algo_cfg.value_target_clip),
            device=resolved_device,
        )

    # ----------------------- Hooks -----------------------

    def _build_mixer(self) -> nn.Module:
        return VDNMixer()

    def _mixer_parameters(self) -> Iterable[torch.nn.Parameter]:
        return ()  # VDN sum mixer 无可学参


__all__ = ["VDNTrainer"]
