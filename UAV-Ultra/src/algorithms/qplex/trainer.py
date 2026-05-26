"""QPLEX trainer：``ValueDecompTrainerBase`` + 3 个 hook 注入 max_agent_qs。

与 baseline ``algorithms/mpdqn/qplex/trainer.py`` 等价：
- ``_collect_critic_extras``：actor 在 no_grad 内产 greedy params；q_net forward **不** no_grad
  且 **不** detach（梯度回流 q_net；baseline qplex/trainer.py:158-172）。
- ``_collect_target_extras``：target_actor + target_q_net forward（整个分支已在 base 的
  ``torch.no_grad()`` 内部；baseline qplex/trainer.py:176-192）。
- ``_collect_actor_extras``：用 base 提供的 ``q_pred_all_list``，``max(dim=1, keepdim=True).values.detach()``
  形成 advantage baseline（baseline qplex/trainer.py:243-247 强调必须 detach）。

形状约束：所有 ``max_agent_qs`` 必须是 ``(B, N)``，per-agent ``(B, A) → max(dim=1, keepdim=True)``
后 cat dim=1。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.algorithms.common.networks.mixers import QPLEXMixer
from src.algorithms.common.value_decomp import ValueDecompTrainerBase
from src.config import specs
from src.config.schema import EnvConfig, QPLEXConfig


class QPLEXTrainer(ValueDecompTrainerBase):
    """QPLEX duplex dueling mixer + MP-DQN agents + joint replay。"""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: QPLEXConfig,
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

        self._mixing_hidden_dim = int(algo_cfg.mixing_hidden_dim)
        self._hypernet_hidden_dim = int(algo_cfg.hypernet_hidden_dim)
        self._n_heads = int(algo_cfg.n_heads)

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
            lr_mixer=float(algo_cfg.lr_mixer),
            target_update_interval=int(algo_cfg.target_update_interval),
            use_amp=bool(algo_cfg.use_amp),
            max_grad_norm=float(algo_cfg.max_grad_norm),
            value_target_clip=float(algo_cfg.value_target_clip),
            device=resolved_device,
        )

    def _build_mixer(self) -> nn.Module:
        return QPLEXMixer(
            n_agents=self.n_agents,
            global_state_dim=self.global_state_dim,
            mixing_hidden_dim=self._mixing_hidden_dim,
            hypernet_hidden_dim=self._hypernet_hidden_dim,
            n_heads=self._n_heads,
        )

    # ----------------------- Mixer forward 覆盖（消费 max_agent_qs） -----------------------

    def _mix(self, agent_qs: torch.Tensor, global_state: torch.Tensor, **extras) -> torch.Tensor:
        max_agent_qs = extras.get("max_agent_qs")
        if max_agent_qs is None:
            raise RuntimeError("QPLEXTrainer._mix requires 'max_agent_qs' in extras")
        return self.mixer(agent_qs, max_agent_qs, global_state)

    def _target_mix(
        self, next_agent_qs: torch.Tensor, next_global_state: torch.Tensor, **extras
    ) -> torch.Tensor:
        max_agent_qs = extras.get("max_agent_qs")
        if max_agent_qs is None:
            raise RuntimeError("QPLEXTrainer._target_mix requires 'max_agent_qs' in extras")
        return self.target_mixer(next_agent_qs, max_agent_qs, next_global_state)

    # ----------------------- Hooks -----------------------

    def _collect_critic_extras(
        self, state: torch.Tensor, action_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Per-agent ``max_agent_qs``：actor no_grad 产 greedy params；q_net 不 no_grad、不 detach。

        baseline `qplex/trainer.py:158-172`。
        """
        q_max_list: List[torch.Tensor] = []
        for i in range(self.n_agents):
            s_i = state[:, i, :]
            with torch.no_grad():
                greedy_params_i = self.agents[i].actor(s_i)
            q_greedy_all_i = self.agents[i].q_net(s_i, greedy_params_i)
            q_max_i = q_greedy_all_i.max(dim=1, keepdim=True).values  # (B, 1)
            q_max_list.append(q_max_i)
        return {"max_agent_qs": torch.cat(q_max_list, dim=1)}  # (B, N)

    def _collect_target_extras(self, next_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-agent target ``max_agent_qs``：target_actor + target_q_net forward。

        base 在 ``torch.no_grad()`` 内调用本 hook（``_critic_step`` 的 target 分支），
        自然 detach。baseline `qplex/trainer.py:176-191`。
        """
        q_max_list: List[torch.Tensor] = []
        for i in range(self.n_agents):
            ns_i = next_state[:, i, :]
            next_params_target = self.agents[i].target_actor(ns_i)
            next_q_target_all = self.agents[i].target_q_net(ns_i, next_params_target)
            next_q_target_max = next_q_target_all.max(dim=1, keepdim=True).values  # (B, 1)
            q_max_list.append(next_q_target_max)
        return {"max_agent_qs": torch.cat(q_max_list, dim=1)}  # (B, N)

    def _collect_actor_extras(
        self,
        state: torch.Tensor,
        params_pred_list: List[torch.Tensor],
        q_pred_all_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-agent ``max_agent_qs`` = q_pred_all.max(...).detach()。

        必须 detach 形成 advantage baseline（baseline `qplex/trainer.py:243-247`）。
        """
        q_max_list: List[torch.Tensor] = [
            q_pred.max(dim=1, keepdim=True).values.detach()  # (B, 1)
            for q_pred in q_pred_all_list
        ]
        return {"max_agent_qs": torch.cat(q_max_list, dim=1)}  # (B, N)


__all__ = ["QPLEXTrainer"]
