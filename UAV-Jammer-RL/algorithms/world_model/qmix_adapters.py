from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from algorithms.mpdqn.qmix.trainer import MPDQNQMIXTrainer


@dataclass(frozen=True)
class MPDQNQMIXDims:
    n_agents: int
    agent_state_dim: int
    n_actions: int
    param_dim: int


class MPDQNQMIXValueTeacher:
    """
    Adapter that exposes:
      - target Q_tot(s, u) as a fixed teacher signal
      - greedy joint action u*(s) from the online MP-DQN policy (per-agent) for rollouts

    This is designed for the "Value-Consistent World Model" described in `doc/价值一致性世界模型.md`.
    """

    def __init__(self, trainer: MPDQNQMIXTrainer, dims: MPDQNQMIXDims):
        self.trainer = trainer
        self.dims = dims
        # Freeze target critic parameters (teacher). We still allow gradients w.r.t the *input state*
        # when computing Q_tot(ŝ, u) inside value-consistency rollouts.
        for agent in self.trainer.agents:
            for p in agent.target_q_net.parameters():
                p.requires_grad_(False)
            agent.target_q_net.eval()
        for p in self.trainer.target_mixer.parameters():
            p.requires_grad_(False)
        self.trainer.target_mixer.eval()

    def _reshape_state(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        global_state: (B, N*S) -> (B, N, S)
        """
        if global_state.ndim != 2:
            raise ValueError(f"global_state must be (B, N*S), got {tuple(global_state.shape)}")
        bsz = int(global_state.shape[0])
        return global_state.view(bsz, int(self.dims.n_agents), int(self.dims.agent_state_dim))

    @torch.no_grad()
    def greedy_action(self, global_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get decentralized greedy joint action u*(s) using online networks (Double DQN style):
          a_i = argmax_a Q_i(s_i, θ_i(s_i))[a]
          params_i = θ_i(s_i)  (all-action params)

        Returns:
            action_discrete: (B, N) int64
            action_params_flat: (B, N, A*P) float32
        """
        state = self._reshape_state(global_state)
        bsz = int(state.shape[0])

        a_list = []
        params_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            params_all = self.trainer.agents[i].actor(s_i)  # (B, A, P)
            q_eval = self.trainer.agents[i].q_net(s_i, params_all)  # (B, A)
            a_i = torch.argmax(q_eval, dim=1)  # (B,)
            a_list.append(a_i)
            params_list.append(params_all.reshape(bsz, -1))

        action_discrete = torch.stack(a_list, dim=1).to(torch.long)
        action_params_flat = torch.stack(params_list, dim=1).to(torch.float32)
        return action_discrete, action_params_flat

    def q_tot_target(
        self,
        global_state: torch.Tensor,
        action_discrete: torch.Tensor,
        action_params_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target Q_tot(s, u) from *target* critics.

        Important:
        - Teacher parameters are frozen (no gradients to θ).
        - Gradients are allowed to flow to the *input state* (e.g., ŝ from the world model),
          so L_VC can shape the learned transition dynamics as described in the docs.

        Args:
            global_state: (B, N*S) float32
            action_discrete: (B, N) int64
            action_params_flat: (B, N, A*P) float32 (all-action params, MP-DQN style)
        Returns:
            q_tot: (B, 1) float32
        """
        state = self._reshape_state(global_state)
        bsz = int(state.shape[0])

        q_sa_list = []
        for i in range(int(self.dims.n_agents)):
            s_i = state[:, i, :]
            a_i = action_discrete[:, i].view(-1, 1)
            params_i = action_params_flat[:, i, :].view(bsz, int(self.dims.n_actions), int(self.dims.param_dim))
            q_all_i = self.trainer.agents[i].target_q_net(s_i, params_i)
            q_sa_i = q_all_i.gather(1, a_i)  # (B,1)
            q_sa_list.append(q_sa_i)

        agent_qs = torch.cat(q_sa_list, dim=1)  # (B,N)
        q_tot = self.trainer.target_mixer(agent_qs, global_state)  # (B,1)
        return q_tot.to(torch.float32)


__all__ = ["MPDQNQMIXDims", "MPDQNQMIXValueTeacher"]
