from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.world_model.action_encoding import encode_joint_action_exec
from algorithms.world_model.model import JointWorldModel, JointWorldModelConfig
from algorithms.world_model.qmix_adapters import MPDQNQMIXDims, MPDQNQMIXValueTeacher
from algorithms.world_model.value_consistency import TDlambdaConfig, rollout_td_lambda_return


@dataclass(frozen=True)
class WorldModelLosses:
    loss_state: float
    loss_reward: float
    loss_vc: float
    loss_total: float


class ValueConsistentWorldModelTrainer:
    """
    GRU world model trainer implementing:

      L_WM = L_S + alpha * L_R + eta * L_VC

    where:
      - L_S: MSE on Δs
      - L_R: MSE on team reward
      - L_VC: value-consistency regularizer (TD(lambda) rollout vs target Q_tot teacher)

    Inputs are sequences (contiguous segments) built from (s,u,r,s',done,env_id) replay.
    """

    def __init__(
        self,
        *,
        wm_cfg: JointWorldModelConfig,
        n_agents: int,
        n_channel: int,
        n_des: int,
        n_actions: int,
        param_dim: int,
        alpha: float = 1.0,
        eta: float = 0.0,
        td_cfg: Optional[TDlambdaConfig] = None,
        lr: float = 1e-3,
        power_min_dbm: float | None = None,
        power_max_dbm: float | None = None,
        device: str | None = None,
    ):
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.td_cfg = td_cfg or TDlambdaConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.wm = JointWorldModel(wm_cfg).to(self.device)
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=float(lr))

        self.n_agents = int(n_agents)
        self.n_channel = int(n_channel)
        self.n_des = int(n_des)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.power_min_dbm = float(power_min_dbm) if power_min_dbm is not None else None
        self.power_max_dbm = float(power_max_dbm) if power_max_dbm is not None else None

    def _encode_actions_seq(self, action_discrete_seq: torch.Tensor, action_params_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode execution-time joint actions for a sequence.

        Args:
            action_discrete_seq: (B,L,N)
            action_params_seq:   (B,L,N,AP)
        Returns:
            action_enc_seq:      (B,L,Du)
        """
        if action_discrete_seq.ndim != 3:
            raise ValueError(f"action_discrete_seq must be (B,L,N), got {tuple(action_discrete_seq.shape)}")
        if action_params_seq.ndim != 4:
            raise ValueError(f"action_params_seq must be (B,L,N,AP), got {tuple(action_params_seq.shape)}")

        bsz, seq_len, n_agents = action_discrete_seq.shape
        if int(n_agents) != int(self.n_agents):
            raise ValueError(f"Expected N={self.n_agents}, got {int(n_agents)}")

        # Flatten time into batch for vectorized encoding: (B*L, N, ...)
        ad = action_discrete_seq.reshape(int(bsz * seq_len), int(n_agents))
        ap = action_params_seq.reshape(int(bsz * seq_len), int(n_agents), -1)
        u = encode_joint_action_exec(
            ad,
            ap,
            n_agents=int(self.n_agents),
            n_channel=int(self.n_channel),
            n_des=int(self.n_des),
            n_actions=int(self.n_actions),
            param_dim=int(self.param_dim),
            power_min_dbm=self.power_min_dbm,
            power_max_dbm=self.power_max_dbm,
        )
        return u.reshape(int(bsz), int(seq_len), -1)

    @staticmethod
    def _assert_contiguous(state_seq: torch.Tensor, next_state_seq: torch.Tensor) -> None:
        """
        Ensure s_{t+1} == next_state at t and equals next state's state at t+1 (no off-by-one).
        """
        if state_seq.ndim != 3 or next_state_seq.ndim != 3:
            return
        if int(state_seq.shape[1]) < 2:
            return
        a = next_state_seq[:, :-1, :]
        b = state_seq[:, 1:, :]
        if torch.allclose(a, b, atol=1e-5, rtol=1e-5):
            return
        diff = (a - b).abs()
        raise ValueError(
            "Non-contiguous sequence: next_state_seq[:, :-1] != state_seq[:, 1:]. "
            f"max|diff|={float(diff.max().item()):.6g}, mean|diff|={float(diff.mean().item()):.6g}"
        )

    @torch.no_grad()
    def eval_losses(
        self,
        *,
        state_seq: torch.Tensor,  # (B,L,Ds)
        action_discrete_seq: torch.Tensor,  # (B,L,N)
        action_params_seq: torch.Tensor,  # (B,L,N,AP)
        reward_seq: torch.Tensor,  # (B,L,1)
        next_state_seq: torch.Tensor,  # (B,L,Ds)
        value_teacher: Optional[MPDQNQMIXValueTeacher] = None,
    ) -> Tuple[WorldModelLosses, dict]:
        """
        Evaluation-only losses (NO gradient).
        """
        state_seq = state_seq.to(self.device).to(torch.float32)
        next_state_seq = next_state_seq.to(self.device).to(torch.float32)
        action_discrete_seq = action_discrete_seq.to(self.device).to(torch.long)
        action_params_seq = action_params_seq.to(self.device).to(torch.float32)
        reward_seq = reward_seq.to(self.device).to(torch.float32)

        self._assert_contiguous(state_seq, next_state_seq)

        action_enc_seq = self._encode_actions_seq(action_discrete_seq, action_params_seq)
        delta_true = next_state_seq - state_seq
        delta_pred, reward_pred, _, h_last = self.wm(state_seq, action_enc_seq)

        loss_state = F.mse_loss(delta_pred, delta_true)
        loss_reward = F.mse_loss(reward_pred, reward_seq)

        loss_vc = torch.zeros((), dtype=torch.float32, device=self.device)
        debug = {"uses_vc": False}

        if value_teacher is not None and float(self.eta) > 0.0:
            debug["uses_vc"] = True
            s_t = state_seq[:, -1, :]
            ad_t = action_discrete_seq[:, -1, :]
            ap_t = action_params_seq[:, -1, :, :]

            q_teacher = value_teacher.q_tot_target(s_t, ad_t, ap_t)  # already frozen params

            def _policy_fn(s_flat: torch.Tensor):
                with torch.no_grad():
                    a_star, p_star = value_teacher.greedy_action(s_flat)
                    u_star = encode_joint_action_exec(
                        a_star,
                        p_star,
                        n_agents=int(self.n_agents),
                        n_channel=int(self.n_channel),
                        n_des=int(self.n_des),
                        n_actions=int(self.n_actions),
                        param_dim=int(self.param_dim),
                        power_min_dbm=self.power_min_dbm,
                        power_max_dbm=self.power_max_dbm,
                    )
                return u_star, a_star, p_star

            g_lambda, rewards_hat = rollout_td_lambda_return(
                wm=self.wm,
                state_seq=state_seq,
                action_seq=action_enc_seq,
                policy_fn=_policy_fn,
                q_tot_target_fn=value_teacher.q_tot_target,
                cfg=self.td_cfg,
            )
            loss_vc = F.mse_loss(q_teacher, g_lambda)
            debug.update(
                {
                    "q_teacher_mean": float(q_teacher.mean().item()),
                    "g_lambda_mean": float(g_lambda.mean().item()),
                    "rewards_hat_mean": float(rewards_hat.mean().item()),
                }
            )

        loss_total = loss_state + self.alpha * loss_reward + self.eta * loss_vc
        losses = WorldModelLosses(
            loss_state=float(loss_state.item()),
            loss_reward=float(loss_reward.item()),
            loss_vc=float(loss_vc.item()),
            loss_total=float(loss_total.item()),
        )
        return losses, debug

    def train_step(
        self,
        *,
        state_seq: torch.Tensor,
        action_discrete_seq: torch.Tensor,
        action_params_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
        value_teacher: Optional[MPDQNQMIXValueTeacher] = None,
    ) -> Tuple[WorldModelLosses, dict]:
        self.wm.train()
        self.opt.zero_grad(set_to_none=True)

        state_seq = state_seq.to(self.device).to(torch.float32)
        next_state_seq = next_state_seq.to(self.device).to(torch.float32)
        action_discrete_seq = action_discrete_seq.to(self.device).to(torch.long)
        action_params_seq = action_params_seq.to(self.device).to(torch.float32)
        reward_seq = reward_seq.to(self.device).to(torch.float32)

        self._assert_contiguous(state_seq, next_state_seq)

        action_enc_seq = self._encode_actions_seq(action_discrete_seq, action_params_seq)
        delta_true = next_state_seq - state_seq
        delta_pred, reward_pred, _, _ = self.wm(state_seq, action_enc_seq)

        loss_state = F.mse_loss(delta_pred, delta_true)
        loss_reward = F.mse_loss(reward_pred, reward_seq)
        loss_vc = torch.zeros((), dtype=torch.float32, device=self.device)

        debug = {}
        loss_total = loss_state + self.alpha * loss_reward

        if value_teacher is not None and float(self.eta) > 0.0:
            s_t = state_seq[:, -1, :]
            ad_t = action_discrete_seq[:, -1, :]
            ap_t = action_params_seq[:, -1, :, :]

            with torch.no_grad():
                q_teacher = value_teacher.q_tot_target(s_t, ad_t, ap_t)

            def _policy_fn(s_flat: torch.Tensor):
                with torch.no_grad():
                    a_star, p_star = value_teacher.greedy_action(s_flat)
                    u_star = encode_joint_action_exec(
                        a_star,
                        p_star,
                        n_agents=int(self.n_agents),
                        n_channel=int(self.n_channel),
                        n_des=int(self.n_des),
                        n_actions=int(self.n_actions),
                        param_dim=int(self.param_dim),
                        power_min_dbm=self.power_min_dbm,
                        power_max_dbm=self.power_max_dbm,
                    )
                return u_star, a_star, p_star

            g_lambda, rewards_hat = rollout_td_lambda_return(
                wm=self.wm,
                state_seq=state_seq,
                action_seq=action_enc_seq,
                policy_fn=_policy_fn,
                q_tot_target_fn=value_teacher.q_tot_target,
                cfg=self.td_cfg,
            )
            loss_vc = F.mse_loss(q_teacher, g_lambda)
            loss_total = loss_total + self.eta * loss_vc
            debug.update(
                {
                    "q_teacher_mean": float(q_teacher.mean().item()),
                    "g_lambda_mean": float(g_lambda.mean().item()),
                    "rewards_hat_mean": float(rewards_hat.mean().item()),
                }
            )

        loss_total.backward()
        self.opt.step()

        losses = WorldModelLosses(
            loss_state=float(loss_state.item()),
            loss_reward=float(loss_reward.item()),
            loss_vc=float(loss_vc.item()),
            loss_total=float(loss_total.item()),
        )
        return losses, debug


__all__ = ["ValueConsistentWorldModelTrainer", "WorldModelLosses", "MPDQNQMIXValueTeacher", "MPDQNQMIXDims"]

