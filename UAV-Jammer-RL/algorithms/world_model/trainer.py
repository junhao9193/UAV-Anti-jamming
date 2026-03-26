from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    loss_kl: float
    loss_vc: float
    loss_total: float


class ValueConsistentWorldModelTrainer:
    """
    RSSM world model trainer implementing:

      L_WM = L_S + alpha * L_R + kl_beta * L_KL + eta * L_VC

    where:
      - L_S: MSE on delta-s
      - L_R: MSE on team reward
      - L_KL: posterior-prior KL for the RSSM latent state
      - L_VC: value-consistency regularizer (TD(lambda) rollout vs target Q_tot teacher)
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
        max_grad_norm: float = 10.0,
        value_clip: float | None = 1000.0,
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
        self.kl_beta = float(self.wm.cfg.kl_beta)
        self.free_nats = float(self.wm.cfg.free_nats)
        self.max_grad_norm = float(max_grad_norm)
        self.value_clip = None if value_clip is None else float(value_clip)

    @staticmethod
    def _robust_loss(pred: torch.Tensor, target: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=float(beta))

    def _clip_value_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_clip is None or self.value_clip <= 0.0:
            return x
        return torch.clamp(x, min=-self.value_clip, max=self.value_clip)

    def _encode_actions_seq(self, action_discrete_seq: torch.Tensor, action_params_seq: torch.Tensor) -> torch.Tensor:
        if action_discrete_seq.ndim != 3:
            raise ValueError(f"action_discrete_seq must be (B,L,N), got {tuple(action_discrete_seq.shape)}")
        if action_params_seq.ndim != 4:
            raise ValueError(f"action_params_seq must be (B,L,N,AP), got {tuple(action_params_seq.shape)}")

        bsz, seq_len, n_agents = action_discrete_seq.shape
        if int(n_agents) != int(self.n_agents):
            raise ValueError(f"Expected N={self.n_agents}, got {int(n_agents)}")

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

    def _kl_loss(
        self,
        prior_mean_seq: torch.Tensor,
        prior_std_seq: torch.Tensor,
        post_mean_seq: torch.Tensor,
        post_std_seq: torch.Tensor,
    ) -> torch.Tensor:
        post_var = post_std_seq.pow(2)
        prior_var = prior_std_seq.pow(2)
        kl = torch.log(prior_std_seq / post_std_seq) + (post_var + (post_mean_seq - prior_mean_seq).pow(2)) / (2.0 * prior_var) - 0.5
        kl = kl.sum(dim=-1)
        if self.free_nats > 0.0:
            kl = torch.clamp(kl, min=self.free_nats)
        return kl.mean()

    @torch.no_grad()
    def eval_losses(
        self,
        *,
        state_seq: torch.Tensor,
        action_discrete_seq: torch.Tensor,
        action_params_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
        value_teacher: Optional[MPDQNQMIXValueTeacher] = None,
    ) -> Tuple[WorldModelLosses, dict]:
        state_seq = state_seq.to(self.device).to(torch.float32)
        next_state_seq = next_state_seq.to(self.device).to(torch.float32)
        action_discrete_seq = action_discrete_seq.to(self.device).to(torch.long)
        action_params_seq = action_params_seq.to(self.device).to(torch.float32)
        reward_seq = reward_seq.to(self.device).to(torch.float32)

        self._assert_contiguous(state_seq, next_state_seq)

        action_enc_seq = self._encode_actions_seq(action_discrete_seq, action_params_seq)
        delta_true = next_state_seq - state_seq
        obs = self.wm.observe(state_seq, action_enc_seq, sample=False)
        delta_pred = obs.delta_seq
        reward_pred = obs.reward_seq

        loss_state = self._robust_loss(delta_pred, delta_true)
        loss_reward = self._robust_loss(reward_pred, reward_seq)
        loss_kl = self._kl_loss(obs.prior_mean_seq, obs.prior_std_seq, obs.post_mean_seq, obs.post_std_seq)

        loss_vc = torch.zeros((), dtype=torch.float32, device=self.device)
        debug = {"uses_vc": False, "uses_kl": True}

        if value_teacher is not None and float(self.eta) > 0.0:
            debug["uses_vc"] = True
            s_t = state_seq[:, -1, :]
            ad_t = action_discrete_seq[:, -1, :]
            ap_t = action_params_seq[:, -1, :, :]

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
            q_teacher = self._clip_value_target(q_teacher)
            g_lambda = self._clip_value_target(g_lambda)
            loss_vc = self._robust_loss(q_teacher, g_lambda, beta=10.0)
            debug.update(
                {
                    "q_teacher_mean": float(q_teacher.mean().item()),
                    "g_lambda_mean": float(g_lambda.mean().item()),
                    "rewards_hat_mean": float(rewards_hat.mean().item()),
                }
            )

        loss_total = loss_state + self.alpha * loss_reward + self.kl_beta * loss_kl + self.eta * loss_vc
        losses = WorldModelLosses(
            loss_state=float(loss_state.item()),
            loss_reward=float(loss_reward.item()),
            loss_kl=float(loss_kl.item()),
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
        obs = self.wm.observe(state_seq, action_enc_seq, sample=True)
        delta_pred = obs.delta_seq
        reward_pred = obs.reward_seq

        if not torch.isfinite(delta_pred).all() or not torch.isfinite(reward_pred).all():
            self.opt.zero_grad(set_to_none=True)
            losses = WorldModelLosses(
                loss_state=float("inf"),
                loss_reward=float("inf"),
                loss_kl=float("inf"),
                loss_vc=float("inf"),
                loss_total=float("inf"),
            )
            return losses, {"skipped_nonfinite_outputs": True}

        loss_state = self._robust_loss(delta_pred, delta_true)
        loss_reward = self._robust_loss(reward_pred, reward_seq)
        loss_kl = self._kl_loss(obs.prior_mean_seq, obs.prior_std_seq, obs.post_mean_seq, obs.post_std_seq)
        loss_vc = torch.zeros((), dtype=torch.float32, device=self.device)

        debug = {"uses_kl": True}
        loss_total = loss_state + self.alpha * loss_reward + self.kl_beta * loss_kl

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
            if torch.isfinite(q_teacher).all() and torch.isfinite(g_lambda).all():
                q_teacher = self._clip_value_target(q_teacher)
                g_lambda = self._clip_value_target(g_lambda)
                loss_vc = self._robust_loss(q_teacher, g_lambda, beta=10.0)
                loss_total = loss_total + self.eta * loss_vc
                debug.update(
                    {
                        "q_teacher_mean": float(q_teacher.mean().item()),
                        "g_lambda_mean": float(g_lambda.mean().item()),
                        "rewards_hat_mean": float(rewards_hat.mean().item()),
                    }
                )
            else:
                debug["skipped_nonfinite_vc"] = True

        if not torch.isfinite(loss_total):
            self.opt.zero_grad(set_to_none=True)
            losses = WorldModelLosses(
                loss_state=float(loss_state.item()),
                loss_reward=float(loss_reward.item()),
                loss_kl=float(loss_kl.item()),
                loss_vc=float(loss_vc.item()),
                loss_total=float("inf"),
            )
            return losses, {**debug, "skipped_nonfinite_loss": True}

        loss_total.backward()
        grad_norm = None
        if self.max_grad_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                self.opt.zero_grad(set_to_none=True)
                losses = WorldModelLosses(
                    loss_state=float(loss_state.item()),
                    loss_reward=float(loss_reward.item()),
                    loss_kl=float(loss_kl.item()),
                    loss_vc=float(loss_vc.item()),
                    loss_total=float("inf"),
                )
                return losses, {**debug, "skipped_nonfinite_grad": True}
        self.opt.step()
        if grad_norm is not None:
            debug["grad_norm"] = float(torch.as_tensor(grad_norm).item())

        losses = WorldModelLosses(
            loss_state=float(loss_state.item()),
            loss_reward=float(loss_reward.item()),
            loss_kl=float(loss_kl.item()),
            loss_vc=float(loss_vc.item()),
            loss_total=float(loss_total.item()),
        )
        return losses, debug


__all__ = ["ValueConsistentWorldModelTrainer", "WorldModelLosses", "MPDQNQMIXValueTeacher", "MPDQNQMIXDims"]
