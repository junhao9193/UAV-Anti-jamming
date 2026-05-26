"""MAPPO trainer：actor + critic + RolloutBuffer + GAE + PPO clip。

与 baseline ``algorithms/mappo/agent.py`` 1:1 对应：
- 所有 PPO 超参（``epochs / minibatch / clip / ent_coef / vf_coef / gae_lambda``）从
  ``MAPPOConfig`` 取，**不硬编码**（plan locked decision #9）。
- ``cont_dim = specs.param_dim_per_action(env_cfg) = n_des``。
- ``update(batch)`` 接受 ``RolloutBatch``（已展平为 ``(T*N, ...)``）+ 完成 GAE 归一化。
- ``train_step()`` 协议返回 ``Optional[dict]``：当 buffer 未准备好时为 None；MAPPO 是
  on-policy，由 Runner（Stage 5）每个 rollout 段调一次 update，本类不持 buffer。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.common.buffers import RolloutBatch
from src.algorithms.common.networks import CentralValueNet, HybridActor
from src.algorithms.mappo.agent import ActResult
from src.config import specs
from src.config.schema import EnvConfig, MAPPOConfig


class MAPPOTrainer:
    """中心化 critic + 分布式 hybrid actor（discrete + Beta 分布）。"""

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        algo_cfg: MAPPOConfig,
        device: Optional[str] = None,
    ):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg

        self.n_agents = int(env_cfg.n_ch)
        self.obs_dim = int(specs.state_dim(env_cfg))
        self.n_actions = int(specs.action_dim(env_cfg))
        # 连续动作维 = 每动作的功率参数数 = n_des
        self.cont_dim = int(specs.param_dim_per_action(env_cfg))
        self.global_state_dim = self.n_agents * self.obs_dim

        if device is None:
            chosen = algo_cfg.device
            resolved = "cuda" if (chosen == "auto" and torch.cuda.is_available()) else (
                "cpu" if chosen == "auto" else chosen
            )
            self.device = torch.device(resolved)
        else:
            self.device = torch.device(device)

        # PPO 超参全部从 algo_cfg 取
        self.gamma = float(algo_cfg.gamma)
        self.gae_lambda = float(algo_cfg.gae_lambda)
        self.clip_range = float(algo_cfg.clip_range)
        self.ent_coef = float(algo_cfg.ent_coef)
        self.vf_coef = float(algo_cfg.vf_coef)
        self.update_epochs = int(algo_cfg.update_epochs)
        self.minibatch_size = int(algo_cfg.minibatch_size)
        self.max_grad_norm = float(algo_cfg.max_grad_norm)
        self.lr = float(algo_cfg.lr)

        self.actor = HybridActor(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            cont_dim=self.cont_dim,
            n_agents=self.n_agents,
        ).to(self.device)
        self.critic = CentralValueNet(
            global_state_dim=self.global_state_dim,
            n_agents=self.n_agents,
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    # ----------------------- Helper -----------------------

    def _onehot(self, agent_id: torch.Tensor) -> torch.Tensor:
        return F.one_hot(agent_id.long(), num_classes=self.n_agents).float()

    # ----------------------- 动作选择 -----------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        agent_id: int,
        deterministic: bool = False,
    ) -> ActResult:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device).view(1, -1)
        gs_t = torch.from_numpy(np.asarray(global_state, dtype=np.float32)).to(self.device).view(1, -1)
        aid_t = torch.tensor([int(agent_id)], device=self.device, dtype=torch.int64)
        aid_oh = self._onehot(aid_t)

        logits, alpha, beta = self.actor(obs_t, aid_oh)
        dist_disc = torch.distributions.Categorical(logits=logits)
        dist_cont = torch.distributions.Beta(alpha, beta)

        if deterministic:
            action_discrete = int(torch.argmax(logits, dim=-1).item())
            action_cont = (alpha / (alpha + beta)).squeeze(0)
        else:
            action_discrete = int(dist_disc.sample().item())
            action_cont = dist_cont.sample().squeeze(0)

        action_cont = torch.clamp(action_cont, 1e-6, 1.0 - 1e-6)

        log_prob = dist_disc.log_prob(torch.tensor(action_discrete, device=self.device)).view(1)
        log_prob = log_prob + dist_cont.log_prob(action_cont.unsqueeze(0)).sum(dim=-1)

        value = self.critic(gs_t, aid_oh)

        return ActResult(
            action_discrete=action_discrete,
            action_cont=action_cont.detach().cpu().numpy().astype(np.float32),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
        )

    @torch.no_grad()
    def value(self, global_state: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
        gs_t = torch.from_numpy(np.asarray(global_state, dtype=np.float32)).to(self.device)
        aid_t = torch.from_numpy(np.asarray(agent_ids, dtype=np.int64)).to(self.device)
        aid_oh = self._onehot(aid_t)
        v = self.critic(gs_t, aid_oh)
        return v.detach().cpu().numpy().astype(np.float32)

    # ----------------------- Update -----------------------

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        """一次 PPO update，与 baseline ``MAPPOAgent.update`` 等价。

        batch 已被 ``RolloutBuffer.as_batch`` 展平为 ``(T*N, ...)``。
        """
        obs = torch.from_numpy(batch.obs).to(self.device)
        global_state = torch.from_numpy(batch.global_state).to(self.device)
        agent_id = torch.from_numpy(batch.agent_id).to(self.device)
        action_discrete = torch.from_numpy(batch.action_discrete).to(self.device)
        action_cont = torch.from_numpy(batch.action_cont).to(self.device)
        old_log_prob = torch.from_numpy(batch.old_log_prob).to(self.device)
        returns = torch.from_numpy(batch.returns).to(self.device)
        advantages = torch.from_numpy(batch.advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        aid_oh = self._onehot(agent_id)
        n = obs.shape[0]
        indices = np.arange(n)

        loss_pi_sum = 0.0
        loss_v_sum = 0.0
        entropy_sum = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_idx_t = torch.from_numpy(mb_idx).to(self.device)

                obs_mb = obs[mb_idx_t]
                gs_mb = global_state[mb_idx_t]
                aid_mb = aid_oh[mb_idx_t]
                act_d_mb = action_discrete[mb_idx_t]
                act_c_mb = torch.clamp(action_cont[mb_idx_t], 1e-6, 1.0 - 1e-6)
                old_lp_mb = old_log_prob[mb_idx_t]
                ret_mb = returns[mb_idx_t]
                adv_mb = advantages[mb_idx_t]

                logits, alpha, beta = self.actor(obs_mb, aid_mb)
                dist_disc = torch.distributions.Categorical(logits=logits)
                dist_cont = torch.distributions.Beta(alpha, beta)

                new_log_prob = dist_disc.log_prob(act_d_mb) + dist_cont.log_prob(act_c_mb).sum(dim=-1)
                entropy = dist_disc.entropy() + dist_cont.entropy().sum(dim=-1)

                ratio = torch.exp(new_log_prob - old_lp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                loss_pi = -torch.min(surr1, surr2).mean()

                v_pred = self.critic(gs_mb, aid_mb)
                loss_v = F.mse_loss(v_pred, ret_mb)

                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy.mean()

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.actor_opt.step()
                self.critic_opt.step()

                loss_pi_sum += float(loss_pi.item())
                loss_v_sum += float(loss_v.item())
                entropy_sum += float(entropy.mean().item())
                n_updates += 1

        return {
            "loss_pi": loss_pi_sum / max(n_updates, 1),
            "loss_v": loss_v_sum / max(n_updates, 1),
            "entropy": entropy_sum / max(n_updates, 1),
        }

    # ----------------------- Trainer 协议占位 -----------------------

    def train_step(self) -> Optional[Dict[str, float]]:
        """MAPPO 是 on-policy，本类不持 rollout buffer；返回 None。

        Stage 5 Runner 直接调用 ``update(batch)`` 喂入 ``RolloutBuffer.as_batch()`` 输出。
        """
        return None


__all__ = ["MAPPOTrainer"]
