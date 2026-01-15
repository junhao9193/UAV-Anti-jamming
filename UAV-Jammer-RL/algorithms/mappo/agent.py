from __future__ import division

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.mappo.buffer import RolloutBatch
from algorithms.mappo.model import CentralValueNet, HybridActor


@dataclass
class ActResult:
    action_discrete: int
    action_cont: np.ndarray
    log_prob: float
    value: float


class MAPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cont_dim: int,
        n_agents: int,
        global_state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        lr: float = 3e-4,
        update_epochs: int = 10,
        minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cont_dim = int(cont_dim)
        self.n_agents = int(n_agents)
        self.global_state_dim = int(global_state_dim)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = HybridActor(self.obs_dim, self.n_actions, self.cont_dim, self.n_agents).to(self.device)
        self.critic = CentralValueNet(self.global_state_dim, self.n_agents).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(lr))

    def _onehot(self, agent_id: torch.Tensor) -> torch.Tensor:
        return F.one_hot(agent_id.long(), num_classes=self.n_agents).float()

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

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
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
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
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


__all__ = ["MAPPOAgent", "ActResult"]

