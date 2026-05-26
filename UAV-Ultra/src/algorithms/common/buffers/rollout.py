"""MAPPO on-policy rollout buffer + GAE 计算（baseline `algorithms/mappo/buffer.py` 直接迁移）。

``compute_gae`` 是公式级回归测试的目标（plan 通过标准）：手算 3-step GAE 验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class RolloutBatch:
    obs: np.ndarray
    global_state: np.ndarray
    agent_id: np.ndarray
    action_discrete: np.ndarray
    action_cont: np.ndarray
    old_log_prob: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 ``(T, n_agents)`` 形状的 reward/value/done 序列计算 GAE。

    与 baseline ``RolloutBuffer.compute_returns_and_advantages`` 公式 1:1：

        last_gae = 0
        for t = T-1 ... 0:
            next_values = last_value if t == T-1 else values[t+1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    last_value = np.asarray(last_value, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"shape mismatch: rewards={rewards.shape}, values={values.shape}, dones={dones.shape}"
        )

    T = rewards.shape[0]
    n_agents = rewards.shape[1] if rewards.ndim == 2 else 1
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros((n_agents,), dtype=np.float32) if rewards.ndim == 2 else np.float32(0.0)

    for t in reversed(range(T)):
        next_values = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + float(gamma) * next_values * next_non_terminal - values[t]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages


class RolloutBuffer:
    """On-policy rollout，与 baseline 等价。"""

    def __init__(self, n_agents: int):
        self.n_agents = int(n_agents)
        self.clear()

    def clear(self) -> None:
        self.obs = []
        self.global_state = []
        self.agent_id = []
        self.action_discrete = []
        self.action_cont = []
        self.log_prob = []
        self.value = []
        self.reward = []
        self.done = []

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        agent_id: np.ndarray,
        action_discrete: np.ndarray,
        action_cont: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.global_state.append(np.asarray(global_state, dtype=np.float32))
        self.agent_id.append(np.asarray(agent_id, dtype=np.int64))
        self.action_discrete.append(np.asarray(action_discrete, dtype=np.int64))
        self.action_cont.append(np.asarray(action_cont, dtype=np.float32))
        self.log_prob.append(np.asarray(log_prob, dtype=np.float32))
        self.value.append(np.asarray(value, dtype=np.float32))
        self.reward.append(np.asarray(reward, dtype=np.float32))
        self.done.append(np.asarray(done, dtype=np.float32))

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.reward, dtype=np.float32)
        values = np.asarray(self.value, dtype=np.float32)
        dones = np.asarray(self.done, dtype=np.float32)
        return compute_gae(rewards, values, dones, last_value, gamma=gamma, gae_lambda=gae_lambda)

    def as_batch(self, returns: np.ndarray, advantages: np.ndarray) -> RolloutBatch:
        obs = np.asarray(self.obs, dtype=np.float32)
        global_state = np.asarray(self.global_state, dtype=np.float32)
        agent_id = np.asarray(self.agent_id, dtype=np.int64)
        action_discrete = np.asarray(self.action_discrete, dtype=np.int64)
        action_cont = np.asarray(self.action_cont, dtype=np.float32)
        old_log_prob = np.asarray(self.log_prob, dtype=np.float32)

        T = obs.shape[0]
        obs = obs.reshape(T * self.n_agents, -1)
        global_state = global_state.reshape(T * self.n_agents, -1)
        agent_id = agent_id.reshape(T * self.n_agents)
        action_discrete = action_discrete.reshape(T * self.n_agents)
        action_cont = action_cont.reshape(T * self.n_agents, -1)
        old_log_prob = old_log_prob.reshape(T * self.n_agents)
        returns = np.asarray(returns, dtype=np.float32).reshape(T * self.n_agents)
        advantages = np.asarray(advantages, dtype=np.float32).reshape(T * self.n_agents)

        return RolloutBatch(
            obs=obs,
            global_state=global_state,
            agent_id=agent_id,
            action_discrete=action_discrete,
            action_cont=action_cont,
            old_log_prob=old_log_prob,
            returns=returns,
            advantages=advantages,
        )

    def summary(self) -> Dict[str, float]:
        rewards = np.asarray(self.reward, dtype=np.float32)
        return {
            "reward_mean": float(rewards.mean()) if rewards.size > 0 else 0.0,
            "reward_sum": float(rewards.sum()) if rewards.size > 0 else 0.0,
        }


__all__ = ["RolloutBatch", "RolloutBuffer", "compute_gae"]
