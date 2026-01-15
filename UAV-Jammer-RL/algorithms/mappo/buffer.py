from __future__ import division

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


class RolloutBuffer:
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
        rewards = np.asarray(self.reward, dtype=np.float32)  # (T, n_agents)
        values = np.asarray(self.value, dtype=np.float32)  # (T, n_agents)
        dones = np.asarray(self.done, dtype=np.float32)  # (T, n_agents)

        T = rewards.shape[0]
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros((self.n_agents,), dtype=np.float32)

        for t in reversed(range(T)):
            next_values = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + float(gamma) * next_values * next_non_terminal - values[t]
            last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns, advantages

    def as_batch(self, returns: np.ndarray, advantages: np.ndarray) -> RolloutBatch:
        obs = np.asarray(self.obs, dtype=np.float32)
        global_state = np.asarray(self.global_state, dtype=np.float32)
        agent_id = np.asarray(self.agent_id, dtype=np.int64)
        action_discrete = np.asarray(self.action_discrete, dtype=np.int64)
        action_cont = np.asarray(self.action_cont, dtype=np.float32)
        old_log_prob = np.asarray(self.log_prob, dtype=np.float32)

        # Flatten: (T, n_agents, ...) -> (T*n_agents, ...)
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
            "reward_mean": float(rewards.mean()),
            "reward_sum": float(rewards.sum()),
        }


__all__ = ["RolloutBuffer", "RolloutBatch"]

