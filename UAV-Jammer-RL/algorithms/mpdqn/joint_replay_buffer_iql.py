from __future__ import division

from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


class MPDQNJointIQLReplayBuffer:
    """
    Joint replay buffer for IQL-style training (no mixer, no centralized critic).

    Stores one transition containing *all agents* data, but rewards remain per-agent:
        state:         (N, S)
        action_discrete:(N,)
        action_params: (N, A*P)
        reward:        (N,)
        next_state:    (N, S)
        done:          bool
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._buffer: Deque[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]
        ] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action_discrete, dtype=np.int64),
                np.asarray(action_params, dtype=np.float32),
                np.asarray(reward, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        idx = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, params, rewards, next_states, dones = zip(*(self._buffer[i] for i in idx))

        return {
            "state": np.asarray(states, dtype=np.float32),  # (B,N,S)
            "action_discrete": np.asarray(actions, dtype=np.int64),  # (B,N)
            "action_params": np.asarray(params, dtype=np.float32),  # (B,N,A*P)
            "reward": np.asarray(rewards, dtype=np.float32),  # (B,N)
            "next_state": np.asarray(next_states, dtype=np.float32),  # (B,N,S)
            "done": np.asarray(dones, dtype=np.float32),  # (B,)
        }


__all__ = ["MPDQNJointIQLReplayBuffer"]

