from __future__ import division

from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


class MPDQNReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._buffer: Deque[Tuple[np.ndarray, int, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action_discrete: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((state, int(action_discrete), action_params, float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        idx = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, params, rewards, next_states, dones = zip(*(self._buffer[i] for i in idx))

        return {
            "state": np.asarray(states, dtype=np.float32),
            "action_discrete": np.asarray(actions, dtype=np.int64),
            "action_params": np.asarray(params, dtype=np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),
            "next_state": np.asarray(next_states, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
        }


__all__ = ["MPDQNReplayBuffer"]
