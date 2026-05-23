from typing import Dict, Optional

import numpy as np


class MPDQNReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._size = 0
        self._pos = 0
        self._state: Optional[np.ndarray] = None
        self._action_discrete: Optional[np.ndarray] = None
        self._action_params: Optional[np.ndarray] = None
        self._reward: Optional[np.ndarray] = None
        self._next_state: Optional[np.ndarray] = None
        self._done: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._size

    def _ensure_storage(self, state: np.ndarray, action_params: np.ndarray, next_state: np.ndarray) -> None:
        if self._state is not None:
            return

        state_shape = tuple(np.asarray(state, dtype=np.float32).shape)
        params_shape = tuple(np.asarray(action_params, dtype=np.float32).reshape(-1).shape)
        next_state_shape = tuple(np.asarray(next_state, dtype=np.float32).shape)
        if state_shape != next_state_shape:
            raise ValueError(f"state and next_state shapes must match, got {state_shape} and {next_state_shape}")

        self._state = np.empty((self.capacity, *state_shape), dtype=np.float32)
        self._action_discrete = np.empty((self.capacity,), dtype=np.int64)
        self._action_params = np.empty((self.capacity, *params_shape), dtype=np.float32)
        self._reward = np.empty((self.capacity,), dtype=np.float32)
        self._next_state = np.empty((self.capacity, *next_state_shape), dtype=np.float32)
        self._done = np.empty((self.capacity,), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action_discrete: int,
        action_params: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        params_arr = np.asarray(action_params, dtype=np.float32).reshape(-1)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        self._ensure_storage(state_arr, params_arr, next_state_arr)

        assert self._state is not None
        assert self._action_discrete is not None
        assert self._action_params is not None
        assert self._reward is not None
        assert self._next_state is not None
        assert self._done is not None

        self._state[self._pos] = state_arr
        self._action_discrete[self._pos] = int(action_discrete)
        self._action_params[self._pos] = params_arr
        self._reward[self._pos] = float(reward)
        self._next_state[self._pos] = next_state_arr
        self._done[self._pos] = float(bool(done))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size > self._size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer size={self._size}")
        idx = np.random.choice(self._size, size=batch_size, replace=False)

        assert self._state is not None
        assert self._action_discrete is not None
        assert self._action_params is not None
        assert self._reward is not None
        assert self._next_state is not None
        assert self._done is not None

        return {
            "state": self._state[idx],
            "action_discrete": self._action_discrete[idx],
            "action_params": self._action_params[idx],
            "reward": self._reward[idx],
            "next_state": self._next_state[idx],
            "done": self._done[idx],
        }


__all__ = ["MPDQNReplayBuffer"]
