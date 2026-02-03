from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class WorldModelSequenceReplayBuffer:
    """
    Replay buffer that stores step transitions:
      (s_t, u_t, r_t, s_{t+1}, done_t, env_id)

    and can *sample contiguous sequences* for training recurrent world models (e.g., GRU/RSSM).

    Notes:
    - Although each stored item is 1-step, RNN training still needs contiguous segments.
      This buffer uses `env_id` to keep per-env time order and sample sequences without
      explicitly storing pre-stacked history tensors on every transition.
    """

    # Tuple layout per transition:
    #   s:  (Ds,)
    #   ad: (N,)
    #   ap: (N, AP)
    #   r:  float
    #   ns: (Ds,)
    #   done: bool
    _Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, bool]

    def __init__(self, *, n_envs: int, capacity: int = 500_000):
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")

        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        # Approximate total-capacity by splitting into per-env ring buffers.
        self.capacity_total = capacity
        self.capacity_per_env = max(1, int(capacity) // int(self.n_envs))
        self._buffers: List[Deque[WorldModelSequenceReplayBuffer._Transition]] = [
            deque(maxlen=int(self.capacity_per_env)) for _ in range(int(self.n_envs))
        ]

    def __len__(self) -> int:
        return int(sum(len(b) for b in self._buffers))

    def add(
        self,
        *,
        env_id: int,
        state: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        reward_team: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        e = int(env_id)
        if e < 0 or e >= self.n_envs:
            raise ValueError(f"env_id out of range: {e} not in [0,{self.n_envs})")

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
        ad = np.asarray(action_discrete, dtype=np.int64).reshape(-1)
        ap = np.asarray(action_params, dtype=np.float32)
        if ap.ndim == 1:
            # Allow flattened (N*AP)
            ap = ap.reshape(int(ad.shape[0]), -1)
        self._buffers[e].append((s, ad, ap, float(reward_team), ns, bool(done)))

    def count_ready_envs(self, *, seq_len: int) -> int:
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        return int(sum(1 for b in self._buffers if len(b) >= seq_len))

    def sample_sequences(self, *, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        Sample contiguous sequences (no terminal transitions inside).

        Returns:
            state_seq:            (B,L,Ds)
            action_discrete_seq:  (B,L,N)
            action_params_seq:    (B,L,N,AP)
            reward_seq:           (B,L,1)
            next_state_seq:       (B,L,Ds)
            done_seq:             (B,L,1)
            env_id:               (B,)
        """
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        ready_envs = [i for i, b in enumerate(self._buffers) if len(b) >= seq_len]
        if len(ready_envs) == 0:
            raise ValueError(f"No env has enough transitions for seq_len={seq_len}")

        rng = np.random.default_rng()

        state_list = []
        ad_list = []
        ap_list = []
        r_list = []
        ns_list = []
        d_list = []
        env_id_list = []

        max_tries = 128
        for _ in range(batch_size):
            ok = False
            for _try in range(max_tries):
                e = int(rng.choice(ready_envs))
                buf = self._buffers[e]
                start = int(rng.integers(0, len(buf) - seq_len + 1))

                # Reject sequences containing terminal transitions (including the last one),
                # because next_state after done is often a reset state (non-physical).
                if any(bool(buf[start + k][5]) for k in range(seq_len)):
                    continue

                seq = [buf[start + k] for k in range(seq_len)]

                # Optional continuity check: s_{t+1} should match next transition's s_t.
                continuous = True
                for k in range(seq_len - 1):
                    if not np.allclose(seq[k][4], seq[k + 1][0], atol=1e-5, rtol=1e-5):
                        continuous = False
                        break
                if not continuous:
                    continue

                s_seq, ad_seq, ap_seq, r_seq, ns_seq, done_seq = zip(*seq)

                state_list.append(np.stack(s_seq, axis=0))
                ad_list.append(np.stack(ad_seq, axis=0))
                ap_list.append(np.stack(ap_seq, axis=0))
                r_list.append(np.asarray(r_seq, dtype=np.float32).reshape(seq_len, 1))
                ns_list.append(np.stack(ns_seq, axis=0))
                d_list.append(np.asarray(done_seq, dtype=np.float32).reshape(seq_len, 1))
                env_id_list.append(e)
                ok = True
                break

            if not ok:
                raise RuntimeError(
                    f"Failed to sample a valid sequence after {max_tries} tries. "
                    "Consider increasing capacity, lowering seq_len, or ensuring done/reset is handled."
                )

        return {
            "state_seq": np.asarray(state_list, dtype=np.float32),
            "action_discrete_seq": np.asarray(ad_list, dtype=np.int64),
            "action_params_seq": np.asarray(ap_list, dtype=np.float32),
            "reward_seq": np.asarray(r_list, dtype=np.float32),
            "next_state_seq": np.asarray(ns_list, dtype=np.float32),
            "done_seq": np.asarray(d_list, dtype=np.float32),
            "env_id": np.asarray(env_id_list, dtype=np.int32),
        }


__all__ = ["WorldModelSequenceReplayBuffer"]
