"""``MPDQNReplayBuffer``（compat-only）push / sample 形状 + dtype 测试。"""

from __future__ import annotations

import numpy as np

from src.algorithms.common.buffers import MPDQNReplayBuffer


STATE_DIM, A_TIMES_P = 11, 15


def test_replay_buffer_add_sample_shapes():
    np.random.seed(0)
    buf = MPDQNReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(
            state=np.random.randn(STATE_DIM).astype(np.float32),
            action_discrete=int(np.random.randint(5)),
            action_params=np.random.rand(A_TIMES_P).astype(np.float32),
            reward=float(np.random.randn()),
            next_state=np.random.randn(STATE_DIM).astype(np.float32),
            done=bool(False),
        )
    assert len(buf) == 20

    batch = buf.sample(8)
    assert batch["state"].shape == (8, STATE_DIM) and batch["state"].dtype == np.float32
    assert batch["action_discrete"].shape == (8,) and batch["action_discrete"].dtype == np.int64
    assert batch["action_params"].shape == (8, A_TIMES_P) and batch["action_params"].dtype == np.float32
    assert batch["reward"].shape == (8,) and batch["reward"].dtype == np.float32
    assert batch["next_state"].shape == (8, STATE_DIM)
    assert batch["done"].shape == (8,) and batch["done"].dtype == np.float32


def test_replay_buffer_capacity_enforced():
    buf = MPDQNReplayBuffer(capacity=5)
    for _ in range(10):
        buf.add(
            state=np.zeros(STATE_DIM, dtype=np.float32),
            action_discrete=0,
            action_params=np.zeros(A_TIMES_P, dtype=np.float32),
            reward=0.0,
            next_state=np.zeros(STATE_DIM, dtype=np.float32),
            done=False,
        )
    assert len(buf) == 5
