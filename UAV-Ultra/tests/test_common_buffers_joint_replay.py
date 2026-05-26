"""``JointReplayBuffer``：``per_agent_reward`` 两种模式 reward shape 区分。

合并自 baseline `iql/joint_replay_buffer.py`（per-agent vector reward）
与 `qmix/joint_replay_buffer.py`（global scalar reward）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.common.buffers import JointReplayBuffer


N, STATE_DIM, A_TIMES_P = 4, 11, 15


def _make_transition(per_agent_reward: bool):
    state = np.random.randn(N, STATE_DIM).astype(np.float32)
    next_state = np.random.randn(N, STATE_DIM).astype(np.float32)
    action_discrete = np.random.randint(0, 5, size=(N,), dtype=np.int64)
    action_params = np.random.rand(N, A_TIMES_P).astype(np.float32)
    if per_agent_reward:
        reward = np.random.randn(N).astype(np.float32)
    else:
        reward = float(np.random.randn())
    return state, action_discrete, action_params, reward, next_state, False


def test_global_reward_mode_returns_1d_reward():
    """QMIX/VDN/QPLEX 模式：reward shape `(B,)`。"""
    np.random.seed(0)
    buf = JointReplayBuffer(capacity=100, per_agent_reward=False)
    for _ in range(20):
        s, a, p, r, ns, d = _make_transition(per_agent_reward=False)
        buf.add(state=s, action_discrete=a, action_params=p, reward=r, next_state=ns, done=d)

    batch = buf.sample(8)
    assert batch["state"].shape == (8, N, STATE_DIM)
    assert batch["action_discrete"].shape == (8, N)
    assert batch["action_params"].shape == (8, N, A_TIMES_P)
    assert batch["reward"].shape == (8,)
    assert batch["reward"].dtype == np.float32


def test_per_agent_reward_mode_returns_2d_reward():
    """IQL 模式：reward shape `(B, N)`。"""
    np.random.seed(0)
    buf = JointReplayBuffer(capacity=100, per_agent_reward=True)
    for _ in range(20):
        s, a, p, r, ns, d = _make_transition(per_agent_reward=True)
        buf.add(state=s, action_discrete=a, action_params=p, reward=r, next_state=ns, done=d)

    batch = buf.sample(8)
    assert batch["reward"].shape == (8, N)
    assert batch["reward"].dtype == np.float32


def test_per_agent_reward_mode_rejects_scalar_reward():
    buf = JointReplayBuffer(capacity=100, per_agent_reward=True)
    s, a, p, _, ns, d = _make_transition(per_agent_reward=False)
    with pytest.raises(ValueError, match="per_agent_reward=True"):
        buf.add(state=s, action_discrete=a, action_params=p, reward=0.5, next_state=ns, done=d)
