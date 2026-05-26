"""Stage 8 JointReplayBuffer track_jammer 扩展单测。"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.common.buffers.joint_replay import JointReplayBuffer


def _add_one(buffer, *, with_jp=False, n_agents=4, state_dim=18, n_actions=4, param_dim=2, H=4, C=6):
    kwargs = dict(
        state=np.zeros((n_agents, state_dim), dtype=np.float32),
        action_discrete=np.zeros((n_agents,), dtype=np.int64),
        action_params=np.zeros((n_agents, n_actions * param_dim), dtype=np.float32),
        reward=0.5,
        next_state=np.zeros((n_agents, state_dim), dtype=np.float32),
        done=False,
    )
    if with_jp:
        kwargs["sensing_history"] = np.zeros((n_agents, H, C), dtype=np.float32)
        kwargs["next_sensing_history"] = np.zeros((n_agents, H, C), dtype=np.float32)
        kwargs["jammer_target"] = np.zeros((C,), dtype=np.float32)
    buffer.add(**kwargs)


def test_track_jammer_true_sample_returns_jp_fields_with_baseline_shapes():
    buffer = JointReplayBuffer(capacity=16, per_agent_reward=False, track_jammer=True)
    for _ in range(5):
        _add_one(buffer, with_jp=True)
    batch = buffer.sample(4)
    # baseline layout: jammer_target 是 (B, C) 不是 (B, N, C)
    assert batch["sensing_history"].shape == (4, 4, 4, 6)       # (B, N, H, C)
    assert batch["next_sensing_history"].shape == (4, 4, 4, 6)
    assert batch["jammer_target"].shape == (4, 6)                # (B, C)
    assert batch["sensing_history"].dtype == np.float32
    assert batch["jammer_target"].dtype == np.float32
    # plain 字段不变
    assert batch["state"].shape == (4, 4, 18)
    assert batch["action_discrete"].dtype == np.int64


def test_track_jammer_false_sample_does_not_return_jp_fields():
    buffer = JointReplayBuffer(capacity=16, per_agent_reward=False, track_jammer=False)
    for _ in range(5):
        _add_one(buffer, with_jp=False)
    batch = buffer.sample(4)
    assert "sensing_history" not in batch
    assert "next_sensing_history" not in batch
    assert "jammer_target" not in batch


def test_track_jammer_true_add_without_jp_kwargs_raises():
    buffer = JointReplayBuffer(capacity=16, per_agent_reward=False, track_jammer=True)
    with pytest.raises(ValueError, match="track_jammer=True requires"):
        _add_one(buffer, with_jp=False)


def test_track_jammer_false_add_with_jp_kwargs_raises():
    buffer = JointReplayBuffer(capacity=16, per_agent_reward=False, track_jammer=False)
    with pytest.raises(ValueError, match="track_jammer=False does not accept"):
        _add_one(buffer, with_jp=True)
