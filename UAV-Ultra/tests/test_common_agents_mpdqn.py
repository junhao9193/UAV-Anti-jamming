"""``MPDQNAgent`` 契约（plan locked #5）：仅持网络 + optimizer + target，**无 self.buffer**。"""

from __future__ import annotations

import numpy as np
import torch

from src.algorithms.common.agents import MPDQNAgent


STATE_DIM, N_ACTIONS, PARAM_DIM = 11, 5, 3


def test_mpdqn_agent_has_no_internal_buffer_field():
    """plan locked decision #5：Agent 不持 replay；所有 replay 在 trainer 侧。"""
    agent = MPDQNAgent(STATE_DIM, N_ACTIONS, PARAM_DIM, device="cpu")
    assert not hasattr(agent, "buffer"), "MPDQNAgent must not own a replay buffer (plan #5)"


def test_mpdqn_agent_required_fields_present():
    agent = MPDQNAgent(STATE_DIM, N_ACTIONS, PARAM_DIM, device="cpu")
    for name in ("actor", "q_net", "target_actor", "target_q_net", "actor_opt", "q_opt"):
        assert hasattr(agent, name), f"MPDQNAgent missing required field: {name}"


def test_mpdqn_agent_select_action_returns_int_and_flat_params():
    agent = MPDQNAgent(STATE_DIM, N_ACTIONS, PARAM_DIM, device="cpu")
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action_discrete, params_flat = agent.select_action(state, epsilon=0.0)
    assert isinstance(action_discrete, int)
    assert 0 <= action_discrete < N_ACTIONS
    assert params_flat.shape == (N_ACTIONS * PARAM_DIM,)
    assert params_flat.dtype == np.float32


def test_mpdqn_agent_train_step_from_tensors_returns_losses_dict():
    torch.manual_seed(0)
    agent = MPDQNAgent(STATE_DIM, N_ACTIONS, PARAM_DIM, batch_size=4, device="cpu")
    B = 4
    state = torch.randn(B, STATE_DIM)
    action_discrete = torch.randint(0, N_ACTIONS, (B, 1))
    action_params = torch.rand(B, N_ACTIONS, PARAM_DIM)
    reward = torch.randn(B, 1)
    next_state = torch.randn(B, STATE_DIM)
    done = torch.zeros(B, 1)

    result = agent.train_step_from_tensors(
        state=state,
        action_discrete=action_discrete,
        action_params=action_params,
        reward=reward,
        next_state=next_state,
        done=done,
    )
    assert result is not None
    assert "loss_q" in result and "loss_actor" in result
