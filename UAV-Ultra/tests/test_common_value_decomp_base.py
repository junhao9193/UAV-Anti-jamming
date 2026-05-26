"""``ValueDecompTrainerBase`` 在 stub mixer 下完整 ``train_step`` 跑通。

验证：
- 基类生效（子类只需实现 ``_build_mixer``，hooks 默认空 OK）；
- ``JointReplayBuffer(per_agent_reward=False)`` 形状契约；
- ``train_step`` 在数据不足时返回 None；数据足够时返回 ``{"loss_q", "loss_actor"}``。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.algorithms.common.value_decomp import TDTargetContext, ValueDecompTrainerBase


N_AGENTS, STATE_DIM, N_ACTIONS, PARAM_DIM = 2, 7, 3, 2
GLOBAL_STATE_DIM = N_AGENTS * STATE_DIM


class _StubLinearMixer(nn.Module):
    """简单 mixer：Q_tot = sum(qs) + linear(global_state)。带可学参以验 mixer_opt 启用。"""

    def __init__(self, n_agents: int, global_state_dim: int):
        super().__init__()
        self.n_agents = int(n_agents)
        self.proj = nn.Linear(global_state_dim, 1)

    def forward(self, agent_qs, global_state):
        if agent_qs.dim() == 3:
            agent_qs = agent_qs.squeeze(-1)
        return agent_qs.sum(dim=1, keepdim=True) + self.proj(global_state)


class _StubTrainer(ValueDecompTrainerBase):
    def _build_mixer(self):
        return _StubLinearMixer(self.n_agents, self.global_state_dim)


def test_value_decomp_base_train_step_skips_when_buffer_empty():
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        batch_size=8,
        device="cpu",
    )
    assert trainer.train_step() is None


def test_value_decomp_base_train_step_runs_with_synthetic_batch():
    np.random.seed(0)
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        buffer_capacity=64,
        batch_size=4,
        device="cpu",
    )

    # 填充 buffer
    for _ in range(20):
        states = [np.random.randn(STATE_DIM).astype(np.float32) for _ in range(N_AGENTS)]
        actions = [
            (int(np.random.randint(N_ACTIONS)), np.random.rand(N_ACTIONS * PARAM_DIM).astype(np.float32))
            for _ in range(N_AGENTS)
        ]
        rewards = np.random.randn(N_AGENTS).astype(np.float32)
        next_states = [np.random.randn(STATE_DIM).astype(np.float32) for _ in range(N_AGENTS)]
        trainer.store_transition(states, actions, rewards, next_states, done=False)

    result = trainer.train_step()
    assert result is not None
    assert "loss_q" in result
    assert "loss_actor" in result


def test_value_decomp_base_exposes_stage4_concrete_methods():
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        device="cpu",
    )
    for name in (
        "_critic_step",
        "_actor_step",
        "_target_sync",
        "select_actions",
        "store_transition",
        "store_transition_batch",
    ):
        assert callable(getattr(trainer, name))


def test_value_decomp_base_store_transition_batch_adds_global_rewards():
    rng = np.random.RandomState(123)
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        buffer_capacity=64,
        batch_size=4,
        device="cpu",
    )
    batch_size = 5
    states = rng.randn(batch_size, N_AGENTS, STATE_DIM).astype(np.float32)
    next_states = rng.randn(batch_size, N_AGENTS, STATE_DIM).astype(np.float32)
    action_discrete = rng.randint(0, N_ACTIONS, size=(batch_size, N_AGENTS)).astype(np.int64)
    action_params = rng.rand(batch_size, N_AGENTS, N_ACTIONS * PARAM_DIM).astype(np.float32)
    rewards = rng.randn(batch_size, N_AGENTS).astype(np.float32)
    dones = np.zeros((batch_size,), dtype=np.float32)

    trainer.store_transition_batch(
        states=states,
        action_discrete=action_discrete,
        action_params=action_params,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    assert len(trainer.buffer) == batch_size
    sample = trainer.buffer.sample(batch_size)
    assert sample["state"].shape == (batch_size, N_AGENTS, STATE_DIM)
    assert sample["action_params"].shape == (batch_size, N_AGENTS, N_ACTIONS * PARAM_DIM)
    assert sample["reward"].shape == (batch_size,)


def test_value_decomp_base_holds_n_agents_mpdqn_agents():
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        device="cpu",
    )
    assert len(trainer.agents) == N_AGENTS
    for agent in trainer.agents:
        for name in ("actor", "q_net", "target_actor", "target_q_net", "actor_opt", "q_opt"):
            assert hasattr(agent, name)


def test_compute_td_target_plain_context_keeps_stage4_formula():
    trainer = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        gamma=0.5,
        value_target_clip=10.0,
        device="cpu",
    )
    reward = torch.tensor([[1.0], [2.0]])
    done = torch.tensor([[0.0], [1.0]])
    next_q_tot = torch.tensor([[4.0], [8.0]])

    plain = trainer._compute_td_target(reward, done, next_q_tot, target_context=None)
    with_context = trainer._compute_td_target(
        reward,
        done,
        next_q_tot,
        target_context=TDTargetContext(alpha_model=0.0),
    )

    expected = torch.tensor([[3.0], [2.0]])
    torch.testing.assert_close(plain, expected)
    torch.testing.assert_close(with_context, expected)


def test_train_step_from_batch_matches_train_step_on_same_sample():
    np.random.seed(0)
    torch.manual_seed(0)
    trainer_a = _StubTrainer(
        n_agents=N_AGENTS,
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        param_dim=PARAM_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        buffer_capacity=64,
        batch_size=4,
        target_update_interval=999,
        device="cpu",
    )
    trainer_b = copy.deepcopy(trainer_a)

    for trainer in (trainer_a, trainer_b):
        np.random.seed(123)
        for _ in range(12):
            states = [np.random.randn(STATE_DIM).astype(np.float32) for _ in range(N_AGENTS)]
            actions = [
                (int(np.random.randint(N_ACTIONS)), np.random.rand(N_ACTIONS * PARAM_DIM).astype(np.float32))
                for _ in range(N_AGENTS)
            ]
            rewards = np.random.randn(N_AGENTS).astype(np.float32)
            next_states = [np.random.randn(STATE_DIM).astype(np.float32) for _ in range(N_AGENTS)]
            trainer.store_transition(states, actions, rewards, next_states, done=False)

    np.random.seed(777)
    batch_np = trainer_a.buffer.sample(trainer_a.batch_size)
    batch_t = {
        "state": torch.from_numpy(batch_np["state"]).to(trainer_a.device),
        "action_discrete": torch.from_numpy(batch_np["action_discrete"]).long().to(trainer_a.device),
        "action_params": torch.from_numpy(batch_np["action_params"]).to(trainer_a.device),
        "reward": torch.from_numpy(batch_np["reward"]).to(trainer_a.device).view(-1, 1),
        "next_state": torch.from_numpy(batch_np["next_state"]).to(trainer_a.device),
        "done": torch.from_numpy(batch_np["done"]).to(trainer_a.device).view(-1, 1),
    }
    result_a = trainer_a.train_step_from_batch(batch_t, target_context=None)

    trainer_b.buffer.sample = lambda _batch_size: batch_np  # type: ignore[method-assign]
    result_b = trainer_b.train_step()

    assert result_a is not None
    assert result_b is not None
    assert result_a["loss_q"] == result_b["loss_q"]
    assert result_a["loss_actor"] == result_b["loss_actor"]
