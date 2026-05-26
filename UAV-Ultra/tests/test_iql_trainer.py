"""IQL trainer 测试：注册成功 / 构造 / N 独立 agent / per-agent reward 流向 / train_step 跑通。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms import build_trainer
from src.algorithms.common import registered_names
from src.algorithms.iql.trainer import IQLTrainer
from src.config import specs
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def cfgs():
    return load_env_config(), load_algo_config("iql")


def test_iql_registered_in_global_registry(cfgs):
    """子包 import 后注册表必须包含 iql。"""
    assert "iql" in registered_names()


def test_build_trainer_returns_iql_trainer_instance(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = build_trainer("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert isinstance(trainer, IQLTrainer)


def test_iql_trainer_holds_n_agents_with_required_fields(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert len(trainer.agents) == env_cfg.n_ch
    # plan locked #6：agents 契约保持 actor / q_net / target_q_net 字段
    for agent in trainer.agents:
        for name in ("actor", "q_net", "target_actor", "target_q_net", "actor_opt", "q_opt"):
            assert hasattr(agent, name)
    # plan locked #5：agent 无内部 buffer
    for agent in trainer.agents:
        assert not hasattr(agent, "buffer")


def test_iql_trainer_buffer_per_agent_reward_mode(cfgs):
    """IQL 必须用 per_agent_reward=True 的 joint buffer。"""
    env_cfg, algo_cfg = cfgs
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.buffer.per_agent_reward is True


def test_iql_trainer_train_step_returns_none_when_buffer_empty(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.train_step() is None


def test_iql_trainer_store_and_train_step_smoke(cfgs):
    """填充 buffer 后 train_step 返回 loss dict 且形状契约正确。"""
    env_cfg, algo_cfg = cfgs
    # 小 batch 加快测试
    small_cfg = load_algo_config("iql", overrides={"batch_size": 4, "buffer_capacity": 64})
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=small_cfg, device="cpu")
    n = trainer.n_agents
    sdim = trainer.state_dim
    total_param = trainer.n_actions * trainer.param_dim

    rng = np.random.RandomState(0)
    for _ in range(16):
        states = [rng.randn(sdim).astype(np.float32) for _ in range(n)]
        next_states = [rng.randn(sdim).astype(np.float32) for _ in range(n)]
        actions = [(int(rng.randint(trainer.n_actions)),
                    rng.rand(total_param).astype(np.float32)) for _ in range(n)]
        rewards = rng.randn(n).astype(np.float32)
        trainer.store_transition(states, actions, rewards, next_states, done=False)

    result = trainer.train_step()
    assert result is not None
    assert "loss_q" in result and "loss_actor" in result


def test_iql_trainer_store_transition_rejects_wrong_reward_shape(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    n = trainer.n_agents
    sdim = trainer.state_dim
    total_param = trainer.n_actions * trainer.param_dim
    states = [np.zeros(sdim, dtype=np.float32) for _ in range(n)]
    actions = [(0, np.zeros(total_param, dtype=np.float32)) for _ in range(n)]
    with pytest.raises(ValueError, match="rewards must have shape"):
        trainer.store_transition(states, actions, np.zeros(n + 1, dtype=np.float32), states)


def test_iql_trainer_store_transition_batch_writes_per_agent_reward(cfgs):
    env_cfg, _ = cfgs
    algo_cfg = load_algo_config("iql", overrides={"batch_size": 2, "buffer_capacity": 16})
    trainer = IQLTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    n = trainer.n_agents
    B = 3
    states = np.zeros((B, n, trainer.state_dim), dtype=np.float32)
    next_states = np.zeros_like(states)
    discrete = np.zeros((B, n), dtype=np.int64)
    params = np.zeros((B, n, trainer.n_actions * trainer.param_dim), dtype=np.float32)
    rewards = np.random.randn(B, n).astype(np.float32)
    dones = np.zeros((B,), dtype=np.float32)

    trainer.store_transition_batch(
        states=states,
        action_discrete=discrete,
        action_params=params,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
    assert len(trainer.buffer) == B
    sample = trainer.buffer.sample(B)
    assert sample["reward"].shape == (B, n)  # IQL 模式必须保留 (B, N)
