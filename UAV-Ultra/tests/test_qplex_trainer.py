"""QPLEX trainer 测试：3 个 hook 的 max_agent_qs 形状 / detach 边界。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms import build_trainer
from src.algorithms.common import registered_names
from src.algorithms.qplex.trainer import QPLEXTrainer
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def cfgs():
    return load_env_config(), load_algo_config("qplex", overrides={"batch_size": 4, "buffer_capacity": 64})


def test_qplex_registered():
    assert "qplex" in registered_names()


def test_qplex_build_trainer_returns_correct_instance(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = build_trainer("qplex", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert isinstance(trainer, QPLEXTrainer)


def test_qplex_train_step_smoke(cfgs):
    """填充 buffer 后 train_step 跑通（覆盖 critic + target + actor 三个 hook）。"""
    env_cfg, algo_cfg = cfgs
    trainer = QPLEXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    n, sdim = trainer.n_agents, trainer.state_dim
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


def test_qplex_collect_critic_extras_shape_and_grad(cfgs):
    """plan 形状约束：``max_agent_qs`` 必须是 (B, N)；critic extras **不** detach。"""
    env_cfg, algo_cfg = cfgs
    trainer = QPLEXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    B = 4
    n, sdim = trainer.n_agents, trainer.state_dim
    total_param = trainer.n_actions * trainer.param_dim
    state = torch.randn(B, n, sdim)
    action_params = torch.rand(B, n, total_param)

    extras = trainer._collect_critic_extras(state, action_params)
    assert "max_agent_qs" in extras
    max_qs = extras["max_agent_qs"]
    assert max_qs.shape == (B, n)
    # critic extras 不 detach（梯度回流 q_net）
    assert max_qs.requires_grad


def test_qplex_collect_actor_extras_is_detached(cfgs):
    """actor extras 必须 detach（advantage baseline 引用，不能共享梯度）。"""
    env_cfg, algo_cfg = cfgs
    trainer = QPLEXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    B = 4
    n = trainer.n_agents

    # 构造 (B, A) tensor，模拟 q_pred_all 列表
    fake_q_pred = [torch.randn(B, trainer.n_actions, requires_grad=True) for _ in range(n)]
    fake_state = torch.randn(B, n, trainer.state_dim)
    fake_params_pred = [torch.randn(B, trainer.n_actions, trainer.param_dim) for _ in range(n)]

    extras = trainer._collect_actor_extras(fake_state, fake_params_pred, fake_q_pred)
    max_qs = extras["max_agent_qs"]
    assert max_qs.shape == (B, n)
    # detach 后 requires_grad 应为 False
    assert not max_qs.requires_grad


def test_qplex_collect_target_extras_shape(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = QPLEXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    B = 4
    next_state = torch.randn(B, trainer.n_agents, trainer.state_dim)
    with torch.no_grad():  # base 在 no_grad 内调用
        extras = trainer._collect_target_extras(next_state)
    assert extras["max_agent_qs"].shape == (B, trainer.n_agents)
