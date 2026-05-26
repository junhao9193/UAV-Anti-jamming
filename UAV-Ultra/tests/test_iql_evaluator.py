"""IQL evaluator：每 agent greedy MP-DQN，无 epsilon 探索。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms import build_evaluator, build_trainer
from src.algorithms.iql.evaluator import IQLEvalPolicy
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def env_and_algo_cfg():
    return load_env_config(), load_algo_config("iql")


def test_build_evaluator_returns_iql_eval_policy(env_and_algo_cfg):
    env_cfg, algo_cfg = env_and_algo_cfg
    trainer = build_trainer("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    evaluator = build_evaluator(
        "iql", env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer
    )
    assert isinstance(evaluator, IQLEvalPolicy)


def test_iql_evaluator_requires_trainer(env_and_algo_cfg):
    env_cfg, algo_cfg = env_and_algo_cfg
    with pytest.raises(ValueError, match="requires trainer"):
        build_evaluator("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=None)


def test_iql_evaluator_select_actions_deterministic(env_and_algo_cfg):
    """Greedy 同 trainer + 同 state → 同 action。"""
    env_cfg, algo_cfg = env_and_algo_cfg
    torch.manual_seed(0)
    trainer = build_trainer("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    evaluator = IQLEvalPolicy(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    states = [np.random.RandomState(i).randn(trainer.state_dim).astype(np.float32)
              for i in range(trainer.n_agents)]

    a1 = evaluator.select_actions(states)
    a2 = evaluator.select_actions(states)
    assert len(a1) == trainer.n_agents
    for (d1, p1), (d2, p2) in zip(a1, a2):
        assert d1 == d2
        np.testing.assert_array_equal(p1, p2)


def test_iql_evaluator_returns_correct_action_shapes(env_and_algo_cfg):
    env_cfg, algo_cfg = env_and_algo_cfg
    trainer = build_trainer("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    evaluator = IQLEvalPolicy(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    states = [np.zeros(trainer.state_dim, dtype=np.float32) for _ in range(trainer.n_agents)]
    actions = evaluator.select_actions(states)
    total_param_dim = trainer.n_actions * trainer.param_dim
    for discrete, params in actions:
        assert isinstance(discrete, int)
        assert 0 <= discrete < trainer.n_actions
        assert params.shape == (total_param_dim,)
        assert params.dtype == np.float32


def test_iql_evaluator_rejects_wrong_states_length(env_and_algo_cfg):
    env_cfg, algo_cfg = env_and_algo_cfg
    trainer = build_trainer("iql", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    evaluator = IQLEvalPolicy(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    with pytest.raises(ValueError, match="Expected"):
        evaluator.select_actions([np.zeros(trainer.state_dim, dtype=np.float32)])
