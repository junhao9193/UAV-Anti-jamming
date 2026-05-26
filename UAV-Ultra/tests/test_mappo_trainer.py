"""MAPPO trainer 测试：构造 / actor+critic shape / act / update with rollout batch。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms import build_trainer
from src.algorithms.common import registered_names
from src.algorithms.common.buffers import RolloutBatch
from src.algorithms.mappo.agent import ActResult
from src.algorithms.mappo.trainer import MAPPOTrainer
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def cfgs():
    return load_env_config(), load_algo_config(
        "mappo",
        overrides={"update_epochs": 2, "minibatch_size": 8},
    )


def test_mappo_registered():
    assert "mappo" in registered_names()


def test_mappo_build_trainer_returns_correct_instance(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = build_trainer("mappo", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert isinstance(trainer, MAPPOTrainer)


def test_mappo_trainer_dimensions(cfgs):
    """obs_dim/n_actions/cont_dim 来自 specs；不读取 policy mobility 额外维。"""
    env_cfg, algo_cfg = cfgs
    trainer = MAPPOTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.obs_dim == 18  # baseline default state_dim
    assert trainer.n_agents == 4
    # cont_dim = n_des = param_dim_per_action
    assert trainer.cont_dim == 2


def test_mappo_hyperparameters_from_config_not_hardcoded(cfgs):
    """plan locked decision #9：PPO 超参全部从 algo_cfg 取。"""
    env_cfg, algo_cfg = cfgs
    trainer = MAPPOTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.update_epochs == 2
    assert trainer.minibatch_size == 8
    assert trainer.gae_lambda == algo_cfg.gae_lambda
    assert trainer.clip_range == algo_cfg.clip_range
    assert trainer.ent_coef == algo_cfg.ent_coef
    assert trainer.vf_coef == algo_cfg.vf_coef
    assert trainer.max_grad_norm == algo_cfg.max_grad_norm


def test_mappo_train_step_returns_none(cfgs):
    """MAPPO on-policy：trainer 不持 rollout buffer；train_step 始终返回 None。"""
    env_cfg, algo_cfg = cfgs
    trainer = MAPPOTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.train_step() is None


def test_mappo_act_returns_act_result(cfgs):
    env_cfg, algo_cfg = cfgs
    torch.manual_seed(0)
    trainer = MAPPOTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    obs = np.random.randn(trainer.obs_dim).astype(np.float32)
    gs = np.random.randn(trainer.global_state_dim).astype(np.float32)
    result = trainer.act(obs=obs, global_state=gs, agent_id=0, deterministic=False)
    assert isinstance(result, ActResult)
    assert 0 <= result.action_discrete < trainer.n_actions
    assert result.action_cont.shape == (trainer.cont_dim,)
    assert (result.action_cont >= 0.0).all() and (result.action_cont <= 1.0).all()


def test_mappo_update_runs_with_synthetic_rollout_batch(cfgs):
    """喂入合成 RolloutBatch，update 应返回 loss_pi / loss_v / entropy。"""
    env_cfg, algo_cfg = cfgs
    torch.manual_seed(0)
    np.random.seed(0)
    trainer = MAPPOTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")

    T = 16
    n = T * trainer.n_agents
    obs = np.random.randn(n, trainer.obs_dim).astype(np.float32)
    gs = np.random.randn(n, trainer.global_state_dim).astype(np.float32)
    agent_id = np.tile(np.arange(trainer.n_agents, dtype=np.int64), T)
    action_discrete = np.random.randint(0, trainer.n_actions, size=n).astype(np.int64)
    action_cont = np.random.rand(n, trainer.cont_dim).astype(np.float32) * 0.5 + 0.25
    old_log_prob = np.random.randn(n).astype(np.float32) * 0.1
    returns = np.random.randn(n).astype(np.float32)
    advantages = np.random.randn(n).astype(np.float32)

    batch = RolloutBatch(
        obs=obs,
        global_state=gs,
        agent_id=agent_id,
        action_discrete=action_discrete,
        action_cont=action_cont,
        old_log_prob=old_log_prob,
        returns=returns,
        advantages=advantages,
    )
    losses = trainer.update(batch)
    assert "loss_pi" in losses and "loss_v" in losses and "entropy" in losses
    assert np.isfinite(losses["loss_pi"])
    assert np.isfinite(losses["loss_v"])


def test_mappo_evaluator_select_actions(cfgs):
    """deterministic act 应在每个 agent 上返回 (discrete, params_flat)；params_flat 在 chosen 槽填值。"""
    from src.algorithms import build_evaluator

    env_cfg, algo_cfg = cfgs
    torch.manual_seed(0)
    trainer = build_trainer("mappo", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    evaluator = build_evaluator("mappo", env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    states = [np.random.RandomState(i).randn(trainer.obs_dim).astype(np.float32)
              for i in range(trainer.n_agents)]
    actions = evaluator.select_actions(states)
    assert len(actions) == trainer.n_agents
    total_param_dim = trainer.n_actions * trainer.cont_dim
    for discrete, params in actions:
        assert 0 <= discrete < trainer.n_actions
        assert params.shape == (total_param_dim,)
        # chosen action 槽位非零
        start = discrete * trainer.cont_dim
        assert (params[start : start + trainer.cont_dim] > 0.0).any()
