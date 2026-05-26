"""``QMIXValueTeacher`` 接口：greedy_action / q_tot_target shape + 与 baseline 等价。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms.qmix.trainer import QMIXTrainer
from src.algorithms.qmix.world_model_adapter import MPDQNQMIXDims, QMIXValueTeacher
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def trainer_and_dims():
    env_cfg = load_env_config()
    algo_cfg = load_algo_config("qmix", overrides={"batch_size": 4, "buffer_capacity": 16})
    trainer = QMIXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    dims = MPDQNQMIXDims(
        n_agents=trainer.n_agents,
        agent_state_dim=trainer.state_dim,
        n_actions=trainer.n_actions,
        param_dim=trainer.param_dim,
    )
    return trainer, dims


def test_value_teacher_freezes_target_networks(trainer_and_dims):
    """target_q_net + target_mixer 应在 adapter 构造后被 freeze。"""
    trainer, dims = trainer_and_dims
    QMIXValueTeacher(trainer, dims)
    for agent in trainer.agents:
        for p in agent.target_q_net.parameters():
            assert not p.requires_grad
    for p in trainer.target_mixer.parameters():
        assert not p.requires_grad


def test_value_teacher_greedy_action_shapes(trainer_and_dims):
    trainer, dims = trainer_and_dims
    teacher = QMIXValueTeacher(trainer, dims)
    B = 5
    global_state = torch.randn(B, dims.n_agents * dims.agent_state_dim)
    a_disc, a_params = teacher.greedy_action(global_state)
    assert a_disc.shape == (B, dims.n_agents)
    assert a_disc.dtype == torch.long
    assert a_params.shape == (B, dims.n_agents, dims.n_actions * dims.param_dim)
    assert a_params.dtype == torch.float32


def test_value_teacher_q_tot_target_shape(trainer_and_dims):
    trainer, dims = trainer_and_dims
    teacher = QMIXValueTeacher(trainer, dims)
    B = 5
    global_state = torch.randn(B, dims.n_agents * dims.agent_state_dim)
    a_disc = torch.zeros(B, dims.n_agents, dtype=torch.long)
    a_params = torch.rand(B, dims.n_agents, dims.n_actions * dims.param_dim)
    q_tot = teacher.q_tot_target(global_state, a_disc, a_params)
    assert q_tot.shape == (B, 1)
    assert q_tot.dtype == torch.float32


def test_value_teacher_matches_baseline_q_tot_target(baseline_import):
    """copy 新 trainer 的所有 weight 到 baseline QMIX trainer，对比 q_tot_target 输出。"""
    baseline_adapt_mod = baseline_import("algorithms.world_model.qmix_adapters")
    baseline_qmix_mod = baseline_import("algorithms.mpdqn.qmix.trainer_greedy_actor")

    env_cfg = load_env_config()
    algo_cfg = load_algo_config("qmix", overrides={"batch_size": 4, "buffer_capacity": 16})
    new_trainer = QMIXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")

    old_trainer = baseline_qmix_mod.MPDQNQMIXTrainer(
        n_agents=new_trainer.n_agents,
        state_dim=new_trainer.state_dim,
        n_actions=new_trainer.n_actions,
        param_dim=new_trainer.param_dim,
        global_state_dim=new_trainer.global_state_dim,
        buffer_capacity=16,
        batch_size=4,
        gamma=float(algo_cfg.gamma),
        lr_actor=float(algo_cfg.lr_actor),
        lr_q=float(algo_cfg.lr_q),
        lr_mixer=float(algo_cfg.lr_mixer),
        target_update_interval=int(algo_cfg.target_update_interval),
        mixing_hidden_dim=int(algo_cfg.mixing_hidden_dim),
        hypernet_hidden_dim=int(algo_cfg.hypernet_hidden_dim),
        use_amp=False,
        max_grad_norm=float(algo_cfg.max_grad_norm),
        value_target_clip=float(algo_cfg.value_target_clip),
        device="cpu",
    )
    # copy weights: new → old
    for old_agent, new_agent in zip(old_trainer.agents, new_trainer.agents):
        old_agent.actor.load_state_dict(new_agent.actor.state_dict())
        old_agent.q_net.load_state_dict(new_agent.q_net.state_dict())
        old_agent.target_actor.load_state_dict(new_agent.target_actor.state_dict())
        old_agent.target_q_net.load_state_dict(new_agent.target_q_net.state_dict())
    old_trainer.mixer.load_state_dict(new_trainer.mixer.state_dict())
    old_trainer.target_mixer.load_state_dict(new_trainer.target_mixer.state_dict())

    dims = MPDQNQMIXDims(
        n_agents=new_trainer.n_agents,
        agent_state_dim=new_trainer.state_dim,
        n_actions=new_trainer.n_actions,
        param_dim=new_trainer.param_dim,
    )
    new_teacher = QMIXValueTeacher(new_trainer, dims)
    old_teacher = baseline_adapt_mod.MPDQNQMIXValueTeacher(
        old_trainer,
        baseline_adapt_mod.MPDQNQMIXDims(
            n_agents=dims.n_agents,
            agent_state_dim=dims.agent_state_dim,
            n_actions=dims.n_actions,
            param_dim=dims.param_dim,
        ),
    )

    torch.manual_seed(0)
    B = 5
    global_state = torch.randn(B, dims.n_agents * dims.agent_state_dim)
    a_disc = torch.zeros(B, dims.n_agents, dtype=torch.long)
    a_params = torch.rand(B, dims.n_agents, dims.n_actions * dims.param_dim)

    with torch.no_grad():
        q_old = old_teacher.q_tot_target(global_state, a_disc, a_params)
        q_new = new_teacher.q_tot_target(global_state, a_disc, a_params)
    torch.testing.assert_close(q_new, q_old, rtol=0, atol=1e-7)
