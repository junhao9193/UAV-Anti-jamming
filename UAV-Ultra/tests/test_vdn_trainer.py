"""VDN trainer 测试：注册成功 / 无可学参 mixer / train_step 跑通。"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import build_trainer
from src.algorithms.common import registered_names
from src.algorithms.vdn.trainer import VDNTrainer
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def cfgs():
    return load_env_config(), load_algo_config("vdn", overrides={"batch_size": 4, "buffer_capacity": 64})


def test_vdn_registered():
    assert "vdn" in registered_names()


def test_vdn_build_trainer_returns_correct_instance(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = build_trainer("vdn", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert isinstance(trainer, VDNTrainer)


def test_vdn_mixer_has_no_parameters_and_no_mixer_opt(cfgs):
    """plan：VDN sum mixer 无可学参，mixer_opt 必为 None。"""
    env_cfg, algo_cfg = cfgs
    trainer = VDNTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.mixer_opt is None
    assert list(trainer.mixer.parameters()) == []


def test_vdn_buffer_global_reward_mode(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = VDNTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.buffer.per_agent_reward is False


def test_vdn_train_step_smoke(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = VDNTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
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
