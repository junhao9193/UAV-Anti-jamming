"""QMIX trainer 测试：注册成功 / 可学习 mixer / train_step 跑通。"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import build_trainer
from src.algorithms.common import registered_names
from src.algorithms.qmix.trainer import QMIXTrainer
from src.config.loader import load_algo_config, load_env_config


@pytest.fixture()
def cfgs():
    return load_env_config(), load_algo_config("qmix", overrides={"batch_size": 4, "buffer_capacity": 64})


def test_qmix_registered():
    assert "qmix" in registered_names()


def test_qmix_build_trainer_returns_correct_instance(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert isinstance(trainer, QMIXTrainer)


def test_qmix_mixer_has_parameters_and_mixer_opt_set(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = QMIXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    assert trainer.mixer_opt is not None
    assert len(list(trainer.mixer.parameters())) > 0


def test_qmix_lr_mixer_settled_from_config(cfgs):
    """loader 把 lr_mixer=null 落定为 lr_q；trainer 看到的应是 float。"""
    env_cfg, algo_cfg = cfgs
    trainer = QMIXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    # 由于默认 lr_mixer: null 在 loader 落定为 lr_q
    for group in trainer.mixer_opt.param_groups:
        assert group["lr"] == float(algo_cfg.lr_q)


def test_qmix_train_step_smoke(cfgs):
    env_cfg, algo_cfg = cfgs
    trainer = QMIXTrainer(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
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
