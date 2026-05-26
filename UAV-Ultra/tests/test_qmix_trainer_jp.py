"""Stage 8 QMIXTrainer JP-aware 构造 + critic/target sync 行为单测。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.algorithms import build_trainer
from src.algorithms.common.agents.jammer_aware_mpdqn_agent import JammerAwareMPDQNAgent
from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent
from src.config.loader import load_algo_config, load_env_config
from src.training.callbacks.base import TrainHookContext
from src.training.callbacks.jammer_prediction import JammerPredictionCallback


def _build(callbacks: list[str]):
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={
            "callbacks": callbacks,
            "batch_size": 4,
            "buffer_capacity": 16,
            "device": "cpu",
        },
        env_cfg=env_cfg,
    )
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    return env_cfg, algo_cfg, trainer


def test_qmix_trainer_jp_off_uses_plain_mpdqn_agent_and_buffer():
    env_cfg, _, trainer = _build([])
    assert all(type(a) is MPDQNAgent for a in trainer.agents)
    assert trainer.buffer.track_jammer is False
    # state_dim 仍 raw（Stage 4 行为字段级一致）
    assert trainer.state_dim == int(env_cfg.n_ch and trainer.agents[0].state_dim)
    # mixer global_state_dim 是 n_agents * raw_state_dim
    assert trainer.global_state_dim == int(env_cfg.n_ch * trainer.state_dim)


def test_qmix_trainer_jp_on_uses_jammer_aware_agent_and_track_jammer_buffer():
    env_cfg, _, trainer = _build(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    assert all(isinstance(a, JammerAwareMPDQNAgent) for a in trainer.agents)
    assert trainer.buffer.track_jammer is True
    # ★ 关键：agent.state_dim 仍 raw（baseline 同款）
    assert trainer.agents[0].state_dim == trainer.state_dim
    # mixer global_state_dim 仍是 raw n_agents*state_dim
    assert trainer.global_state_dim == int(env_cfg.n_ch * trainer.state_dim)
    # actor / q_net 内部第一层 in_features = raw + n_channel
    assert trainer.agents[0].actor.net[0].in_features == trainer.state_dim + int(env_cfg.n_channel)
    assert (
        trainer.agents[0].q_net.state_encoder[0].in_features
        == trainer.state_dim + int(env_cfg.n_channel)
    )


def test_qmix_target_sync_includes_target_jammer_predictor():
    """_target_sync 应在 cadence 命中时把 jammer_predictor 同步给 target_jammer_predictor。"""
    _, _, trainer = _build(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    # 故意把 jammer_predictor 改一下让两者不一致
    with torch.no_grad():
        first_param = next(trainer.agents[0].jammer_predictor.parameters())
        first_param.add_(1.0)
    # learn_steps 在 trainer 内部计数；强制走一次 sync
    trainer.learn_steps = int(trainer.target_update_interval) - 1
    trainer._target_sync()
    # target 应等于 online
    online_sd = trainer.agents[0].jammer_predictor.state_dict()
    target_sd = trainer.agents[0].target_jammer_predictor.state_dict()
    for key in online_sd:
        torch.testing.assert_close(target_sd[key], online_sd[key], rtol=0.0, atol=1e-12)


def test_qmix_jp_train_step_reports_loss_jammer():
    """JP aux loss 应随 train result 透出，便于对齐 baseline loss_jammer 曲线。"""
    env_cfg, algo_cfg, trainer = _build(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"]
    )
    callback = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    callback.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)

    batch_size = int(algo_cfg.batch_size)
    generator = torch.Generator().manual_seed(123)
    batch = {
        "state": torch.randn(batch_size, trainer.n_agents, trainer.state_dim, generator=generator),
        "action_discrete": torch.randint(
            0, trainer.n_actions, (batch_size, trainer.n_agents), generator=generator
        ),
        "action_params": torch.randn(
            batch_size,
            trainer.n_agents,
            trainer.n_actions * trainer.param_dim,
            generator=generator,
        ),
        "reward": torch.randn(batch_size, 1, generator=generator),
        "next_state": torch.randn(batch_size, trainer.n_agents, trainer.state_dim, generator=generator),
        "done": torch.zeros(batch_size, 1),
        "sensing_history": torch.randn(
            batch_size,
            trainer.n_agents,
            int(algo_cfg.jammer_history_len),
            int(env_cfg.n_channel),
            generator=generator,
        ),
        "next_sensing_history": torch.randn(
            batch_size,
            trainer.n_agents,
            int(algo_cfg.jammer_history_len),
            int(env_cfg.n_channel),
            generator=generator,
        ),
        "jammer_target": torch.randint(
            0, 2, (batch_size, int(env_cfg.n_channel)), generator=generator
        ).float(),
    }
    trainer._aux_loss_fns = [callback.on_aux_loss]
    try:
        result = trainer.train_step_from_batch(
            batch,
            hook_context=TrainHookContext(trainer=trainer, episode=0, step=0),
        )
    finally:
        trainer._aux_loss_fns = []

    assert result is not None
    assert "loss_jammer" in result
    assert np.isfinite(result["loss_jammer"])


def test_qmix_jp_store_transition_requires_and_accepts_jp_fields():
    env_cfg, algo_cfg, trainer = _build(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"]
    )
    states = [np.zeros((trainer.state_dim,), dtype=np.float32) for _ in range(trainer.n_agents)]
    next_states = [np.ones((trainer.state_dim,), dtype=np.float32) for _ in range(trainer.n_agents)]
    actions = [
        (0, np.zeros((trainer.n_actions * trainer.param_dim,), dtype=np.float32))
        for _ in range(trainer.n_agents)
    ]
    rewards = np.zeros((trainer.n_agents,), dtype=np.float32)

    with pytest.raises(ValueError, match="JP-aware store_transition requires"):
        trainer.store_transition(states, actions, rewards, next_states, done=False)

    history = np.zeros(
        (trainer.n_agents, int(algo_cfg.jammer_history_len), int(env_cfg.n_channel)),
        dtype=np.float32,
    )
    trainer.store_transition(
        states,
        actions,
        rewards,
        next_states,
        done=False,
        sensing_history=history,
        next_sensing_history=history,
        jammer_target=np.zeros((int(env_cfg.n_channel),), dtype=np.float32),
    )
    assert len(trainer.buffer) == 1
