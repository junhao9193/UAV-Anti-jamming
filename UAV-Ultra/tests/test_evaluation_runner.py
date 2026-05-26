from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.algorithms import build_trainer
from src.config import specs
from src.config.loader import load_algo_config, load_env_config
from src.evaluation.runner import _run_eval_loop, run_evaluation
from src.training.callbacks import CallbackManager
from src.training.callbacks.policy_mobility import PolicyMobilityCallback
from src.training.checkpoint import load_trainer_state_dict, save_checkpoint
from src.training.runner import run_training
from src.training.vec_env import SubprocVecEnv, _spawn_worker_seeds, make_fixed_p_trans
from src.envs import Environ


def _tiny_dqn_overrides(seed: int = 11) -> dict:
    return {
        "n_episode": 1,
        "n_steps": 2,
        "num_envs": 1,
        "batch_size": 2,
        "buffer_capacity": 8,
        "learn_every": 1,
        "updates_per_learn": 1,
        "seed": seed,
        "device": "cpu",
        "start_method": "fork",
    }


def _tiny_mappo_overrides(seed: int = 12) -> dict:
    return {
        "n_episode": 1,
        "n_steps": 2,
        "seed": seed,
        "device": "cpu",
        "minibatch_size": 8,
    }


def test_spawn_worker_seeds_uses_int_seedsequence_outputs():
    assert _spawn_worker_seeds(None, 3) == [None, None, None]
    seeds = _spawn_worker_seeds(0, 8)
    assert len(seeds) == 8
    assert all(type(seed) is int for seed in seeds)
    assert len(set(seeds)) == len(seeds)
    assert _spawn_worker_seeds(123, 2)[0] != _spawn_worker_seeds(1123, 2)[1]


def test_heuristic_evaluation_smoke_and_eval_schema(tmp_path):
    result = run_evaluation(
        "heuristic",
        episodes=1,
        steps=2,
        num_envs=1,
        seed=5,
        start_method="fork",
        no_save=False,
        output_root=tmp_path,
    )

    assert result.trainer is None
    assert result.output_dir is not None
    assert result.output_dir.name.startswith("heuristic_greedy_sensing_quality_adaptive_")
    json_path = result.output_dir / "evaluation_data.json"
    npz_path = result.output_dir / "evaluation_data.npz"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"algorithm", "timestamp", "config", "metrics"}
    assert data["config"]["artifact_kind"] == "eval"
    assert data["config"]["mode"] == "heuristic"
    assert data["config"]["evaluation_only"] is True
    assert data["config"]["policy_name"] == "greedy_sensing"
    assert data["config"]["requested_power_mode"] == "quality_adaptive"
    assert data["config"]["power_mode"] == "quality_adaptive"
    assert set(data["metrics"]) == {"reward", "success_rate", "energy", "jump"}
    npz = np.load(npz_path)
    assert set(npz.files) == {"reward", "success_rate", "energy", "jump"}
    assert npz["reward"].dtype == np.float32


def test_qmix_and_mappo_evaluation_reload_smoke(tmp_path):
    qmix_result = run_training(
        "qmix",
        algo_overrides=_tiny_dqn_overrides(seed=21),
        no_save=False,
        output_root=tmp_path,
    )
    qmix_eval = run_evaluation(
        "qmix",
        checkpoint_path=qmix_result.output_dir / "qmix_weights.pth",
        episodes=1,
        steps=2,
        num_envs=1,
        seed=21,
        device="cpu",
        start_method="fork",
        no_save=True,
    )
    assert qmix_eval.trainer is not None
    assert len(qmix_eval.metrics["reward"]) == 1

    mappo_result = run_training(
        "mappo",
        algo_overrides=_tiny_mappo_overrides(seed=22),
        no_save=False,
        output_root=tmp_path,
    )
    mappo_eval = run_evaluation(
        "mappo",
        checkpoint_path=mappo_result.output_dir / "mappo_weights.pth",
        episodes=1,
        steps=2,
        num_envs=1,
        seed=22,
        device="cpu",
        start_method="fork",
        no_save=True,
    )
    assert mappo_eval.trainer is not None
    assert len(mappo_eval.metrics["reward"]) == 1


def test_qmix_evaluation_callback_overrides_reload_happy_path(tmp_path):
    callbacks = ["value_expansion", "wm_concurrent"]
    qmix_result = run_training(
        "qmix",
        algo_overrides={**_tiny_dqn_overrides(seed=24), "callbacks": callbacks},
        no_save=False,
        output_root=tmp_path,
    )

    qmix_eval = run_evaluation(
        "qmix",
        checkpoint_path=qmix_result.output_dir / "qmix_weights.pth",
        episodes=1,
        steps=2,
        num_envs=1,
        seed=24,
        device="cpu",
        start_method="fork",
        callback_overrides=callbacks,
        no_save=False,
        output_root=tmp_path,
    )

    assert qmix_eval.callback_states is not None
    assert set(qmix_eval.callback_states) == set(callbacks)
    assert qmix_eval.output_dir is not None
    data = json.loads((qmix_eval.output_dir / "evaluation_data.json").read_text(encoding="utf-8"))
    assert data["config"]["mode"] == "qmix"
    assert data["config"]["evaluation_only"] is True
    assert data["config"]["source_algorithm"] == "qmix"
    assert data["config"]["callbacks"] == callbacks
    assert data["config"]["weights"].endswith("qmix_weights.pth")


def test_eval_loop_applies_policy_mobility_zero_delta():
    env_cfg = load_env_config(
        overrides={
            "uav_mobility_control": "policy",
            "uav_velocity_delta_max": 1.0,
            "uav_direction_delta_max": 0.1,
            "uav_p_delta_max": 0.05,
        }
    )
    n_agents = int(env_cfg.n_ch)
    state_dim = specs.state_dim(env_cfg)
    base_dim = specs.total_param_dim(env_cfg)
    full_dim = specs.per_ch_param_dim(env_cfg)
    seen_dims: list[int] = []

    class _Evaluator:
        def select_actions(self, states):
            return [(0, np.zeros((base_dim,), dtype=np.float32)) for _ in states]

    class _VecEnv:
        n_envs = 1

        def reset(self, p_trans=None):
            return np.zeros((1, n_agents, state_dim), dtype=np.float32)

        def step(self, actions):
            for _, params in actions[0]:
                seen_dims.append(int(np.asarray(params).size))
                np.testing.assert_allclose(np.asarray(params)[-3:], 0.0)
            return (
                np.ones((1, n_agents, state_dim), dtype=np.float32),
                np.zeros((1, n_agents), dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                [{}],
            )

        def get_metrics(self):
            return (
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([float(n_agents * specs.n_des(env_cfg))], dtype=np.float32),
            )

    metrics = _run_eval_loop(
        algorithm="qmix",
        evaluator=_Evaluator(),
        vecenv=_VecEnv(),
        env_cfg=env_cfg,
        callbacks=CallbackManager([PolicyMobilityCallback(env_cfg=env_cfg)]),
        p_trans=None,
        episodes=1,
        steps=1,
    )
    assert seen_dims == [full_dim] * n_agents
    assert len(metrics["reward"]) == 1


def test_evaluation_cross_algo_argument_misuse_is_rejected():
    with pytest.raises(ValueError, match="policy_name"):
        run_evaluation("qmix", checkpoint_path=Path("missing.pth"), policy_name="random")
    with pytest.raises(ValueError, match="power_mode"):
        run_evaluation("qmix", checkpoint_path=Path("missing.pth"), power_mode="fixed_mid")
    with pytest.raises(ValueError, match="checkpoint"):
        run_evaluation("heuristic", checkpoint_path=Path("bad.pth"))
    with pytest.raises(ValueError, match="callbacks"):
        run_evaluation("vdn", checkpoint_path=Path("missing.pth"), callback_overrides=["policy_mobility"])


def _greedy_batch_actions(trainer, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_envs, n_agents, _ = states.shape
    discrete = np.zeros((n_envs, n_agents), dtype=np.int32)
    params = np.zeros((n_envs, n_agents, trainer.n_actions * trainer.param_dim), dtype=np.float32)
    for i, agent in enumerate(trainer.agents):
        ad, ap = agent.select_action_batch(states[:, i, :], epsilon=0.0)
        discrete[:, i] = ad
        params[:, i, :] = ap
    return discrete, params


def test_train_save_reload_eval_first_greedy_action_is_identical(tmp_path):
    overrides = _tiny_dqn_overrides(seed=31)
    trained = run_training("qmix", algo_overrides=overrides, no_save=True)
    ckpt_path = save_checkpoint(
        path=tmp_path / "qmix_weights.pth",
        algorithm="qmix",
        trainer=trained.trainer,
        callbacks=[],
    )

    env = Environ()
    p_trans = make_fixed_p_trans(env)
    vecenv = SubprocVecEnv(1, p_trans=p_trans, start_method="fork", seed=31)
    try:
        reset_states = vecenv.reset(p_trans)
    finally:
        vecenv.close()
    a_orig, p_orig = _greedy_batch_actions(trained.trainer, reset_states)

    env_cfg = load_env_config()
    algo_cfg = load_algo_config("qmix", overrides=overrides, env_cfg=env_cfg)
    fresh = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    load_trainer_state_dict(fresh, ckpt_path, "qmix", device="cpu")

    vecenv = SubprocVecEnv(1, p_trans=p_trans, start_method="fork", seed=31)
    try:
        reloaded_states = vecenv.reset(p_trans)
    finally:
        vecenv.close()
    np.testing.assert_allclose(reloaded_states, reset_states, rtol=0.0, atol=0.0)
    a_reloaded, p_reloaded = _greedy_batch_actions(fresh, reloaded_states)

    np.testing.assert_array_equal(a_reloaded, a_orig)
    np.testing.assert_allclose(p_reloaded, p_orig, rtol=0.0, atol=1e-7)
