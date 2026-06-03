from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.algorithms import build_trainer
from src.algorithms.qmix.evaluator import QMIXEvalPolicy
from src.config import specs
from src.config.loader import load_algo_config, load_env_config
from src.evaluation.runner import _run_eval_loop, run_evaluation
from src.training.callbacks import CallbackManager, build_callbacks
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


# --------------------------------------------------------------------------
# Stage 10 修复回归：eval 期 JP 必须用真实滚动的 sensing history，不退化成 default。
# --------------------------------------------------------------------------

_JP_CALLBACKS = ["value_expansion", "wm_concurrent", "jammer_prediction"]


def test_qmix_eval_policy_passes_sensing_history_to_jp_agent():
    """锁住 bug 修复点：QMIXEvalPolicy 把每 agent 的 sensing_history 透传给
    JP-aware agent.select_action；不传时为 None（旧 bug 路径）。"""
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={**_tiny_dqn_overrides(seed=41), "callbacks": _JP_CALLBACKS},
        env_cfg=env_cfg,
    )
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    # 前置确认：JP 组合下 trainer 造的是 JP-aware agent
    assert all(hasattr(a, "jammer_predictor") for a in trainer.agents)

    n_agents = int(env_cfg.n_ch)
    state_dim = int(specs.state_dim(env_cfg))
    H = int(algo_cfg.jammer_history_len)
    C = int(env_cfg.n_channel)

    captured: list = []
    for agent in trainer.agents:
        orig = agent.select_action

        def _wrap(orig):
            def _inner(state, epsilon, sensing_history=None):
                captured.append(sensing_history)
                return orig(state, epsilon, sensing_history=sensing_history)
            return _inner

        agent.select_action = _wrap(orig)

    policy = QMIXEvalPolicy(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    states = [np.zeros((state_dim,), dtype=np.float32) for _ in range(n_agents)]
    hist = np.arange(n_agents * H * C, dtype=np.float32).reshape(n_agents, H, C)

    # 传 history：每个 agent 收到对应的 (H, C) 切片，且不是 None
    policy.select_actions(states, sensing_histories=hist)
    assert len(captured) == n_agents
    assert all(h is not None for h in captured)
    for i, h in enumerate(captured):
        np.testing.assert_array_equal(np.asarray(h).reshape(H, C), hist[i])

    # 不传 history（旧 bug 行为）：agent 收到 None → 退回 default history
    captured.clear()
    policy.select_actions(states, sensing_histories=None)
    assert captured == [None] * n_agents


def test_qmix_jp_evaluation_smoke_runs_state_machine(tmp_path):
    """端到端：JP 模型 eval 全程跑通（reset_jp_state/advance/commit 不崩），
    callback_states 含 jammer_prediction。"""
    trained = run_training(
        "qmix",
        algo_overrides={**_tiny_dqn_overrides(seed=42), "callbacks": _JP_CALLBACKS},
        no_save=False,
        output_root=tmp_path,
    )
    result = run_evaluation(
        "qmix",
        checkpoint_path=trained.output_dir / "qmix_weights.pth",
        episodes=1,
        steps=3,
        num_envs=1,
        seed=42,
        device="cpu",
        start_method="fork",
        callback_overrides=_JP_CALLBACKS,
        no_save=True,
    )
    assert len(result.metrics["success_rate"]) == 1
    assert result.callback_states is not None
    assert "jammer_prediction" in result.callback_states


def test_eval_jp_history_advances_during_loop():
    """直接验证 _run_eval_loop 中 JP history 真的滚动（current_sensing_histories 在
    多步后不再等于 reset 时的初始 repeat）——证明状态机在 eval 中工作。"""
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={**_tiny_dqn_overrides(seed=43), "callbacks": _JP_CALLBACKS},
        env_cfg=env_cfg,
    )
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    callbacks = build_callbacks(_JP_CALLBACKS, env_cfg=env_cfg, algo_cfg=algo_cfg)
    callbacks.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    evaluator = QMIXEvalPolicy(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)

    p_trans = make_fixed_p_trans(Environ())
    vecenv = SubprocVecEnv(1, p_trans=p_trans, start_method="fork", seed=43)
    try:
        _run_eval_loop(
            algorithm="qmix",
            evaluator=evaluator,
            vecenv=vecenv,
            env_cfg=env_cfg,
            callbacks=callbacks,
            p_trans=p_trans,
            episodes=1,
            steps=5,
        )
    finally:
        vecenv.close()

    jp = callbacks._find_jp()
    assert jp is not None
    # reset 时 history 是初始 sensing 沿 H 维 repeat（H 帧全相同）；滚动后 H 帧应出现差异
    hist = jp.current_sensing_histories  # (n_envs, n_agents, H, C)
    assert hist is not None
    H = int(algo_cfg.jammer_history_len)
    if H >= 2:
        # 至少一个 (env, agent) 的相邻两帧不再相等 → 证明 history 在 eval 中被滚动过
        frame_diff = np.abs(hist[:, :, 1:, :] - hist[:, :, :-1, :]).sum()
        assert frame_diff > 0.0
