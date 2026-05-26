"""Heuristic policies：4 个策略对固定 obs 输出与 baseline 等价；evaluator 接口测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import build_evaluator, build_trainer
from src.algorithms.common import registered_names
from src.algorithms.heuristic import (
    GreedySensingPolicy,
    HeuristicDims,
    HeuristicEvalPolicy,
    MaxCSIPolicy,
    MinInterferencePolicy,
    RandomHoppingPolicy,
    build_heuristic_policy,
)
from src.config import specs
from src.config.loader import load_env_config


DIMS = HeuristicDims(n_channel=6, n_des=2, n_actions=36, param_dim=2)


def test_heuristic_registered():
    assert "heuristic" in registered_names()


def test_heuristic_build_trainer_returns_none():
    """plan locked decision #2：heuristic build_trainer 始终返回 None。"""
    env_cfg = load_env_config()
    trainer = build_trainer("heuristic", env_cfg=env_cfg)
    assert trainer is None


def test_heuristic_build_evaluator_works_without_trainer():
    env_cfg = load_env_config()
    evaluator = build_evaluator("heuristic", env_cfg=env_cfg)
    assert isinstance(evaluator, HeuristicEvalPolicy)


def test_random_hopping_policy_action_in_bounds():
    rng_seed = 42
    policy = RandomHoppingPolicy(DIMS, power_mode="fixed_mid", seed=rng_seed)
    obs = np.zeros(DIMS.n_des * DIMS.n_channel + DIMS.n_channel, dtype=np.float32)
    discrete, params = policy.select_action(obs)
    assert 0 <= discrete < DIMS.n_actions
    assert params.shape == (DIMS.n_actions * DIMS.param_dim,)
    # fixed_mid 在 chosen 槽位写 0.5
    start = discrete * DIMS.param_dim
    assert np.allclose(params[start : start + DIMS.param_dim], 0.5)


def test_greedy_sensing_policy_matches_baseline(baseline_import):
    """新 GreedySensingPolicy 与 baseline 对同一 obs 输出等价。"""
    baseline_mod = baseline_import("algorithms.heuristic.policies")
    bdims = baseline_mod.HeuristicDims(
        n_channel=DIMS.n_channel, n_des=DIMS.n_des,
        n_actions=DIMS.n_actions, param_dim=DIMS.param_dim,
    )
    old = baseline_mod.GreedySensingPolicy(bdims)
    new = GreedySensingPolicy(DIMS)

    rng = np.random.RandomState(7)
    obs = rng.randn(DIMS.n_des * DIMS.n_channel + DIMS.n_channel).astype(np.float32)
    d_old, p_old = old.select_action(obs)
    d_new, p_new = new.select_action(obs)
    assert d_old == d_new
    np.testing.assert_array_equal(p_old, p_new)


def test_max_csi_policy_matches_baseline(baseline_import):
    baseline_mod = baseline_import("algorithms.heuristic.policies")
    bdims = baseline_mod.HeuristicDims(
        n_channel=DIMS.n_channel, n_des=DIMS.n_des,
        n_actions=DIMS.n_actions, param_dim=DIMS.param_dim,
    )
    old = baseline_mod.MaxCSIPolicy(bdims)
    new = MaxCSIPolicy(DIMS)

    rng = np.random.RandomState(11)
    obs = rng.randn(DIMS.n_des * DIMS.n_channel + DIMS.n_channel).astype(np.float32)
    d_old, p_old = old.select_action(obs)
    d_new, p_new = new.select_action(obs)
    assert d_old == d_new
    np.testing.assert_array_equal(p_old, p_new)


def test_min_interference_policy_matches_baseline(baseline_import):
    baseline_mod = baseline_import("algorithms.heuristic.policies")
    bdims = baseline_mod.HeuristicDims(
        n_channel=DIMS.n_channel, n_des=DIMS.n_des,
        n_actions=DIMS.n_actions, param_dim=DIMS.param_dim,
    )
    old = baseline_mod.MinInterferencePolicy(bdims)
    new = MinInterferencePolicy(DIMS)

    rng = np.random.RandomState(13)
    obs = rng.randn(DIMS.n_des * DIMS.n_channel + DIMS.n_channel).astype(np.float32)
    d_old, p_old = old.select_action(obs)
    d_new, p_new = new.select_action(obs)
    assert d_old == d_new
    np.testing.assert_array_equal(p_old, p_new)


def test_build_heuristic_policy_dispatch():
    for name in ("random", "greedy_sensing", "max_csi", "min_interference"):
        policy = build_heuristic_policy(name, DIMS)
        obs = np.zeros(DIMS.n_des * DIMS.n_channel + DIMS.n_channel, dtype=np.float32)
        discrete, params = policy.select_action(obs)
        assert 0 <= discrete < DIMS.n_actions


def test_build_heuristic_policy_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown heuristic policy"):
        build_heuristic_policy("notreal", DIMS)


def test_heuristic_eval_policy_select_actions():
    env_cfg = load_env_config()
    evaluator = HeuristicEvalPolicy(env_cfg=env_cfg, policy_name="greedy_sensing")
    # state_dim = 18 = n_des(2)*n_channel(6) + n_channel(6)
    states = [np.random.RandomState(i).randn(18).astype(np.float32) for i in range(env_cfg.n_ch)]
    actions = evaluator.select_actions(states)
    assert len(actions) == env_cfg.n_ch
    for discrete, params in actions:
        assert 0 <= discrete < 36
        assert params.shape == (72,)  # 36 * 2


def test_heuristic_eval_policy_random_matches_baseline_per_agent_seeds(baseline_import):
    """Random evaluator 必须按 baseline 为每个 agent 使用 seed + 1009*i。"""
    baseline_mod = baseline_import("algorithms.heuristic.policies")
    env_cfg = load_env_config()
    seed = 17
    evaluator = HeuristicEvalPolicy(
        env_cfg=env_cfg,
        policy_name="random",
        seed=seed,
        power_mode="fixed_mid",
    )
    bdims = baseline_mod.HeuristicDims(
        n_channel=int(env_cfg.n_channel),
        n_des=int(specs.n_des(env_cfg)),
        n_actions=int(specs.action_dim(env_cfg)),
        param_dim=int(specs.param_dim_per_action(env_cfg)),
    )
    old_policies = [
        baseline_mod.build_heuristic_policy(
            "random",
            bdims,
            seed=seed + 1009 * i,
            power_mode="fixed_mid",
        )
        for i in range(int(env_cfg.n_ch))
    ]
    states = [
        np.random.RandomState(i).randn(specs.state_dim(env_cfg)).astype(np.float32)
        for i in range(int(env_cfg.n_ch))
    ]

    old_actions = [old_policies[i].select_action(states[i]) for i in range(int(env_cfg.n_ch))]
    new_actions = evaluator.select_actions(states)

    for (d_old, p_old), (d_new, p_new) in zip(old_actions, new_actions):
        assert d_new == d_old
        np.testing.assert_array_equal(p_new, p_old)
