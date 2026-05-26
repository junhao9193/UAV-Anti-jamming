"""Stage 3 核心契约测试：新 ``Environ`` 与 Stage 0 golden master 严格对齐。

加载 ``env_trace_2360ab92.npz`` 的 ``seed_{0,1,2}`` 输入（p_trans + action trace），
逐步喂入新 ``Environ``，逐数值对比 ``state / reward / last_link_metrics`` 与 fixture 是否
在 ``rtol=1e-9`` 内一致。

强制断言：``np.asarray(env.get_state()).shape == (cfg.n_ch, specs.state_dim(cfg))``。

**不 monkey-patch**：所有 link metrics 走 ``env.last_link_metrics`` 公开调试接口，
完全替代 Stage 0 generator 用过的 ``compute_reward`` 包裹模式。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import specs
from src.config.loader import load_env_config
from src.envs import Environ


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = (
    _REPO_ROOT / "UAV-Ultra" / "tests" / "fixtures" / "golden_master"
)
_NPZ_PATH = _FIXTURE_DIR / "env_trace_2360ab92.npz"
_JSON_PATH = _FIXTURE_DIR / "env_trace_2360ab92.json"

# 旧环境用 float64；fixture 已存 float64。容差略松，避免不同 BLAS 实现的尾差。
_RTOL = 1e-9
_ATOL = 1e-9


def _load_fixture() -> tuple[dict, np.lib.npyio.NpzFile]:
    metadata = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    arrays = np.load(_NPZ_PATH)
    return metadata, arrays


def _build_action(discrete_step: np.ndarray, params_step: np.ndarray) -> list:
    """fixture 的 action 是 (n_ch,) 离散 + (n_ch, total_param_dim) 连续；还原为 env 输入。"""
    n_ch = discrete_step.shape[0]
    return [
        (int(discrete_step[i]), params_step[i].astype(np.float32).copy())
        for i in range(n_ch)
    ]


@pytest.mark.parametrize("trace_index", [0, 1, 2])
def test_env_step_aligns_with_golden_master(trace_index: int):
    """逐 seed 重放 golden master 输入，断言 state / reward / link metrics 数值对齐。"""
    metadata, arrays = _load_fixture()
    trace = metadata["traces"][trace_index]
    env_seed = int(trace["env_seed"])
    prefix = f"seed_{env_seed}"

    expected_states = arrays[f"{prefix}_states"]
    expected_rewards = arrays[f"{prefix}_rewards"]
    expected_actions_discrete = arrays[f"{prefix}_actions_discrete"]
    expected_actions_params = arrays[f"{prefix}_actions_params"]
    expected_deliveries = arrays[f"{prefix}_deliveries"]
    expected_success_flags = arrays[f"{prefix}_success_flags"]
    expected_transmit_times = arrays[f"{prefix}_transmit_times"]
    p_trans_loaded = arrays[f"{prefix}_p_trans"]

    cfg = load_env_config()
    env = Environ(config={"env_seed": env_seed})
    state = env.reset(p_trans=p_trans_loaded)

    # 强制形状 / state_dim 同步断言
    state_arr = np.asarray(state)
    assert state_arr.shape == (cfg.n_ch, specs.state_dim(cfg))
    # 初始 state 与 fixture 第 0 步对齐
    np.testing.assert_allclose(state_arr, expected_states[0], rtol=_RTOL, atol=_ATOL)

    steps = int(trace["steps_recorded"])
    for t in range(steps):
        action = _build_action(expected_actions_discrete[t], expected_actions_params[t])
        next_state, reward, done, info = env.step(action)

        # state 对齐
        next_state_arr = np.asarray(next_state)
        np.testing.assert_allclose(
            next_state_arr, expected_states[t + 1],
            rtol=_RTOL, atol=_ATOL,
            err_msg=f"state mismatch at seed={env_seed} step={t}",
        )

        # reward 对齐
        np.testing.assert_allclose(
            np.asarray(reward), expected_rewards[t],
            rtol=_RTOL, atol=_ATOL,
            err_msg=f"reward mismatch at seed={env_seed} step={t}",
        )

        # link metrics 三项全部对齐（plan §G：钉死链路成功 / 失败语义）
        np.testing.assert_allclose(
            env.last_link_metrics["delivery"], expected_deliveries[t],
            rtol=_RTOL, atol=_ATOL,
            err_msg=f"delivery mismatch at seed={env_seed} step={t}",
        )
        np.testing.assert_allclose(
            env.last_link_metrics["success_flags"], expected_success_flags[t],
            rtol=_RTOL, atol=_ATOL,
            err_msg=f"success_flags mismatch at seed={env_seed} step={t}",
        )
        np.testing.assert_allclose(
            env.last_link_metrics["transmit_times"], expected_transmit_times[t],
            rtol=_RTOL, atol=_ATOL,
            err_msg=f"transmit_times mismatch at seed={env_seed} step={t}",
        )


def test_state_shape_matches_specs_under_default_config():
    """默认 cfg 下，state 最后一维 = specs.state_dim(cfg) = 18；n_ch 维 = state_shape[0]。"""
    cfg = load_env_config()
    env = Environ(config={"env_seed": 0})
    state = env.reset()
    state_arr = np.asarray(state)
    assert state_arr.shape == (cfg.n_ch, specs.state_dim(cfg))
    assert state_arr.shape[-1] == 18


def test_state_shape_grows_with_observation_include_mobility():
    """Stage 3.5 mobility obs 开启时 state_dim 由 18 增到 24。"""
    cfg = load_env_config(overrides={"observation_include_mobility": True})
    env = Environ(config={"env_seed": 0, "observation_include_mobility": True})
    state = env.reset()
    state_arr = np.asarray(state)
    assert state_arr.shape == (cfg.n_ch, specs.state_dim(cfg))
    assert state_arr.shape[-1] == 24
