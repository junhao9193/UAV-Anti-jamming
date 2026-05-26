"""Stage 2 specs 测试。

期望值全部从 Stage 0 golden master JSON 读出（``env_trace_2360ab92.json`` 的 ``traces[0]``），
零硬编码。state_shape 形状是 ``[n_ch, state_dim]``，因此用 ``[-1]`` 取 state_dim、``[0]`` 取 n_ch。
"""

import json
from math import perm
from pathlib import Path

from src.config import specs
from src.config.loader import load_env_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO_ROOT / "UAV-Ultra" / "tests" / "fixtures" / "golden_master"
    / "env_trace_2360ab92.json"
)


def _trace0() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["traces"][0]


def test_state_dim_equals_state_shape_last_dim():
    cfg = load_env_config()
    trace = _trace0()
    assert specs.state_dim(cfg) == trace["state_shape"][-1]


def test_n_ch_equals_state_shape_first_dim():
    cfg = load_env_config()
    trace = _trace0()
    assert cfg.n_ch == trace["state_shape"][0]


def test_action_dim_matches_golden_master():
    cfg = load_env_config()
    trace = _trace0()
    assert specs.action_dim(cfg) == trace["action_dim"]


def test_param_dim_per_action_matches_golden_master():
    cfg = load_env_config()
    trace = _trace0()
    assert specs.param_dim_per_action(cfg) == trace["param_dim_per_action"]


def test_total_param_dim_matches_golden_master():
    cfg = load_env_config()
    trace = _trace0()
    assert specs.total_param_dim(cfg) == trace["total_param_dim"]


def test_n_des_matches_yaml_field():
    cfg = load_env_config()
    assert specs.n_des(cfg) == cfg.n_cm_for_a_ch


def test_n_uav_pair_is_n_ch_times_n_des():
    cfg = load_env_config()
    assert specs.n_uav_pair(cfg) == cfg.n_ch * cfg.n_cm_for_a_ch


def test_jammer_state_dim_is_perm():
    cfg = load_env_config()
    assert specs.jammer_state_dim(cfg) == perm(cfg.n_channel, cfg.n_jammer)


# -------- Stage 3.5 mobility 维度切换 --------

def test_mobility_action_dim_per_ch_switches_with_uav_mobility_control():
    """gauss_markov → 0；policy → 3。"""
    cfg = load_env_config()
    assert specs.mobility_action_dim_per_ch(cfg) == 0

    cfg_policy = load_env_config(overrides={
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 0.1,
        "uav_p_delta_max": 0.05,
    })
    assert specs.mobility_action_dim_per_ch(cfg_policy) == 3


def test_mobility_obs_dim_per_ch_switches_with_observation_flag():
    """observation_include_mobility=False → 0；True → 6。"""
    cfg = load_env_config()
    assert specs.mobility_obs_dim_per_ch(cfg) == 0

    cfg_ext = load_env_config(overrides={"observation_include_mobility": True})
    assert specs.mobility_obs_dim_per_ch(cfg_ext) == 6


def test_per_ch_param_dim_includes_mobility_extension():
    cfg = load_env_config()
    assert specs.per_ch_param_dim(cfg) == specs.total_param_dim(cfg)

    cfg_policy = load_env_config(overrides={
        "uav_mobility_control": "policy",
        "uav_velocity_delta_max": 1.0,
        "uav_direction_delta_max": 0.1,
        "uav_p_delta_max": 0.05,
    })
    assert specs.per_ch_param_dim(cfg_policy) == specs.total_param_dim(cfg_policy) + 3


def test_state_dim_default_stays_18_after_mobility_extension():
    """默认 cfg 下 state_dim 仍 = 18（与 Stage 0 golden master state_shape[-1] 一致）。"""
    cfg = load_env_config()
    assert specs.state_dim(cfg) == 18


def test_state_dim_grows_by_6_with_mobility_obs():
    cfg_ext = load_env_config(overrides={"observation_include_mobility": True})
    assert specs.state_dim(cfg_ext) == 24
