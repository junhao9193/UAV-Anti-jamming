"""Stage 2 loader 测试。

覆盖：
- baseline env.yaml 解析 dict 与 ``load_env_config()`` 等价；
- sha256 与 Stage 0 一致；
- unknown / missing / 类型校验；
- yaml_path 是叠加而非完整替代；
- DQN 族 / MAPPO 的合并顺序；
- ``lr_mixer: null`` 在 QMIX / QPLEX 落定为 ``lr_q``，VDN 没有该字段；
- ``env_run_summary`` 含 overrides 字段且 28 键完整。
"""

import dataclasses
import subprocess
from pathlib import Path

import pytest
import yaml

from src.config.loader import (
    _validate_complete,
    config_sha256,
    env_run_summary,
    load_algo_config,
    load_env_config,
    load_train_config,
)
from src.config.schema import EnvConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED_ENV_YAML = _REPO_ROOT / "UAV-Ultra" / "src" / "config" / "defaults" / "env.yaml"
BASELINE_COMMIT = "2360ab92ec438528f6e194feda2405f9e943179d"
BASELINE_SHA = "34c3f0a53b28c414656d91a0ec136a64fb9de546d873e2e6f9cd2421879119da"


def _git_show_baseline_env_yaml() -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "show",
            f"{BASELINE_COMMIT}:UAV-Jammer-RL/configs/env.yaml",
        ],
        text=True,
    )


# -------- (a) baseline 63 键作为子集严格相等 --------
# Stage 3.5 给 packaged env.yaml 额外加了 9 个 mobility 键，因此原 dict-equal 断言失效；
# 改为「baseline 中每个 key 都出现在 EnvConfig 里，且值完全相等」的子集校验。

def test_load_env_config_baseline_keys_subset_equal():
    baseline = yaml.safe_load(_git_show_baseline_env_yaml())
    cfg_dict = dataclasses.asdict(load_env_config())
    for key, value in baseline.items():
        assert key in cfg_dict, f"baseline key missing from EnvConfig: {key!r}"
        assert cfg_dict[key] == value, (
            f"baseline key {key!r} value drift: got {cfg_dict[key]!r}, expected {value!r}"
        )


def test_env_config_fields_superset_of_baseline_yaml():
    baseline = yaml.safe_load(_git_show_baseline_env_yaml())
    env_fields = {f.name for f in dataclasses.fields(EnvConfig)}
    assert set(baseline.keys()).issubset(env_fields)


# -------- (b) Stage 3.5 mobility 默认值必须保持 baseline 行为 --------

def test_mobility_defaults_preserve_baseline_behavior():
    cfg = load_env_config()
    assert cfg.uav_mobility_control == "gauss_markov"
    assert cfg.jammer_mobility_model == "gauss_markov"
    assert cfg.uav_velocity_delta_max == 0.0
    assert cfg.uav_direction_delta_max == 0.0
    assert cfg.uav_p_delta_max == 0.0
    assert cfg.jammer_guidance_strength == 0.0
    assert cfg.observation_include_mobility is False
    assert cfg.mobility_oob_penalty_weight == 0.0
    assert cfg.mobility_energy_weight == 0.0


# -------- (c) sha256 函数本身用 tmp 文件测，不再钉 packaged env.yaml 的 sha --------

def test_config_sha256_on_tmp_file(tmp_path):
    """packaged env.yaml 会随 stage 增长，sha 也会变；这里只测函数本身正确。"""
    import hashlib
    f = tmp_path / "data.bin"
    payload = b"uav-ultra stage 3.5\n"
    f.write_bytes(payload)
    assert config_sha256(f) == hashlib.sha256(payload).hexdigest()


# -------- (c) unknown key --------

def test_unknown_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_real_field: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_env_config(yaml_path=bad)


# -------- (d) missing key 通过 _validate_complete 单独验 --------

def test_validate_complete_rejects_missing_key():
    with pytest.raises(ValueError, match="missing required key"):
        _validate_complete({"n_ch": 4}, EnvConfig)


# -------- (e) 类型校验 --------

def test_int_field_rejects_float_literal(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("n_ch: 4.0\n", encoding="utf-8")
    with pytest.raises(TypeError, match="n_ch"):
        load_env_config(yaml_path=bad)


def test_int_field_rejects_bool_literal(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("n_ch: true\n", encoding="utf-8")
    with pytest.raises(TypeError, match="n_ch"):
        load_env_config(yaml_path=bad)


def test_bool_field_rejects_int_literal(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("csi_clip: 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="csi_clip"):
        load_env_config(yaml_path=bad)


def test_float_field_accepts_int_and_casts(tmp_path):
    over = tmp_path / "over.yaml"
    over.write_text("t_Rx: 1\n", encoding="utf-8")
    cfg = load_env_config(yaml_path=over)
    assert cfg.t_Rx == 1.0
    assert type(cfg.t_Rx) is float


def test_fast_fading_clip_only_low_null_rejected(tmp_path):
    """clip 区间必须成对：只把 low 设为 null 应在 loader 层报错。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("fast_fading_db_clip_low: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fast_fading_db_clip"):
        load_env_config(yaml_path=bad)


def test_fast_fading_clip_only_high_null_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("fast_fading_db_clip_high: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fast_fading_db_clip"):
        load_env_config(yaml_path=bad)


def test_fast_fading_clip_low_not_less_than_high_rejected(tmp_path):
    """反向区间或相等区间应在 loader 层报错，避免环境层延迟暴露。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "fast_fading_db_clip_low: 10.0\nfast_fading_db_clip_high: -30.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly less than"):
        load_env_config(yaml_path=bad)

    eq = tmp_path / "eq.yaml"
    eq.write_text(
        "fast_fading_db_clip_low: 5.0\nfast_fading_db_clip_high: 5.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly less than"):
        load_env_config(yaml_path=eq)


def test_fast_fading_clip_both_null_is_ok(tmp_path):
    """两个都关掉是合法的（关闭剪切）。"""
    over = tmp_path / "over.yaml"
    over.write_text(
        "fast_fading_db_clip_low: null\nfast_fading_db_clip_high: null\n",
        encoding="utf-8",
    )
    cfg = load_env_config(yaml_path=over)
    assert cfg.fast_fading_db_clip_low is None
    assert cfg.fast_fading_db_clip_high is None


def test_p_trans_preferred_next_states_out_of_range_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("p_trans_preferred_next_states: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="p_trans_preferred_next_states"):
        load_env_config(yaml_path=bad)

    too_large = tmp_path / "too_large.yaml"
    too_large.write_text("p_trans_preferred_next_states: 999999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="p_trans_preferred_next_states"):
        load_env_config(yaml_path=too_large)


def test_p_trans_preference_strength_negative_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("p_trans_preference_strength: -0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="p_trans_preference_strength"):
        load_env_config(yaml_path=bad)


def test_optional_int_seed_accepts_null_and_int(tmp_path):
    over = tmp_path / "over.yaml"
    over.write_text("env_seed: 123\n", encoding="utf-8")
    cfg = load_env_config(yaml_path=over)
    assert cfg.env_seed == 123

    over2 = tmp_path / "over2.yaml"
    over2.write_text("env_seed: null\n", encoding="utf-8")
    cfg2 = load_env_config(yaml_path=over2)
    assert cfg2.env_seed is None


# -------- (f) yaml_path 是叠加而非完整替代 --------

def test_yaml_path_partial_override_inherits_defaults(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("env_seed: 42\n", encoding="utf-8")
    cfg = load_env_config(yaml_path=partial)
    assert cfg.env_seed == 42
    # 其他字段从 packaged defaults 补齐
    assert cfg.n_channel == 6
    assert cfg.n_ch == 4
    assert cfg.t_Rx == 0.9


# -------- (g) DQN 族合并顺序 --------

def test_qmix_merges_train_default_and_algo():
    cfg = load_algo_config("qmix")
    # 来自 train/default.yaml
    assert cfg.num_envs == 32
    assert cfg.batch_size == 256
    assert cfg.gamma == 0.99
    assert cfg.target_update_interval == 200
    # 来自 qmix.yaml
    assert cfg.epsilon_start == 1.0
    assert cfg.mixing_hidden_dim == 32
    assert cfg.hypernet_hidden_dim == 64
    assert cfg.value_target_clip == 1000.0
    assert cfg.callbacks == []
    assert cfg.value_expansion_alpha_model == pytest.approx(0.5)
    assert cfg.value_expansion_seq_len == 4
    assert cfg.value_expansion_td_lambda == pytest.approx(0.8)
    assert cfg.value_expansion_rollout_k == 4
    assert cfg.value_expansion_model_warmup_ep == 200
    assert cfg.value_expansion_ramp_start_ep == 300
    assert cfg.value_expansion_ramp_end_ep == 500
    assert cfg.value_expansion_alpha_model_max == pytest.approx(0.01)
    assert cfg.wm_buffer_capacity == 200000
    assert cfg.wm_hidden_dim == 256
    assert cfg.wm_n_layers == 1
    assert cfg.wm_stochastic_dim == 32
    assert cfg.wm_kl_beta == pytest.approx(0.1)
    assert cfg.wm_free_nats == pytest.approx(1.0)
    assert cfg.wm_lr == pytest.approx(0.001)
    assert cfg.wm_max_grad_norm == pytest.approx(0.0)
    assert cfg.critic_stable_tau == pytest.approx(0.005)
    assert cfg.critic_stable_lr_scale == pytest.approx(1.0)
    assert cfg.critic_stable_lr_decay_enabled is False
    assert cfg.critic_stable_lr_decay_start_ep == 1500
    assert cfg.critic_stable_lr_decay_end_ep == 3000
    assert cfg.critic_stable_lr_decay_min == pytest.approx(0.1)
    assert not hasattr(cfg, "value_expansion_critic_warmup_ep")


def test_iql_merges_train_default_and_algo():
    cfg = load_algo_config("iql")
    assert cfg.num_envs == 32
    assert cfg.epsilon_start == 1.0
    assert cfg.epsilon_decay == 0.995


def test_qplex_inherits_n_heads():
    cfg = load_algo_config("qplex")
    assert cfg.n_heads == 4


def test_overrides_beat_yaml_path(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("num_envs: 8\n", encoding="utf-8")
    cfg = load_algo_config("qmix", yaml_path=partial, overrides={"num_envs": 4})
    assert cfg.num_envs == 4


# -------- (h) MAPPO 不合并 train/default --------

def test_mappo_does_not_merge_train_default():
    cfg = load_algo_config("mappo")
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_range == 0.2
    assert cfg.n_steps is None
    assert cfg.lr == pytest.approx(3.0e-4)
    # 这些字段属于 DQN 族训练循环，MAPPO 配置不应携带它们
    assert not hasattr(cfg, "num_envs")
    assert not hasattr(cfg, "buffer_capacity")
    assert not hasattr(cfg, "epsilon_start")
    assert not hasattr(cfg, "lr_q")


# -------- (i) lr_mixer 落定 & VDN 没有此字段 --------

def test_qmix_lr_mixer_falls_back_to_lr_q():
    cfg = load_algo_config("qmix")
    assert cfg.lr_mixer is not None
    assert cfg.lr_mixer == cfg.lr_q


def test_qplex_lr_mixer_falls_back_to_lr_q():
    cfg = load_algo_config("qplex")
    assert cfg.lr_mixer is not None
    assert cfg.lr_mixer == cfg.lr_q


def test_qmix_lr_mixer_explicit_overrides_takes_priority():
    cfg = load_algo_config("qmix", overrides={"lr_mixer": 5.0e-4})
    assert cfg.lr_mixer == pytest.approx(5.0e-4)


def test_vdn_has_no_lr_mixer_attr():
    cfg = load_algo_config("vdn")
    assert not hasattr(cfg, "lr_mixer")


def test_qmix_callbacks_validate_allowed_combinations():
    cfg = load_algo_config(
        "qmix",
        overrides={"callbacks": ["critic_stable", "value_expansion", "wm_concurrent"]},
    )
    assert cfg.callbacks == ["critic_stable", "value_expansion", "wm_concurrent"]

    cfg_block = load_algo_config(
        "qmix",
        overrides={"callbacks": ["value_expansion", "wm_block_alternating"]},
    )
    assert cfg_block.callbacks == ["value_expansion", "wm_block_alternating"]

    with pytest.raises(ValueError, match="unknown callback"):
        load_algo_config("qmix", overrides={"callbacks": ["not_real"]})

    with pytest.raises(ValueError, match="wm_concurrent.*requires 'value_expansion'"):
        load_algo_config("qmix", overrides={"callbacks": ["wm_concurrent"]})

    with pytest.raises(ValueError, match="wm_block_alternating.*requires 'value_expansion'"):
        load_algo_config("qmix", overrides={"callbacks": ["wm_block_alternating"]})

    with pytest.raises(ValueError, match="value_expansion.*requires exactly one"):
        load_algo_config("qmix", overrides={"callbacks": ["value_expansion"]})

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_algo_config(
            "qmix",
            overrides={"callbacks": ["value_expansion", "wm_concurrent", "wm_block_alternating"]},
        )


def test_qmix_callbacks_alias_wm_alternating_to_wm_concurrent_with_future_warning():
    with pytest.warns(FutureWarning, match="wm_alternating"):
        cfg = load_algo_config(
            "qmix",
            overrides={"callbacks": ["value_expansion", "wm_alternating"]},
        )
    assert cfg.callbacks == ["value_expansion", "wm_concurrent"]


def test_qmix_value_expansion_fields_validate_ranges():
    cfg = load_algo_config(
        "qmix",
        overrides={
            "value_expansion_alpha_model": 0.25,
            "value_expansion_seq_len": 3,
            "value_expansion_td_lambda": 0.7,
            "value_expansion_rollout_k": 2,
            "value_expansion_model_warmup_ep": 20,
            "value_expansion_ramp_start_ep": 30,
            "value_expansion_ramp_end_ep": 50,
            "value_expansion_alpha_model_max": 0.02,
        },
    )
    assert cfg.value_expansion_alpha_model == pytest.approx(0.25)
    assert cfg.value_expansion_seq_len == 3
    assert cfg.value_expansion_td_lambda == pytest.approx(0.7)
    assert cfg.value_expansion_rollout_k == 2
    assert cfg.value_expansion_model_warmup_ep == 20
    assert cfg.value_expansion_ramp_start_ep == 30
    assert cfg.value_expansion_ramp_end_ep == 50
    assert cfg.value_expansion_alpha_model_max == pytest.approx(0.02)

    for key, value in (
        ("value_expansion_alpha_model", 1.5),
        ("value_expansion_seq_len", 0),
        ("value_expansion_td_lambda", -0.1),
        ("value_expansion_rollout_k", 0),
        ("value_expansion_model_warmup_ep", -1),
        ("value_expansion_alpha_model_max", 1.5),
    ):
        with pytest.raises(ValueError, match=key):
            load_algo_config("qmix", overrides={key: value})

    with pytest.raises(ValueError, match="value_expansion_ramp_start_ep"):
        load_algo_config(
            "qmix",
            overrides={
                "value_expansion_model_warmup_ep": 20,
                "value_expansion_ramp_start_ep": 19,
            },
        )
    with pytest.raises(ValueError, match="value_expansion_ramp_end_ep"):
        load_algo_config(
            "qmix",
            overrides={
                "value_expansion_model_warmup_ep": 20,
                "value_expansion_ramp_start_ep": 30,
                "value_expansion_ramp_end_ep": 29,
            },
        )


def test_non_qmix_callbacks_are_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        load_algo_config("vdn", overrides={"callbacks": ["critic_stable"]})


def test_qmix_jp_fields_default_values_and_lr_jammer_settle():
    """Stage 8：默认 lr_jammer=null → loader settle 为 lr_q。"""
    cfg = load_algo_config("qmix")
    assert cfg.jammer_history_len == 4
    assert cfg.jammer_pred_hidden_dim == 64
    assert cfg.jammer_aux_weight == pytest.approx(0.1)
    assert cfg.jammer_warmup_episodes == 200
    assert cfg.use_jammer_feature is True
    assert cfg.lr_jammer == pytest.approx(cfg.lr_q)


def test_qmix_jp_lr_jammer_explicit_override_skips_settle():
    cfg = load_algo_config("qmix", overrides={"lr_jammer": 5e-4})
    assert cfg.lr_jammer == pytest.approx(5e-4)


def test_qmix_jp_fields_range_checks():
    for key, value in (
        ("jammer_history_len", 0),
        ("jammer_pred_hidden_dim", 0),
        ("jammer_aux_weight", -1.0),
        ("jammer_warmup_episodes", -1),
    ):
        with pytest.raises(ValueError, match=key):
            load_algo_config("qmix", overrides={key: value})


def test_qmix_wm_and_critic_stable_fields_range_checks():
    for key, value in (
        ("wm_buffer_capacity", 0),
        ("wm_hidden_dim", 0),
        ("wm_n_layers", 0),
        ("wm_stochastic_dim", 0),
        ("wm_kl_beta", -0.1),
        ("wm_free_nats", -0.1),
        ("wm_lr", 0.0),
        ("wm_max_grad_norm", -0.1),
        ("critic_stable_tau", 1.5),
        ("critic_stable_lr_scale", 0.0),
        ("critic_stable_lr_decay_start_ep", -1),
        ("critic_stable_lr_decay_min", 1.5),
    ):
        with pytest.raises(ValueError, match=key):
            load_algo_config("qmix", overrides={key: value})

    with pytest.raises(ValueError, match="critic_stable_lr_decay_end_ep"):
        load_algo_config(
            "qmix",
            overrides={
                "critic_stable_lr_decay_start_ep": 10,
                "critic_stable_lr_decay_end_ep": 9,
            },
        )


def test_policy_env_requires_explicit_policy_mobility_callback():
    env_cfg = load_env_config(
        overrides={
            "uav_mobility_control": "policy",
            "uav_velocity_delta_max": 1.0,
            "uav_direction_delta_max": 0.1,
            "uav_p_delta_max": 0.05,
        }
    )
    with pytest.raises(ValueError, match="policy_mobility"):
        load_algo_config("qmix", env_cfg=env_cfg)

    cfg = load_algo_config("qmix", env_cfg=env_cfg, overrides={"callbacks": ["policy_mobility"]})
    assert cfg.callbacks == ["policy_mobility"]


# -------- load_train_config & 未知算法名 --------

def test_load_train_config_smoke():
    cfg = load_train_config()
    assert cfg.n_episode == 1500
    assert cfg.num_envs == 32
    assert cfg.n_steps is None


def test_unknown_algo_name_raises():
    with pytest.raises(ValueError, match="unknown algo"):
        load_algo_config("notreal")


# -------- Stage 3.5 mobility cross-field 校验 --------

def test_is_jammer_moving_false_rejected_by_loader(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("is_jammer_moving: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is_jammer_moving"):
        load_env_config(yaml_path=bad)


def test_uav_mobility_control_invalid_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("uav_mobility_control: random_walk\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uav_mobility_control"):
        load_env_config(yaml_path=bad)


def test_jammer_mobility_model_invalid_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("jammer_mobility_model: fixed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jammer_mobility_model"):
        load_env_config(yaml_path=bad)


def test_policy_mode_requires_positive_delta_max(tmp_path):
    """policy 模式 + 默认 0 的 delta_max → loader 报错（避免环境层延迟暴露）。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("uav_mobility_control: policy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="delta_max"):
        load_env_config(yaml_path=bad)


def test_policy_mode_accepts_positive_delta_max(tmp_path):
    over = tmp_path / "over.yaml"
    over.write_text(
        "uav_mobility_control: policy\n"
        "uav_velocity_delta_max: 1.0\n"
        "uav_direction_delta_max: 0.1\n"
        "uav_p_delta_max: 0.05\n",
        encoding="utf-8",
    )
    cfg = load_env_config(yaml_path=over)
    assert cfg.uav_mobility_control == "policy"
    assert cfg.uav_velocity_delta_max == 1.0


def test_jammer_guidance_strength_out_of_range_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("jammer_guidance_strength: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jammer_guidance_strength"):
        load_env_config(yaml_path=bad)

    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("jammer_guidance_strength: -0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jammer_guidance_strength"):
        load_env_config(yaml_path=bad2)


def test_jammer_guidance_strength_boundary_accepted(tmp_path):
    for value in (0.0, 0.5, 1.0):
        over = tmp_path / f"g_{value}.yaml"
        over.write_text(f"jammer_guidance_strength: {value}\n", encoding="utf-8")
        cfg = load_env_config(yaml_path=over)
        assert cfg.jammer_guidance_strength == value


# -------- env_run_summary --------

def test_env_run_summary_contains_overrides_and_28_keys():
    cfg = load_env_config()
    summary = env_run_summary(cfg, _PACKAGED_ENV_YAML, overrides={"foo": "bar"})
    # Stage 3.5 后 packaged env.yaml 已扩展，sha 不再 == baseline；只验长度与函数本身正确。
    assert isinstance(summary["env_config_sha256"], str)
    assert len(summary["env_config_sha256"]) == 64
    assert summary["overrides"] == {"foo": "bar"}
    expected = {
        "n_ch", "n_des", "n_jammer", "n_channel", "state_dim",
        "action_dim", "param_dim_per_action", "total_param_dim",
        "env_seed", "data_size", "t_Rx", "jammer_power", "max_episode_steps",
        "p_trans_seed", "p_trans_preferred_next_states", "p_trans_preference_strength",
        "jammer_reactive_beta", "jammer_memory_window", "jammer_reactive_observe_prob",
        "uav_interference_scale", "reward_energy_weight", "reward_jump_weight",
        "fairness_min_success_rate", "fairness_weight", "enable_fast_fading",
        "fast_fading_rho", "csi_noise_std", "sensing_noise_std",
    }
    assert set(summary["env_summary"].keys()) == expected
    assert len(summary["env_summary"]) == 28
