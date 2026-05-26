from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from _preset_expectations import EXPECTED_PRESET_VALUES, PRESET_ALGORITHMS
from src.config.loader import (
    _deep_merge,
    load_algo_config,
    load_env_config,
    load_experiment_preset,
)
from src.config.schema import ALGO_CONFIG_TYPES, MAPPOConfig
from src.training.runner import run_training


def _write_preset(path: Path, *, extra: str = "", env: str = "{}", algo: str = "{}") -> None:
    path.write_text(
        "algorithm: qmix\n"
        "description: temp preset\n"
        "source: temp@local\n"
        f"env: {env}\n"
        f"algo: {algo}\n"
        f"{extra}",
        encoding="utf-8",
    )


def test_builtin_preset_names_allow_underscores_and_hyphens():
    ep = load_experiment_preset("qmix_plain_baseline")
    assert ep.algorithm == "qmix"
    assert ep.path.name == "qmix_plain_baseline.yaml"

    with pytest.raises(FileNotFoundError, match="not-real.yaml"):
        load_experiment_preset("not-real")


def test_explicit_yaml_path_resolves_relative_to_cwd(tmp_path, monkeypatch):
    preset = tmp_path / "local.yaml"
    _write_preset(preset)
    monkeypatch.chdir(tmp_path)

    ep = load_experiment_preset("local.yaml")

    assert ep.path == preset.resolve()
    assert len(ep.sha256) == 64


def test_preset_rejects_unknown_top_level_and_non_mapping_sections(tmp_path):
    unknown = tmp_path / "unknown.yaml"
    _write_preset(unknown, extra="extra: nope\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_experiment_preset(unknown)

    bad_env = tmp_path / "bad_env.yaml"
    _write_preset(bad_env, env="1")
    with pytest.raises(TypeError, match="env"):
        load_experiment_preset(bad_env)

    bad_algo = tmp_path / "bad_algo.yaml"
    _write_preset(bad_algo, algo="[]")
    with pytest.raises(TypeError, match="algo"):
        load_experiment_preset(bad_algo)


def test_preset_algorithm_mismatch_is_rejected_before_training():
    with pytest.raises(ValueError, match="does not match requested algorithm"):
        run_training("iql", preset="qmix_plain_baseline", no_save=True)


@pytest.mark.parametrize("name,algorithm", PRESET_ALGORITHMS.items())
def test_preset_algo_keys_match_config_dataclass_exactly(name, algorithm):
    ep = load_experiment_preset(name)
    expected_fields = {field.name for field in dataclasses.fields(ALGO_CONFIG_TYPES[algorithm])}
    assert set(ep.algo) == expected_fields


@pytest.mark.parametrize("name,algorithm", PRESET_ALGORITHMS.items())
def test_all_presets_load_through_env_and_algo_config(name, algorithm):
    ep = load_experiment_preset(name)
    env_cfg = load_env_config(overrides=ep.env)
    cfg = load_algo_config(algorithm, overrides=ep.algo, env_cfg=env_cfg)
    assert type(cfg) is ALGO_CONFIG_TYPES[algorithm]


def test_mappo_preset_exactly_matches_mappo_field_set():
    ep = load_experiment_preset("mappo_baseline")
    assert set(ep.algo) == {field.name for field in dataclasses.fields(MAPPOConfig)}


def test_preset_lr_null_fields_settle_to_lr_q():
    env_cfg = load_env_config()
    qmix = load_algo_config(
        "qmix",
        overrides=load_experiment_preset("qmix_wm_block_jp_baseline").algo,
        env_cfg=env_cfg,
    )
    qplex = load_algo_config(
        "qplex",
        overrides=load_experiment_preset("qplex_baseline").algo,
        env_cfg=env_cfg,
    )

    assert qmix.lr_mixer == pytest.approx(qmix.lr_q)
    assert qmix.lr_jammer == pytest.approx(qmix.lr_q)
    assert qplex.lr_mixer == pytest.approx(qplex.lr_q)


@pytest.mark.parametrize("name,expected", EXPECTED_PRESET_VALUES.items())
def test_preset_expected_values(name, expected):
    ep = load_experiment_preset(name)
    for key, value in expected.items():
        assert ep.algo[key] == value, f"{name}.{key}"


def test_runtime_overrides_beat_preset_env_and_algo(tmp_path):
    preset = tmp_path / "policy_env.yaml"
    _write_preset(
        preset,
        env=(
            "{uav_mobility_control: policy, uav_velocity_delta_max: 1.0, "
            "uav_direction_delta_max: 0.1, uav_p_delta_max: 0.05}"
        ),
        algo="{callbacks: [policy_mobility], n_episode: 10}",
    )
    ep = load_experiment_preset(preset)

    env_overrides = _deep_merge(ep.env, {"uav_mobility_control": "gauss_markov"})
    algo_overrides = _deep_merge(ep.algo, {"n_episode": 1})
    env_cfg = load_env_config(overrides=env_overrides)
    algo_cfg = load_algo_config("qmix", overrides=algo_overrides, env_cfg=env_cfg)

    assert env_cfg.uav_mobility_control == "gauss_markov"
    assert algo_cfg.n_episode == 1


def test_deep_merge_lists_replace_at_leaf_and_nested_dict_level():
    assert _deep_merge({"callbacks": ["a", "b"]}, {"callbacks": ["c"]}) == {
        "callbacks": ["c"]
    }
    assert _deep_merge({"algo": {"callbacks": ["a"]}}, {"algo": {"callbacks": ["c"]}}) == {
        "algo": {"callbacks": ["c"]}
    }


def test_callback_override_replaces_preset_list_and_revalidates_dependencies():
    cfg = load_algo_config("qmix", overrides={"callbacks": ["critic_stable"]})
    assert cfg.callbacks == ["critic_stable"]

    with pytest.raises(ValueError, match="requires 'value_expansion'"):
        run_training(
            "qmix",
            preset="qmix_wm_concurrent_baseline",
            algo_overrides={"callbacks": ["wm_concurrent"]},
            no_save=True,
        )
