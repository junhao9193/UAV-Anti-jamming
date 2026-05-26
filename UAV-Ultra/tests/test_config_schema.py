"""Stage 2 schema 契约测试。

只关心 dataclass 形状与字段集合，不触发任何 YAML 加载。
"""

import dataclasses
from typing import get_args, get_origin, get_type_hints

from src.config.schema import (
    ALGO_CONFIG_TYPES,
    EnvConfig,
    IQLConfig,
    MAPPOConfig,
    QMIXConfig,
    QPLEXConfig,
    TrainConfig,
    VDNConfig,
)


def test_env_config_has_72_fields():
    """baseline 63 键 + Stage 3.5 mobility 9 键 = 72。改动数字需同步更新 env.yaml。"""
    assert len(dataclasses.fields(EnvConfig)) == 72


def test_env_config_mobility_fields_present():
    """Stage 3.5 新增的 9 个 mobility 字段必须全部声明。"""
    names = {f.name for f in dataclasses.fields(EnvConfig)}
    expected = {
        "uav_mobility_control", "jammer_mobility_model",
        "uav_velocity_delta_max", "uav_direction_delta_max", "uav_p_delta_max",
        "jammer_guidance_strength", "observation_include_mobility",
        "mobility_oob_penalty_weight", "mobility_energy_weight",
    }
    assert expected.issubset(names)


def test_env_config_nullable_seeds_are_optional():
    hints = get_type_hints(EnvConfig)
    for name in ("env_seed", "fast_fading_seed", "jammer_seed"):
        assert type(None) in get_args(hints[name]), f"{name} must be Optional[int]"


def test_env_config_clip_floats_are_optional():
    hints = get_type_hints(EnvConfig)
    for name in ("fast_fading_db_clip_low", "fast_fading_db_clip_high"):
        assert type(None) in get_args(hints[name]), f"{name} must be Optional[float]"


def test_env_config_channel_loss_db_is_list_float():
    hints = get_type_hints(EnvConfig)
    ann = hints["channel_loss_db"]
    assert get_origin(ann) is list
    assert get_args(ann) == (float,)


def test_all_configs_are_frozen():
    for dc in (EnvConfig, TrainConfig, IQLConfig, QMIXConfig, VDNConfig, QPLEXConfig, MAPPOConfig):
        assert dc.__dataclass_params__.frozen, f"{dc.__name__} must be frozen"


def test_algo_registry_keys():
    assert set(ALGO_CONFIG_TYPES.keys()) == {"iql", "qmix", "vdn", "qplex", "mappo"}
    assert ALGO_CONFIG_TYPES["iql"] is IQLConfig
    assert ALGO_CONFIG_TYPES["qmix"] is QMIXConfig
    assert ALGO_CONFIG_TYPES["vdn"] is VDNConfig
    assert ALGO_CONFIG_TYPES["qplex"] is QPLEXConfig
    assert ALGO_CONFIG_TYPES["mappo"] is MAPPOConfig


def test_vdn_has_no_lr_mixer_field():
    """VDN sum mixer 无可学习参数，schema 不该出现 lr_mixer。"""
    names = {f.name for f in dataclasses.fields(VDNConfig)}
    assert "lr_mixer" not in names


def test_qmix_qplex_have_lr_mixer_field():
    for dc in (QMIXConfig, QPLEXConfig):
        names = {f.name for f in dataclasses.fields(dc)}
        assert "lr_mixer" in names


def test_only_qmix_has_callbacks_field():
    assert "callbacks" in {f.name for f in dataclasses.fields(QMIXConfig)}
    for dc in (IQLConfig, VDNConfig, QPLEXConfig, MAPPOConfig):
        assert "callbacks" not in {f.name for f in dataclasses.fields(dc)}


def test_mappo_does_not_carry_dqn_fields():
    """MAPPO 不继承 TrainConfig，不应出现 DQN 族特有字段。"""
    names = {f.name for f in dataclasses.fields(MAPPOConfig)}
    for forbidden in (
        "num_envs", "buffer_capacity", "epsilon_start",
        "epsilon_min", "epsilon_decay", "lr_q", "lr_actor",
        "target_update_interval", "use_amp",
    ):
        assert forbidden not in names, f"MAPPO should not have {forbidden}"


def test_train_config_n_steps_is_optional_int():
    hints = get_type_hints(TrainConfig)
    assert type(None) in get_args(hints["n_steps"])


def test_mappo_n_steps_is_optional_int():
    hints = get_type_hints(MAPPOConfig)
    assert type(None) in get_args(hints["n_steps"])
