"""Stage 2 配置加载器。

合并顺序（loader 内部唯一支持的语义）：

- ``load_env_config``:   packaged defaults/env.yaml          ← yaml_path ← overrides
- ``load_train_config``: packaged defaults/train/default.yaml ← yaml_path ← overrides
- ``load_algo_config`` 对 DQN 族 (iql / qmix / vdn / qplex):
      defaults/train/default.yaml ← defaults/algo/<name>.yaml ← yaml_path ← overrides
- ``load_algo_config("mappo")``:
      defaults/algo/mappo.yaml ← yaml_path ← overrides  （不合并 train/default）

``yaml_path`` 始终是「在 packaged 默认上叠加的覆盖文件」，**不是完整替代**：
缺字段从下层默认补齐，超额字段触发 ``unknown key`` 报错。

类型校验在 dataclass 实例化**之前**对 merged dict 完成，规则见 ``_check_type``。
"""

from __future__ import annotations

import hashlib
import warnings
from copy import deepcopy
from dataclasses import is_dataclass
from math import perm
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from src.config.schema import (
    ALGO_CONFIG_TYPES,
    AlgoConfig,
    EnvConfig,
    ExperimentPreset,
    QMIXConfig,
    QPLEXConfig,
    TrainConfig,
)


# 包内 defaults 路径定位。禁止 parents[N] 推断（详见 REFACTOR.md 工程约束）。
_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_ENV_YAML = _PACKAGE_DIR / "defaults" / "env.yaml"
_DEFAULT_TRAIN_YAML = _PACKAGE_DIR / "defaults" / "train" / "default.yaml"
_DEFAULT_ALGO_DIR = _PACKAGE_DIR / "defaults" / "algo"
_DEFAULT_PRESET_DIR = _PACKAGE_DIR / "defaults" / "presets"

_VALID_QMIX_CALLBACKS = {
    "policy_mobility",
    "value_expansion",
    "wm_concurrent",
    "wm_block_alternating",
    "jammer_prediction",
    "critic_stable",
}
# Stage 7：旧名 alias。loader 在合并 YAML/override 后会把 list 里的旧名静默映射到新名并 warn。
_QMIX_CALLBACK_ALIASES = {
    "wm_alternating": "wm_concurrent",
}
# wm 训练范式 callback 集合：value_expansion 必须配其一；二者互斥。
_WM_TRAINING_CALLBACKS = {"wm_concurrent", "wm_block_alternating"}


# ----------------------- YAML 与 dict 工具 -----------------------

def _load_yaml(path: Path) -> dict:
    """读取 YAML 并要求顶层是 mapping。空文件视作空 dict。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"yaml not found: {p}")
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"yaml top-level must be a mapping: {p}")
    return dict(loaded)


def _deep_merge(base: dict, override: dict) -> dict:
    """以 override 的值替换 base 的同名 key（浅合并 —— 当前配置无嵌套字段）。"""
    merged = deepcopy(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def _resolve_preset_path(name_or_path: str | Path) -> Path:
    """Resolve a preset name/path with no fallback between the two modes."""
    raw = str(name_or_path)
    if not raw:
        raise ValueError("preset name/path must not be empty")
    is_builtin_name = not any(ch in raw for ch in ("/", "\\", "."))
    if is_builtin_name:
        return (_DEFAULT_PRESET_DIR / f"{raw}.yaml").resolve()
    return Path(raw).expanduser().resolve()


def load_experiment_preset(name_or_path: str | Path) -> ExperimentPreset:
    """Load a baseline experiment preset.

    Built-in names are strings without ``/``, ``\\`` or ``.`` and resolve to
    ``config/defaults/presets/{name}.yaml``. Anything else is treated as a
    filesystem path relative to the current working directory.
    """
    path = _resolve_preset_path(name_or_path)
    raw = _load_yaml(path)
    allowed = {"algorithm", "description", "source", "env", "algo"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"ExperimentPreset: unknown top-level key(s): {sorted(unknown)}")
    missing = allowed - set(raw)
    if missing:
        raise ValueError(f"ExperimentPreset: missing required key(s): {sorted(missing)}")
    if not isinstance(raw["algorithm"], str):
        raise TypeError("ExperimentPreset.algorithm: expected str")
    if not isinstance(raw["description"], str):
        raise TypeError("ExperimentPreset.description: expected str")
    if not isinstance(raw["source"], str):
        raise TypeError("ExperimentPreset.source: expected str")
    if not isinstance(raw["env"], dict):
        raise TypeError("ExperimentPreset.env: expected mapping")
    if not isinstance(raw["algo"], dict):
        raise TypeError("ExperimentPreset.algo: expected mapping")
    return ExperimentPreset(
        algorithm=str(raw["algorithm"]).lower(),
        description=str(raw["description"]),
        source=str(raw["source"]),
        env=dict(raw["env"]),
        algo=dict(raw["algo"]),
        path=path,
        sha256=config_sha256(path),
    )


# ----------------------- 类型校验 -----------------------

def _check_type(value: Any, annotation: Any, field_name: str) -> Any:
    """按 annotation 校验/转换单字段值，返回最终落定值。

    规则：

    - ``Optional[X]`` / ``X | None``：接受 ``None`` 或按 X 校验。
    - ``list[X]``：接受 list 且每个元素按 X 校验。
    - ``int``：仅接受 ``type(v) is int``（剔除 bool，因为 ``isinstance(True, int)``）。
    - ``float``：接受 int（自动 cast）或 float；**拒绝 bool**。
    - ``bool``：仅 ``type(v) is bool``，不接受 0/1。
    - ``str``：``isinstance(v, str)``。
    """
    args = get_args(annotation)

    # Optional[X] / X | None：args 中含 NoneType。
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _check_type(value, non_none[0], field_name)
        raise TypeError(
            f"{field_name}: unsupported Union annotation {annotation!r}; "
            f"only Optional[X] / X | None is supported"
        )

    origin = get_origin(annotation)

    # list[X]
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(
                f"{field_name}: expected list, got {type(value).__name__} ({value!r})"
            )
        inner = args[0] if args else Any
        return [_check_type(v, inner, f"{field_name}[{i}]") for i, v in enumerate(value)]

    # Any 占位
    if annotation is Any:
        return value

    # 基本类型
    if annotation is int:
        if type(value) is int:  # 剔除 bool（bool 是 int 子类）
            return value
        raise TypeError(
            f"{field_name}: expected int, got {type(value).__name__} ({value!r})"
        )
    if annotation is float:
        if type(value) is bool:
            raise TypeError(f"{field_name}: expected float, got bool ({value!r})")
        if type(value) is int:
            return float(value)
        if type(value) is float:
            return value
        raise TypeError(
            f"{field_name}: expected float, got {type(value).__name__} ({value!r})"
        )
    if annotation is bool:
        if type(value) is bool:
            return value
        raise TypeError(
            f"{field_name}: expected bool, got {type(value).__name__} ({value!r})"
        )
    if annotation is str:
        if isinstance(value, str):
            return value
        raise TypeError(
            f"{field_name}: expected str, got {type(value).__name__} ({value!r})"
        )

    # 未支持的注解 —— 当前 schema 不会走到这里。
    raise TypeError(f"{field_name}: unsupported annotation {annotation!r}")


def _validate_complete(merged: dict, dataclass_type: type) -> dict:
    """对 merged dict 做 unknown / missing / type 校验。

    在 ``dataclass_type(**merged)`` 实例化**之前**调用。返回的 dict 已是类型落定后的
    版本（int → float cast 等已完成），可直接 ``**`` 给 dataclass。
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(
            f"_validate_complete needs a dataclass type, got {dataclass_type!r}"
        )

    hints = get_type_hints(dataclass_type)
    declared = set(hints.keys())
    provided = set(merged.keys())

    unknown = provided - declared
    if unknown:
        raise ValueError(
            f"{dataclass_type.__name__}: unknown key(s): {sorted(unknown)}"
        )

    missing = declared - provided
    if missing:
        raise ValueError(
            f"{dataclass_type.__name__}: missing required key(s): {sorted(missing)}"
        )

    settled: dict[str, Any] = {}
    for name in declared:
        settled[name] = _check_type(merged[name], hints[name], name)
    return settled


# ----------------------- env 配置 -----------------------

_VALID_UAV_MOBILITY = ("gauss_markov", "policy")
_VALID_JAMMER_MOBILITY = ("gauss_markov", "uav_guided_markov")


def _validate_env_cross_field(merged: dict) -> None:
    """跨字段约束。

    - ``channel_loss_db`` 长度必须等于 ``n_channel``。
    - ``fast_fading_db_clip_low`` / ``fast_fading_db_clip_high`` 必须**成对**出现：
      要么都为 ``None``（关闭剪切），要么都为数值且 ``low < high``。
    - **Stage 3.5 校验**（详见 plan locked decisions #1/#2/#3）：
      - ``uav_mobility_control ∈ {"gauss_markov", "policy"}``；
        ``policy`` 模式下 3 个 ``uav_*_delta_max`` 必须 > 0。
      - ``jammer_mobility_model ∈ {"gauss_markov", "uav_guided_markov"}``；
        ``0.0 <= jammer_guidance_strength <= 1.0``（任何模式下成立；默认 0 在
        ``uav_guided_markov`` 下严格退化为 GaussMarkov）。
      - ``is_jammer_moving`` 必须为 ``True``：False 会被 loader 拒绝并提示用
        ``jammer_mobility_model`` 控制运动模型。
    """
    n_channel = merged.get("n_channel")
    n_jammer = merged.get("n_jammer")
    loss = merged.get("channel_loss_db")
    if isinstance(loss, list) and isinstance(n_channel, int) and len(loss) != n_channel:
        raise ValueError(
            f"channel_loss_db: length {len(loss)} must equal n_channel ({n_channel})"
        )

    if (
        type(n_channel) is int
        and type(n_jammer) is int
        and "p_trans_preferred_next_states" in merged
    ):
        jammer_state_dim = int(perm(int(n_channel), int(n_jammer)))
        preferred = merged["p_trans_preferred_next_states"]
        if type(preferred) is int and not (0 <= preferred <= jammer_state_dim):
            raise ValueError(
                "p_trans_preferred_next_states must be in [0, jammer_state_dim], got "
                f"{preferred!r} (jammer_state_dim={jammer_state_dim})"
            )

    if "p_trans_preference_strength" in merged:
        strength = merged["p_trans_preference_strength"]
        if (
            isinstance(strength, (int, float))
            and not isinstance(strength, bool)
            and float(strength) < 0.0
        ):
            raise ValueError(
                f"p_trans_preference_strength must be non-negative; got {strength!r}"
            )

    # 快衰落剪切区间的成对语义。注意 bool 是 int 子类，需排除。
    if "fast_fading_db_clip_low" in merged and "fast_fading_db_clip_high" in merged:
        low = merged["fast_fading_db_clip_low"]
        high = merged["fast_fading_db_clip_high"]

        def _is_number(x: object) -> bool:
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        if (low is None) ^ (high is None):
            raise ValueError(
                "fast_fading_db_clip_low / fast_fading_db_clip_high must both be null "
                f"or both be numeric; got low={low!r}, high={high!r}"
            )
        if _is_number(low) and _is_number(high) and not (float(low) < float(high)):
            raise ValueError(
                "fast_fading_db_clip_low must be strictly less than "
                f"fast_fading_db_clip_high; got low={low!r}, high={high!r}"
            )

    # is_jammer_moving 必须为 True（Stage 3.5 起干扰机始终运动）
    if "is_jammer_moving" in merged and merged["is_jammer_moving"] is False:
        raise ValueError(
            "is_jammer_moving=False is no longer supported; jammers always move under "
            "Stage 3.5. Use jammer_mobility_model='gauss_markov' or 'uav_guided_markov' "
            "to choose the motion model instead."
        )

    # UAV mobility 模式枚举与 policy 模式 delta_max 校验
    if "uav_mobility_control" in merged:
        mode = merged["uav_mobility_control"]
        if mode not in _VALID_UAV_MOBILITY:
            raise ValueError(
                f"uav_mobility_control must be one of {list(_VALID_UAV_MOBILITY)}; "
                f"got {mode!r}"
            )
        if mode == "policy":
            for name in ("uav_velocity_delta_max", "uav_direction_delta_max", "uav_p_delta_max"):
                v = merged.get(name)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) <= 0.0:
                    raise ValueError(
                        f"{name} must be > 0 when uav_mobility_control='policy'; got {v!r}"
                    )

    # Jammer mobility 模式枚举
    if "jammer_mobility_model" in merged:
        mode = merged["jammer_mobility_model"]
        if mode not in _VALID_JAMMER_MOBILITY:
            raise ValueError(
                f"jammer_mobility_model must be one of {list(_VALID_JAMMER_MOBILITY)}; "
                f"got {mode!r}"
            )

    # jammer_guidance_strength 任何模式下都必须 ∈ [0, 1]
    if "jammer_guidance_strength" in merged:
        g = merged["jammer_guidance_strength"]
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not (0.0 <= float(g) <= 1.0):
            raise ValueError(
                f"jammer_guidance_strength must be in [0.0, 1.0]; got {g!r}"
            )


def load_env_config(
    yaml_path: Path | None = None,
    overrides: dict | None = None,
) -> EnvConfig:
    """加载环境配置。

    合并顺序：``packaged defaults/env.yaml ← yaml_path ← overrides``。
    ``yaml_path`` 可只含部分字段，缺字段由 packaged defaults 补齐。
    """
    merged = _load_yaml(_DEFAULT_ENV_YAML)
    if yaml_path is not None:
        merged = _deep_merge(merged, _load_yaml(Path(yaml_path)))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))

    _validate_env_cross_field(merged)
    settled = _validate_complete(merged, EnvConfig)
    return EnvConfig(**settled)


# ----------------------- train 配置 -----------------------

def load_train_config(
    yaml_path: Path | None = None,
    overrides: dict | None = None,
) -> TrainConfig:
    """加载共享训练循环配置。合并顺序：train/default.yaml ← yaml_path ← overrides。"""
    merged = _load_yaml(_DEFAULT_TRAIN_YAML)
    if yaml_path is not None:
        merged = _deep_merge(merged, _load_yaml(Path(yaml_path)))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))
    settled = _validate_complete(merged, TrainConfig)
    return TrainConfig(**settled)


# ----------------------- algo 配置 -----------------------

def _settle_lr_mixer(merged: dict, dataclass_type: type) -> None:
    """QMIX / QPLEX 若 ``lr_mixer is None``，落定为 ``lr_q`` 的值。原地修改。

    VDN sum mixer 没有可学习参数，schema/YAML 都不含 lr_mixer 字段，故不参与此规则。
    下游 trainer 看到 ``lr_mixer`` 时已是确定 float，不需再 fallback。
    """
    if dataclass_type not in (QMIXConfig, QPLEXConfig):
        return
    if merged.get("lr_mixer") is None:
        merged["lr_mixer"] = merged["lr_q"]


def _settle_lr_jammer(merged: dict, dataclass_type: type) -> None:
    """Stage 8：QMIX 若 ``lr_jammer is None``，落定为 ``lr_q``。原地修改。"""
    if dataclass_type is not QMIXConfig:
        return
    if merged.get("lr_jammer") is None:
        merged["lr_jammer"] = merged["lr_q"]


def _canonicalize_callback_names_in_place(merged: dict) -> None:
    """Stage 7：在构造 frozen QMIXConfig dataclass 前把 callback list 的旧名映射到新名。

    检测到旧名（如 ``wm_alternating``）时发出 ``FutureWarning`` 并替换为新名。
    """
    raw = merged.get("callbacks")
    if not isinstance(raw, list):
        return
    new_list: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            new_list.append(item)
            continue
        if item in _QMIX_CALLBACK_ALIASES:
            target = _QMIX_CALLBACK_ALIASES[item]
            warnings.warn(
                f"callback name {item!r} is deprecated; use {target!r}",
                FutureWarning,
                stacklevel=3,
            )
            new_list.append(target)
        else:
            new_list.append(item)
    merged["callbacks"] = new_list


def _validate_qmix_callbacks(cfg: QMIXConfig, env_cfg: EnvConfig | None = None) -> None:
    callbacks = list(cfg.callbacks)
    unknown = sorted(set(callbacks) - _VALID_QMIX_CALLBACKS)
    if unknown:
        raise ValueError(
            "QMIX callbacks contains unknown callback(s): "
            f"{unknown}. Valid callbacks: {sorted(_VALID_QMIX_CALLBACKS)}"
        )
    if len(callbacks) != len(set(callbacks)):
        raise ValueError(f"QMIX callbacks must not contain duplicates: {callbacks!r}")
    # Stage 7：双向约束
    cb_set = set(callbacks)
    wm_present = cb_set & _WM_TRAINING_CALLBACKS
    if len(wm_present) > 1:
        raise ValueError(
            f"callbacks {sorted(wm_present)} are mutually exclusive; pick at most one of "
            f"{sorted(_WM_TRAINING_CALLBACKS)}"
        )
    if wm_present and "value_expansion" not in cb_set:
        raise ValueError(
            f"{next(iter(wm_present))!r} callback requires 'value_expansion'"
        )
    if "value_expansion" in cb_set and not wm_present:
        raise ValueError(
            "value_expansion callback requires exactly one of "
            f"{sorted(_WM_TRAINING_CALLBACKS)}"
        )
    if not (0.0 <= float(cfg.value_expansion_alpha_model) <= 1.0):
        raise ValueError(
            "value_expansion_alpha_model must be in [0.0, 1.0]; "
            f"got {cfg.value_expansion_alpha_model!r}"
        )
    if int(cfg.value_expansion_seq_len) <= 0:
        raise ValueError(
            f"value_expansion_seq_len must be positive; got {cfg.value_expansion_seq_len!r}"
        )
    if not (0.0 <= float(cfg.value_expansion_td_lambda) <= 1.0):
        raise ValueError(
            "value_expansion_td_lambda must be in [0.0, 1.0]; "
            f"got {cfg.value_expansion_td_lambda!r}"
        )
    if int(cfg.value_expansion_rollout_k) <= 0:
        raise ValueError(
            f"value_expansion_rollout_k must be positive; got {cfg.value_expansion_rollout_k!r}"
        )
    # Stage 7：WM 训练超参 + L_VC ramp 范围校验
    if int(cfg.wm_block_qmix_episodes) <= 0:
        raise ValueError(
            f"wm_block_qmix_episodes must be positive; got {cfg.wm_block_qmix_episodes!r}"
        )
    if int(cfg.wm_block_wm_episodes) <= 0:
        raise ValueError(
            f"wm_block_wm_episodes must be positive; got {cfg.wm_block_wm_episodes!r}"
        )
    if int(cfg.wm_batch_size) <= 0:
        raise ValueError(f"wm_batch_size must be positive; got {cfg.wm_batch_size!r}")
    if int(cfg.wm_updates_per_learn) <= 0:
        raise ValueError(
            f"wm_updates_per_learn must be positive; got {cfg.wm_updates_per_learn!r}"
        )
    if float(cfg.wm_vc_eta_max) < 0.0:
        raise ValueError(f"wm_vc_eta_max must be >= 0; got {cfg.wm_vc_eta_max!r}")
    if int(cfg.wm_vc_warmup_ep) < 0:
        raise ValueError(f"wm_vc_warmup_ep must be >= 0; got {cfg.wm_vc_warmup_ep!r}")
    if int(cfg.wm_vc_ramp_end_ep) < int(cfg.wm_vc_warmup_ep):
        raise ValueError(
            "wm_vc_ramp_end_ep must be >= wm_vc_warmup_ep; "
            f"got ramp_end={cfg.wm_vc_ramp_end_ep!r}, warmup={cfg.wm_vc_warmup_ep!r}"
        )
    if int(cfg.wm_buffer_capacity) <= 0:
        raise ValueError(
            f"wm_buffer_capacity must be positive; got {cfg.wm_buffer_capacity!r}"
        )
    if int(cfg.wm_hidden_dim) <= 0:
        raise ValueError(f"wm_hidden_dim must be positive; got {cfg.wm_hidden_dim!r}")
    if int(cfg.wm_n_layers) <= 0:
        raise ValueError(f"wm_n_layers must be positive; got {cfg.wm_n_layers!r}")
    if int(cfg.wm_stochastic_dim) <= 0:
        raise ValueError(
            f"wm_stochastic_dim must be positive; got {cfg.wm_stochastic_dim!r}"
        )
    if float(cfg.wm_kl_beta) < 0.0:
        raise ValueError(f"wm_kl_beta must be >= 0; got {cfg.wm_kl_beta!r}")
    if float(cfg.wm_free_nats) < 0.0:
        raise ValueError(f"wm_free_nats must be >= 0; got {cfg.wm_free_nats!r}")
    if float(cfg.wm_lr) <= 0.0:
        raise ValueError(f"wm_lr must be positive; got {cfg.wm_lr!r}")
    if float(cfg.wm_max_grad_norm) < 0.0:
        raise ValueError(
            f"wm_max_grad_norm must be >= 0; got {cfg.wm_max_grad_norm!r}"
        )
    # Stage 8：JP 字段范围校验
    if int(cfg.jammer_history_len) <= 0:
        raise ValueError(
            f"jammer_history_len must be positive; got {cfg.jammer_history_len!r}"
        )
    if int(cfg.jammer_pred_hidden_dim) <= 0:
        raise ValueError(
            f"jammer_pred_hidden_dim must be positive; got {cfg.jammer_pred_hidden_dim!r}"
        )
    if float(cfg.jammer_aux_weight) < 0.0:
        raise ValueError(
            f"jammer_aux_weight must be >= 0; got {cfg.jammer_aux_weight!r}"
        )
    if int(cfg.jammer_warmup_episodes) < 0:
        raise ValueError(
            f"jammer_warmup_episodes must be >= 0; got {cfg.jammer_warmup_episodes!r}"
        )
    if cfg.lr_jammer is None or float(cfg.lr_jammer) <= 0.0:
        raise ValueError(
            f"lr_jammer must be > 0 (None should have been settled by loader); got {cfg.lr_jammer!r}"
        )
    if not (0.0 <= float(cfg.critic_stable_tau) <= 1.0):
        raise ValueError(
            f"critic_stable_tau must be in [0.0, 1.0]; got {cfg.critic_stable_tau!r}"
        )
    if float(cfg.critic_stable_lr_scale) <= 0.0:
        raise ValueError(
            f"critic_stable_lr_scale must be positive; got {cfg.critic_stable_lr_scale!r}"
        )
    if (
        env_cfg is not None
        and env_cfg.uav_mobility_control == "policy"
        and "policy_mobility" not in callbacks
    ):
        raise ValueError(
            "env.uav_mobility_control='policy' requires QMIX callback "
            "'policy_mobility'. Add callbacks: ['policy_mobility', ...]."
        )


def load_algo_config(
    name: str,
    yaml_path: Path | None = None,
    overrides: dict | None = None,
    env_cfg: EnvConfig | None = None,
) -> AlgoConfig:
    """加载算法配置。

    DQN 族 (iql / qmix / vdn / qplex):
        ``defaults/train/default.yaml ← defaults/algo/<name>.yaml ← yaml_path ← overrides``

    MAPPO:
        ``defaults/algo/mappo.yaml ← yaml_path ← overrides``（**不**合并 train/default）
    """
    if name not in ALGO_CONFIG_TYPES:
        raise ValueError(
            f"unknown algo name {name!r}; valid: {sorted(ALGO_CONFIG_TYPES)}"
        )
    dc_type = ALGO_CONFIG_TYPES[name]
    algo_yaml = _DEFAULT_ALGO_DIR / f"{name}.yaml"

    if name == "mappo":
        merged = _load_yaml(algo_yaml)
    else:
        merged = _load_yaml(_DEFAULT_TRAIN_YAML)
        merged = _deep_merge(merged, _load_yaml(algo_yaml))

    if yaml_path is not None:
        merged = _deep_merge(merged, _load_yaml(Path(yaml_path)))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))

    _settle_lr_mixer(merged, dc_type)
    _settle_lr_jammer(merged, dc_type)
    if dc_type is QMIXConfig:
        _canonicalize_callback_names_in_place(merged)
    settled = _validate_complete(merged, dc_type)
    cfg = dc_type(**settled)
    if isinstance(cfg, QMIXConfig):
        _validate_qmix_callbacks(cfg, env_cfg=env_cfg)
    return cfg


# ----------------------- 哈希与 run summary -----------------------

def config_sha256(yaml_path: Path) -> str:
    """文件级 SHA256，分块读取以兼容大文件。"""
    p = Path(yaml_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"yaml not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# 与旧 ``Main/common.py:env_run_config`` 对齐的字段集合。
# 区分两部分：23 个直接来自 EnvConfig，5 个从 specs.py 派生。
_RUN_SUMMARY_KEYS_FROM_CFG: tuple[str, ...] = (
    "n_ch", "n_jammer", "n_channel", "env_seed",
    "data_size", "t_Rx", "jammer_power", "max_episode_steps",
    "p_trans_seed", "p_trans_preferred_next_states", "p_trans_preference_strength",
    "jammer_reactive_beta", "jammer_memory_window", "jammer_reactive_observe_prob",
    "uav_interference_scale", "reward_energy_weight", "reward_jump_weight",
    "fairness_min_success_rate", "fairness_weight", "enable_fast_fading",
    "fast_fading_rho", "csi_noise_std", "sensing_noise_std",
)

_RUN_SUMMARY_KEYS_FROM_SPECS: tuple[str, ...] = (
    "n_des", "state_dim", "action_dim", "param_dim_per_action", "total_param_dim",
)


def env_run_summary(
    cfg: EnvConfig,
    yaml_path: Path,
    overrides: dict | None = None,
) -> dict:
    """生成与旧 ``env_run_config`` 等价的 run-time 元信息。

    新增 ``overrides`` 字段保留运行时覆盖信息，避免哈希仅代表 base YAML、
    无法溯源。
    """
    # 延迟 import 防止 loader -> specs -> schema -> loader 这种潜在环路。
    from src.config import specs

    summary: dict[str, Any] = {}
    for key in _RUN_SUMMARY_KEYS_FROM_CFG:
        summary[key] = getattr(cfg, key)
    for key in _RUN_SUMMARY_KEYS_FROM_SPECS:
        summary[key] = getattr(specs, key)(cfg)

    return {
        "env_config_path": str(Path(yaml_path).resolve()),
        "env_config_sha256": config_sha256(yaml_path),
        "overrides": dict(overrides) if overrides else None,
        "env_summary": summary,
    }


__all__ = [
    "load_env_config",
    "load_experiment_preset",
    "load_train_config",
    "load_algo_config",
    "config_sha256",
    "env_run_summary",
    # 暴露给测试，方便单独验证 missing key 等逻辑。
    "_validate_complete",
]
