"""类型化配置层。

依赖图位置：``config →（无依赖）``。本包不 import 任何 ``src.envs`` /
``src.algorithms`` / ``torch``，是整个项目所有上层模块的配置入口。

公开 API：

- ``load_env_config(yaml_path=None, overrides=None) -> EnvConfig``
- ``load_train_config(yaml_path=None, overrides=None) -> TrainConfig``
- ``load_algo_config(name, yaml_path=None, overrides=None) -> AlgoConfig``
- ``config_sha256(yaml_path)``
- ``env_run_summary(cfg, yaml_path, overrides=None)``
- ``specs`` 子模块（纯函数维度公式）
- ``ALGO_CONFIG_TYPES`` 注册表
"""

from src.config import specs
from src.config.loader import (
    config_sha256,
    env_run_summary,
    load_algo_config,
    load_env_config,
    load_experiment_preset,
    load_train_config,
)
from src.config.schema import (
    ALGO_CONFIG_TYPES,
    AlgoConfig,
    EnvConfig,
    ExperimentPreset,
    IQLConfig,
    MAPPOConfig,
    QMIXConfig,
    QPLEXConfig,
    TrainConfig,
    VDNConfig,
)

__all__ = [
    "specs",
    "load_env_config",
    "load_experiment_preset",
    "load_train_config",
    "load_algo_config",
    "config_sha256",
    "env_run_summary",
    "EnvConfig",
    "ExperimentPreset",
    "TrainConfig",
    "IQLConfig",
    "QMIXConfig",
    "VDNConfig",
    "QPLEXConfig",
    "MAPPOConfig",
    "AlgoConfig",
    "ALGO_CONFIG_TYPES",
]
