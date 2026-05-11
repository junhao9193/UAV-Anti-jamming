from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


CANONICAL_ENV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "env.yaml"


def _load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"env config file not found: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"env config must be a mapping/dict: {path}")
    return dict(loaded)


# Single source of truth for default environment settings.
# Edit `UAV-Jammer-RL/configs/env.yaml` for difficulty knobs, communication
# workload, jammer behavior, sensing noise, and reward weights. This constant is
# loaded from that YAML for compatibility with code that imports it.
DEFAULT_ENV_CONFIG: Dict[str, Any] = _load_yaml_config(CANONICAL_ENV_CONFIG_PATH)


def load_env_config(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(DEFAULT_ENV_CONFIG)

    if config_path is not None:
        merged.update(_load_yaml_config(config_path))

    if config:
        merged.update(config)

    return merged
