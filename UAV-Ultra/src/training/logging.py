"""Baseline-compatible training artifact writers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def get_repo_root() -> Path:
    """Return the shared repository root that owns ``Draw/experiment-data``."""
    return Path(__file__).absolute().parents[3]


def default_output_root() -> Path:
    return get_repo_root() / "Draw" / "experiment-data"


def make_unique_output_dir(base_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(1024):
        suffix = "" if attempt == 0 else f"_{attempt}"
        out_dir = Path(base_dir) / f"{prefix}_{timestamp}{suffix}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Failed to create unique output directory under {base_dir}")


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def save_training_data(
    *,
    algorithm: str,
    reward_history: Sequence[float],
    success_rate_history: Sequence[float],
    energy_history: Sequence[float],
    jump_history: Sequence[float],
    n_episode: int,
    n_steps: int,
    run_config: dict[str, Any] | None = None,
    output_root: Path | None = None,
    artifact_kind: str = "train",
) -> tuple[Path, Path, Path]:
    """Save training/evaluation metrics using the baseline field schema."""
    artifact_kind = str(artifact_kind).lower()
    if artifact_kind not in {"train", "eval"}:
        raise ValueError(f"artifact_kind must be 'train' or 'eval', got {artifact_kind!r}")
    stem = "evaluation" if artifact_kind == "eval" else "training"

    base_dir = default_output_root() if output_root is None else Path(output_root)
    data_dir = make_unique_output_dir(base_dir, str(algorithm))
    timestamp = data_dir.name.removeprefix(f"{algorithm}_")

    config_data: dict[str, Any] = {
        "n_episode": int(n_episode),
        "n_steps": int(n_steps),
        "artifact_kind": artifact_kind,
    }
    if run_config is not None:
        config_data.update(_json_safe(run_config))

    data = {
        "algorithm": str(algorithm),
        "timestamp": timestamp,
        "config": config_data,
        "metrics": {
            "reward": [float(x) for x in reward_history],
            "success_rate": [float(x) for x in success_rate_history],
            "energy": [float(x) for x in energy_history],
            "jump": [float(x) for x in jump_history],
        },
    }

    json_path = data_dir / f"{stem}_data.json"
    npz_path = data_dir / f"{stem}_data.npz"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if run_config is not None:
        (data_dir / "run_config.json").write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    np.savez(
        str(npz_path),
        reward=np.asarray(reward_history, dtype=np.float32),
        success_rate=np.asarray(success_rate_history, dtype=np.float32),
        energy=np.asarray(energy_history, dtype=np.float32),
        jump=np.asarray(jump_history, dtype=np.float32),
    )
    return json_path, npz_path, data_dir


__all__ = [
    "default_output_root",
    "get_repo_root",
    "make_unique_output_dir",
    "save_training_data",
]
