"""Training and evaluation artifact writers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def get_project_root() -> Path:
    """Return the UAV-Ultra project root that owns ``runs/experiment-data``."""
    return Path(__file__).absolute().parents[2]


def get_repo_root() -> Path:
    """Return the shared repository root.

    Kept for compatibility with callers that imported the old helper.
    New artifacts are written under :func:`get_project_root`.
    """
    return Path(__file__).absolute().parents[3]


def default_output_root() -> Path:
    return get_project_root() / "runs" / "experiment-data"


def make_unique_output_dir(base_dir: Path, prefix: str) -> Path:
    base_dir = Path(base_dir)
    prefix = str(prefix)
    pattern = re.compile(rf"^{re.escape(prefix)}_exp(\d+)$")
    max_seen = 0
    if base_dir.exists():
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match is not None:
                max_seen = max(max_seen, int(match.group(1)))

    for exp_num in range(max_seen + 1, max_seen + 1025):
        out_dir = base_dir / f"{prefix}_exp{exp_num}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Failed to create unique output directory under {base_dir}")


def reserve_output_dir(algorithm: str, output_root: Path | None = None) -> Path:
    base_dir = default_output_root() if output_root is None else Path(output_root)
    return make_unique_output_dir(base_dir, str(algorithm))


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
    data_dir: Path | None = None,
    artifact_kind: str = "train",
) -> tuple[Path, Path, Path]:
    """Save training/evaluation metrics using the baseline field schema."""
    artifact_kind = str(artifact_kind).lower()
    if artifact_kind not in {"train", "eval"}:
        raise ValueError(f"artifact_kind must be 'train' or 'eval', got {artifact_kind!r}")
    stem = "evaluation" if artifact_kind == "eval" else "training"

    if data_dir is None:
        data_dir = reserve_output_dir(str(algorithm), output_root=output_root)
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    run_id = data_dir.name.removeprefix(f"{algorithm}_")

    config_data: dict[str, Any] = {
        "n_episode": int(n_episode),
        "n_steps": int(n_steps),
        "artifact_kind": artifact_kind,
    }
    if run_config is not None:
        config_data.update(_json_safe(run_config))

    data = {
        "algorithm": str(algorithm),
        # Keep the legacy top-level key name for downstream readers; the value
        # is now an experiment id such as ``exp1`` rather than a wall-clock time.
        "timestamp": run_id,
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
    "get_project_root",
    "get_repo_root",
    "make_unique_output_dir",
    "reserve_output_dir",
    "save_training_data",
]
