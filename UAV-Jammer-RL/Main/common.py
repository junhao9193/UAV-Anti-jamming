from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np


def get_repo_root() -> Path:
    # .../MetaRL-for-UAV-Anti-jamming/UAV-Jammer-RL/Main/common.py -> repo root is parents[2]
    # NOTE: use `absolute()` (not `resolve()`) to avoid following Windows junctions/symlinks.
    return Path(__file__).absolute().parents[2]


def save_training_data(
    algorithm: str,
    reward_history,
    success_rate_history,
    energy_history,
    jump_history,
    n_episode: int,
    n_steps: int,
) -> Tuple[str, str]:
    """Save metrics to `Draw/experiment-data` under repo root (json + npz + png)."""
    repo_root = get_repo_root()
    data_dir = repo_root / "Draw" / "experiment-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = data_dir / f"{algorithm}_{timestamp}.json"
    npz_path = data_dir / f"{algorithm}_{timestamp}.npz"
    png_path = data_dir / f"{algorithm}_{timestamp}.png"

    data = {
        "algorithm": algorithm,
        "timestamp": timestamp,
        "config": {
            "n_episode": int(n_episode),
            "n_steps": int(n_steps),
        },
        "metrics": {
            "reward": [float(x) for x in reward_history],
            "success_rate": [float(x) for x in success_rate_history],
            "energy": [float(x) for x in energy_history],
            "jump": [float(x) for x in jump_history],
        },
    }

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        str(npz_path),
        reward=np.asarray(reward_history, dtype=np.float32),
        success_rate=np.asarray(success_rate_history, dtype=np.float32),
        energy=np.asarray(energy_history, dtype=np.float32),
        jump=np.asarray(jump_history, dtype=np.float32),
    )

    print("Training data saved to:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")

    try:
        _plot_metrics_png(
            reward=np.asarray(reward_history, dtype=np.float32),
            success_rate=np.asarray(success_rate_history, dtype=np.float32),
            algorithm=algorithm,
            save_path=str(png_path),
        )
        print(f"  PNG:  {png_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return str(json_path), str(npz_path)


def _plot_metrics_png(reward: np.ndarray, success_rate: np.ndarray, algorithm: str, save_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(x: np.ndarray, window: int = 50) -> np.ndarray:
        if window <= 1 or len(x) < window:
            return x
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(x, kernel, mode="valid")
        pad = len(x) - len(smoothed)
        return np.concatenate([x[:pad], smoothed])

    episodes = np.arange(len(reward))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Metrics - {algorithm}", fontsize=12)

    axes[0].plot(episodes, reward, alpha=0.25, color="blue", label="Raw")
    axes[0].plot(episodes, smooth(reward), color="blue", linewidth=2, label="Smoothed")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, success_rate, alpha=0.25, color="green", label="Raw")
    axes[1].plot(episodes, smooth(success_rate), color="green", linewidth=2, label="Smoothed")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Communication Success Rate")
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fixed_p_trans(env) -> np.ndarray:
    """
    Create a fixed Markov transition matrix for jammer hopping.

    - Uses `env.p_trans_seed` / `env.p_trans_mode` when present.
    - Does NOT change global numpy RNG state for the rest of the program.
    """
    mode = int(getattr(env, "p_trans_mode", 1))
    seed = int(getattr(env, "p_trans_seed", 0))

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        p_trans = env.generate_p_trans(mode=mode)
    finally:
        np.random.set_state(rng_state)
    return np.asarray(p_trans, dtype=np.float32)


__all__ = ["get_repo_root", "save_training_data", "make_fixed_p_trans"]
