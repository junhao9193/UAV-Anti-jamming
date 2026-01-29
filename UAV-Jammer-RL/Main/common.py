from __future__ import annotations

import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

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
    """Save metrics to `Draw/experiment-data/{algorithm}_{timestamp}/` under repo root (json + npz + png)."""
    repo_root = get_repo_root()
    base_dir = repo_root / "Draw" / "experiment-data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = base_dir / f"{algorithm}_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "training_data.json"
    npz_path = data_dir / "training_data.npz"
    png_path = data_dir / "training_metrics.png"

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

def _env_worker(remote, parent_remote, config_path: Optional[str], p_trans: Optional[np.ndarray]) -> None:
    """
    Subprocess worker for environment stepping.

    Important: keep imports inside the worker so that starting many workers does not import torch/CUDA.
    """
    import os

    # Avoid BLAS thread oversubscription when using many workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    parent_remote.close()
    try:
        from envs import Environ

        env = Environ(config_path=config_path) if config_path else Environ()
        if p_trans is not None:
            env.set_p(p_trans)

        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                if data is not None:
                    env.set_p(data)
                env.new_random_game()
                env.clear_reward()
                state = env.get_state()
                remote.send(np.stack(state, axis=0).astype(np.float32))
            elif cmd == "step":
                next_state, reward, done, info = env.step(data)
                remote.send(
                    (
                        np.stack(next_state, axis=0).astype(np.float32),
                        np.asarray(reward, dtype=np.float32),
                        bool(done),
                        info,
                    )
                )
            elif cmd == "metrics":
                remote.send((float(env.rew_energy), float(env.rew_jump), float(env.rew_suc)))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd!r}")
    except KeyboardInterrupt:
        pass


class SubprocVecEnv:
    """A minimal subprocess vectorized env wrapper for throughput (sync step)."""

    def __init__(
        self,
        n_envs: int,
        *,
        config_path: Optional[str] = None,
        p_trans: Optional[np.ndarray] = None,
        start_method: str = "spawn",
    ) -> None:
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.ps = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = ctx.Process(target=_env_worker, args=(work_remote, remote, config_path, p_trans), daemon=True)
            p.start()
            self.ps.append(p)
            work_remote.close()

    def reset(self, p_trans: Optional[np.ndarray] = None) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", p_trans))
        states = [remote.recv() for remote in self.remotes]
        return np.stack(states, axis=0).astype(np.float32)  # (E, N, S)

    def step_async(self, actions: Sequence[Any]) -> None:
        if len(actions) != self.n_envs:
            raise ValueError(f"Expected actions for {self.n_envs} envs, got {len(actions)}")
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", act))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, infos = zip(*results)
        return (
            np.stack(next_states, axis=0).astype(np.float32),  # (E,N,S)
            np.stack(rewards, axis=0).astype(np.float32),  # (E,N)
            np.asarray(dones, dtype=np.bool_),  # (E,)
            list(infos),
        )

    def step(self, actions: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        self.step_async(actions)
        return self.step_wait()

    def get_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for remote in self.remotes:
            remote.send(("metrics", None))
        energy, jump, suc = zip(*(remote.recv() for remote in self.remotes))
        return (
            np.asarray(energy, dtype=np.float32),
            np.asarray(jump, dtype=np.float32),
            np.asarray(suc, dtype=np.float32),
        )

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass


__all__ = ["get_repo_root", "save_training_data", "make_fixed_p_trans", "SubprocVecEnv"]
