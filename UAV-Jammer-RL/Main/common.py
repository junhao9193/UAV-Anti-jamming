from __future__ import annotations

import hashlib
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


def get_project_root() -> Path:
    """Return the `UAV-Jammer-RL/` project root."""
    return Path(__file__).absolute().parents[1]


def resolve_env_config_path(config_path: Optional[str] = None) -> Optional[str]:
    """Return an absolute env config path, defaulting to `configs/env.yaml` when present."""
    if config_path is None:
        default_path = get_project_root() / "configs" / "env.yaml"
        return str(default_path.absolute()) if default_path.exists() else None

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).absolute()
    if not path.exists():
        raise FileNotFoundError(f"env config file not found: {path}")
    return str(path)


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_run_config(env: Any, config_path: Optional[str] = None) -> dict:
    """Small, stable metadata snapshot for comparing experiments across env configs."""
    resolved = resolve_env_config_path(config_path)
    keys = [
        "n_ch",
        "n_des",
        "n_jammer",
        "n_channel",
        "state_dim",
        "action_dim",
        "param_dim_per_action",
        "total_param_dim",
        "p_trans_mode",
        "p_trans_seed",
        "jammer_modes",
        "jammer_mode_switch_prob",
        "jammer_reactive_beta",
        "uav_interference_scale",
        "reward_energy_weight",
        "reward_jump_weight",
        "fairness_min_success_rate",
        "fairness_weight",
        "enable_fast_fading",
        "fast_fading_rho",
    ]
    summary = {}
    for key in keys:
        if hasattr(env, key):
            val = getattr(env, key)
            if isinstance(val, np.generic):
                val = val.item()
            summary[key] = val
    return {
        "env_config_path": resolved,
        "env_config_sha256": _file_sha256(resolved),
        "env_summary": summary,
    }


def validate_positive_run_args(*, n_episode: int, n_steps: int, num_envs: int) -> None:
    if int(n_episode) <= 0:
        raise ValueError(f"n_episode/episodes must be positive, got {n_episode}")
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps/steps must be positive, got {n_steps}")
    if int(num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")


def make_unique_output_dir(base_dir: Path, prefix: str) -> Path:
    """
    Atomically create a unique experiment directory.

    Uses microsecond-resolution timestamps and falls back to a numeric suffix if a
    same-name directory already exists (for example, concurrent runs started in the
    same microsecond).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(1024):
        suffix = "" if attempt == 0 else f"_{attempt}"
        out_dir = base_dir / f"{prefix}_{timestamp}{suffix}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Failed to create a unique output directory under: {base_dir}")


def save_training_data(
    algorithm: str,
    reward_history,
    success_rate_history,
    energy_history,
    jump_history,
    n_episode: int,
    n_steps: int,
    trainer: Optional[Any] = None,
    run_config: Optional[dict] = None,
    artifact_kind: str = "train",
) -> Tuple[str, str, Path]:
    """Save metrics to `Draw/experiment-data/{algorithm}_{timestamp}/` under repo root (json + npz + png).

    If trainer is provided, also saves the network weights (model parameters).

    Returns (json_path, npz_path, data_dir) so callers can write additional artifacts
    into the same directory without re-scanning the filesystem.
    """
    repo_root = get_repo_root()
    base_dir = repo_root / "Draw" / "experiment-data"

    data_dir = make_unique_output_dir(base_dir, algorithm)
    timestamp = data_dir.name.removeprefix(f"{algorithm}_")

    artifact_kind = str(artifact_kind).lower()
    if artifact_kind not in {"train", "eval"}:
        raise ValueError(f"artifact_kind must be 'train' or 'eval', got {artifact_kind!r}")
    stem = "evaluation" if artifact_kind == "eval" else "training"

    json_path = data_dir / f"{stem}_data.json"
    npz_path = data_dir / f"{stem}_data.npz"
    png_path = data_dir / f"{stem}_metrics.png"

    def _json_safe(x):
        if isinstance(x, dict):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_safe(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        return x

    config_data = {
        "n_episode": int(n_episode),
        "n_steps": int(n_steps),
        "artifact_kind": artifact_kind,
    }
    if run_config is not None:
        config_data.update(_json_safe(run_config))

    data = {
        "algorithm": algorithm,
        "timestamp": timestamp,
        "config": config_data,
        "metrics": {
            "reward": [float(x) for x in reward_history],
            "success_rate": [float(x) for x in success_rate_history],
            "energy": [float(x) for x in energy_history],
            "jump": [float(x) for x in jump_history],
        },
    }

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

    print(("Evaluation" if artifact_kind == "eval" else "Training") + " data saved to:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")

    try:
        _plot_metrics_png(
            reward=np.asarray(reward_history, dtype=np.float32),
            success_rate=np.asarray(success_rate_history, dtype=np.float32),
            algorithm=algorithm,
            title_prefix="Evaluation Metrics" if artifact_kind == "eval" else "Training Metrics",
            save_path=str(png_path),
        )
        print(f"  PNG:  {png_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Save network weights if trainer is provided
    if trainer is not None:
        try:
            weights_path = _save_model_weights(trainer, data_dir, algorithm)
            print(f"  Weights: {weights_path}")
        except Exception as e:
            print(f"Model weights saving skipped: {e}")

    return str(json_path), str(npz_path), data_dir


def _save_model_weights(trainer: Any, data_dir: Path, algorithm: str) -> str:
    """Save network weights (model parameters) for the trainer.
    
    Supports both QMIX trainer (with mixer) and IQL trainer (agents only).
    
    Saved structure:
    - agents: list of agent state dicts, each containing:
        - actor: actor network weights
        - q_net: Q-network weights
        - target_actor: target actor network weights
        - target_q_net: target Q-network weights
    - mixer (QMIX only): mixer network weights
    - target_mixer (QMIX only): target mixer network weights
    - config: architecture configuration for reconstruction
    """
    import torch
    
    weights_path = data_dir / f"{algorithm}_weights.pth"
    
    checkpoint = {"algorithm": algorithm}

    # MP-DQN style trainers/agents: a trainer owns `agents`, each with actor/q/targets.
    agents = getattr(trainer, "agents", None)
    if agents is not None:
        checkpoint["agents"] = []

        for agent in agents:
            agent_state = {
                "actor": agent.actor.state_dict(),
                "q_net": agent.q_net.state_dict(),
                "target_actor": agent.target_actor.state_dict(),
                "target_q_net": agent.target_q_net.state_dict(),
            }
            checkpoint["agents"].append(agent_state)

        if agents:
            first_agent = agents[0]
            checkpoint["agent_config"] = {
                "state_dim": first_agent.state_dim,
                "n_actions": first_agent.n_actions,
                "param_dim": first_agent.param_dim,
            }

        if hasattr(trainer, "mixer") and trainer.mixer is not None:
            mixer = trainer.mixer
            checkpoint["mixer"] = mixer.state_dict()
            if hasattr(trainer, "target_mixer") and trainer.target_mixer is not None:
                checkpoint["target_mixer"] = trainer.target_mixer.state_dict()

            mixer_config = {
                "mixer_class": mixer.__class__.__name__,
                "n_agents": int(getattr(trainer, "n_agents")),
                "global_state_dim": int(getattr(trainer, "global_state_dim")),
            }
            for attr in ("mixing_hidden_dim", "hypernet_hidden_dim", "n_heads"):
                if hasattr(mixer, attr):
                    mixer_config[attr] = int(getattr(mixer, attr))
            checkpoint["mixer_config"] = mixer_config

    # MAPPO shared-parameter agent: actor + critic only.
    elif hasattr(trainer, "actor") and hasattr(trainer, "critic"):
        checkpoint["actor"] = trainer.actor.state_dict()
        checkpoint["critic"] = trainer.critic.state_dict()
        if hasattr(trainer, "actor_opt"):
            checkpoint["actor_opt"] = trainer.actor_opt.state_dict()
        if hasattr(trainer, "critic_opt"):
            checkpoint["critic_opt"] = trainer.critic_opt.state_dict()
        checkpoint["agent_config"] = {
            "obs_dim": int(trainer.obs_dim),
            "n_actions": int(trainer.n_actions),
            "cont_dim": int(trainer.cont_dim),
            "n_agents": int(trainer.n_agents),
            "global_state_dim": int(trainer.global_state_dim),
        }
    else:
        raise ValueError("Unsupported trainer type for weight saving")
    
    torch.save(checkpoint, str(weights_path))
    return str(weights_path)


def _plot_metrics_png(
    reward: np.ndarray,
    success_rate: np.ndarray,
    algorithm: str,
    save_path: str,
    title_prefix: str = "Training Metrics",
) -> None:
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
    fig.suptitle(f"{title_prefix} - {algorithm}", fontsize=12)

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

    rng = np.random.default_rng(seed)
    p_trans = env.generate_p_trans(mode=mode, rng=rng)
    return np.asarray(p_trans, dtype=np.float32)

def _env_worker(remote, parent_remote, config_path: Optional[str], p_trans: Optional[np.ndarray], worker_seed: Optional[int] = None) -> None:
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
        import random as _random
        from envs import Environ

        # Seed per-worker random sources for reproducibility.
        if worker_seed is not None:
            _random.seed(int(worker_seed))
            np.random.seed(int(worker_seed) % (2**31))

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
        seed: Optional[int] = None,
    ) -> None:
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.ps = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            worker_seed = (int(seed) + i * 1000) if seed is not None else None
            p = ctx.Process(target=_env_worker, args=(work_remote, remote, config_path, p_trans, worker_seed), daemon=True)
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


__all__ = [
    "get_repo_root",
    "get_project_root",
    "resolve_env_config_path",
    "env_run_config",
    "validate_positive_run_args",
    "make_unique_output_dir",
    "save_training_data",
    "make_fixed_p_trans",
    "SubprocVecEnv",
]
