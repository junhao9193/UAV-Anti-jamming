"""Subprocess vectorized environment wrapper for DQN-family training."""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np


_WORKER_ERROR = "__uav_ultra_vecenv_worker_error__"


def make_fixed_p_trans(env: Any) -> np.ndarray:
    """Create a fixed jammer transition matrix without touching global RNG state."""
    seed = int(getattr(env, "p_trans_seed", 0))
    rng = np.random.default_rng(seed)
    return np.asarray(env.generate_p_trans(rng=rng), dtype=np.float32)


def _spawn_worker_seeds(seed: int | None, n_envs: int) -> list[int | None]:
    """Generate per-worker integer seeds from one root seed."""
    n = int(n_envs)
    if seed is None:
        return [None] * n
    return [
        int(ss.generate_state(1)[0])
        for ss in np.random.SeedSequence(int(seed)).spawn(n)
    ]


def _env_worker(
    remote: Any,
    parent_remote: Any,
    p_trans: Optional[np.ndarray],
    worker_seed: Optional[int],
    env_overrides: Optional[dict[str, Any]],
    config_path: Optional[str],
) -> None:
    import os
    import random as _random

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    parent_remote.close()
    try:
        from src.envs import Environ

        if worker_seed is not None:
            _random.seed(int(worker_seed))
            np.random.seed(int(worker_seed) % (2**31))

        overrides = dict(env_overrides or {})
        if worker_seed is not None:
            overrides["env_seed"] = int(worker_seed)
        env = Environ(config=overrides or None, config_path=config_path)
        if p_trans is not None:
            env.set_p(p_trans)

        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                state = env.reset(p_trans=data if data is not None else p_trans)
                env.clear_reward()
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
                raise RuntimeError(f"Unknown vecenv command: {cmd!r}")
    except KeyboardInterrupt:
        pass
    except Exception:
        message = traceback.format_exc()
        print(message, file=sys.stderr, flush=True)
        try:
            remote.send((_WORKER_ERROR, message))
        except Exception:
            pass


def _raise_if_worker_error(message: Any) -> Any:
    if isinstance(message, tuple) and len(message) == 2 and message[0] == _WORKER_ERROR:
        raise RuntimeError(f"SubprocVecEnv worker failed:\n{message[1]}")
    return message


class SubprocVecEnv:
    """Minimal subprocess vector env with explicit async/wait split."""

    def __init__(
        self,
        n_envs: int,
        *,
        p_trans: Optional[np.ndarray] = None,
        start_method: str = "spawn",
        seed: Optional[int] = None,
        env_overrides: Optional[dict[str, Any]] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.waiting = False
        self.closed = False

        ctx = mp.get_context(str(start_method))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.ps = []
        cfg_path = None if config_path is None else str(Path(config_path))
        worker_seeds = _spawn_worker_seeds(seed, self.n_envs)
        for worker_seed, (work_remote, remote) in zip(worker_seeds, zip(self.work_remotes, self.remotes)):
            proc = ctx.Process(
                target=_env_worker,
                args=(work_remote, remote, p_trans, worker_seed, env_overrides, cfg_path),
                daemon=True,
            )
            proc.start()
            self.ps.append(proc)
            work_remote.close()

    def reset(self, p_trans: Optional[np.ndarray] = None) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", p_trans))
        states = [_raise_if_worker_error(remote.recv()) for remote in self.remotes]
        return np.stack(states, axis=0).astype(np.float32)

    def step_async(self, actions: Sequence[Any]) -> None:
        if self.waiting:
            raise RuntimeError("step_async called while another step is pending")
        if len(actions) != self.n_envs:
            raise ValueError(f"Expected actions for {self.n_envs} envs, got {len(actions)}")
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if not self.waiting:
            raise RuntimeError("step_wait called without a pending step_async")
        results = [_raise_if_worker_error(remote.recv()) for remote in self.remotes]
        self.waiting = False
        next_states, rewards, dones, infos = zip(*results)
        return (
            np.stack(next_states, axis=0).astype(np.float32),
            np.stack(rewards, axis=0).astype(np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
        )

    def step(self, actions: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self.step_async(actions)
        return self.step_wait()

    def get_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for remote in self.remotes:
            remote.send(("metrics", None))
        energy, jump, suc = zip(*(_raise_if_worker_error(remote.recv()) for remote in self.remotes))
        return (
            np.asarray(energy, dtype=np.float32),
            np.asarray(jump, dtype=np.float32),
            np.asarray(suc, dtype=np.float32),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self.ps:
            try:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()
                    proc.join(timeout=1.0)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["SubprocVecEnv", "_spawn_worker_seeds", "make_fixed_p_trans"]
