"""Profile one short MP-DQN training episode by pipeline stage.

Run from ``UAV-Jammer-RL/`` with the project Python environment, for example:

    python -m Main.train.profile_mpdqn_timing --algo qmix --episodes 1 --steps 20

The profiler intentionally mirrors the train_* scripts instead of importing them,
so it can time the inner rollout/update stages without changing trainer code.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from envs import Environ
from Main.common import SubprocVecEnv, get_repo_root, make_fixed_p_trans, resolve_episode_steps


def _configure_torch(torch: Any, device: str) -> None:
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _make_trainer(
    *,
    algo: str,
    env0: Environ,
    batch_size: int,
    buffer_capacity: int,
    lr_actor: float,
    lr_q: float,
    use_amp: bool,
    max_grad_norm: float,
    loss_log_interval: int,
    device: str,
) -> Any:
    if algo == "iql":
        from algorithms.mpdqn.iql.trainer import MPDQNJointIQLTrainer

        return MPDQNJointIQLTrainer(
            n_agents=int(env0.n_ch),
            state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            buffer_capacity=int(buffer_capacity),
            batch_size=int(batch_size),
            lr_actor=float(lr_actor),
            lr_q=float(lr_q),
            use_amp=bool(use_amp),
            max_grad_norm=float(max_grad_norm),
            loss_log_interval=int(loss_log_interval),
            device=device,
        )
    if algo == "qmix":
        from algorithms.mpdqn.qmix.trainer_greedy_actor import MPDQNQMIXTrainer

        return MPDQNQMIXTrainer(
            n_agents=int(env0.n_ch),
            state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            global_state_dim=int(env0.state_dim * env0.n_ch),
            buffer_capacity=int(buffer_capacity),
            batch_size=int(batch_size),
            lr_actor=float(lr_actor),
            lr_q=float(lr_q),
            use_amp=bool(use_amp),
            max_grad_norm=float(max_grad_norm),
            loss_log_interval=int(loss_log_interval),
            device=device,
        )
    if algo == "vdn":
        from algorithms.mpdqn.vdn import MPDQNVDNTrainer

        return MPDQNVDNTrainer(
            n_agents=int(env0.n_ch),
            state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            global_state_dim=int(env0.state_dim * env0.n_ch),
            buffer_capacity=int(buffer_capacity),
            batch_size=int(batch_size),
            lr_actor=float(lr_actor),
            lr_q=float(lr_q),
            use_amp=bool(use_amp),
            max_grad_norm=float(max_grad_norm),
            loss_log_interval=int(loss_log_interval),
            device=device,
        )
    if algo == "qplex":
        from algorithms.mpdqn.qplex import MPDQNQPLEXTrainer

        return MPDQNQPLEXTrainer(
            n_agents=int(env0.n_ch),
            state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            global_state_dim=int(env0.state_dim * env0.n_ch),
            buffer_capacity=int(buffer_capacity),
            batch_size=int(batch_size),
            lr_actor=float(lr_actor),
            lr_q=float(lr_q),
            use_amp=bool(use_amp),
            max_grad_norm=float(max_grad_norm),
            loss_log_interval=int(loss_log_interval),
            device=device,
        )
    raise ValueError(f"Unsupported algo: {algo}")


def _compile_trainer_modules(trainer: Any, *, torch: Any, mode: str) -> list[str]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    compiled = []
    for name in ("actor", "q_net", "target_actor", "target_q_net", "mixer", "target_mixer"):
        module = getattr(trainer, name, None)
        if module is None:
            continue
        setattr(trainer, name, torch.compile(module, mode=str(mode), dynamic=False))
        compiled.append(name)
    return compiled


class Timer:
    def __init__(self, torch: Any, device: str) -> None:
        self.torch = torch
        self.device = str(device)
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def clear(self) -> None:
        self.times.clear()
        self.counts.clear()

    @contextmanager
    def section(self, name: str, *, sync_cuda: bool = False):
        if sync_cuda:
            self.sync()
        start = time.perf_counter()
        yield
        if sync_cuda:
            self.sync()
        elapsed = time.perf_counter() - start
        self.times[name] = self.times.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1


def _build_actions(
    *,
    trainer: Any,
    states: np.ndarray,
    epsilon: float,
    n_envs: int,
    n_agents: int,
    total_param_dim: int,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    if hasattr(trainer, "select_action_batch_all"):
        action_discrete_all, action_params_all = trainer.select_action_batch_all(states, epsilon)
    else:
        action_discrete_all = np.zeros((n_envs, n_agents), dtype=np.int32)
        action_params_all = np.zeros((n_envs, n_agents, total_param_dim), dtype=np.float32)

        for i in range(n_agents):
            ad, ap = trainer.agents[i].select_action_batch(states[:, i, :], epsilon)
            action_discrete_all[:, i] = ad
            action_params_all[:, i, :] = ap

    return (action_discrete_all, action_params_all), action_discrete_all, action_params_all


def _store_transitions(
    *,
    trainer: Any,
    states: np.ndarray,
    actions: Any,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    is_last_step: bool,
    n_envs: int,
) -> None:
    if (
        isinstance(actions, tuple)
        and len(actions) == 2
        and isinstance(actions[0], np.ndarray)
        and isinstance(actions[1], np.ndarray)
    ):
        action_discrete_all = np.asarray(actions[0], dtype=np.int64)
        action_params_all = np.asarray(actions[1], dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=np.bool_) | bool(is_last_step)
        if hasattr(trainer, "store_transition_batch"):
            trainer.store_transition_batch(
                states=states,
                action_discrete=action_discrete_all,
                action_params=action_params_all,
                rewards=rewards,
                next_states=next_states,
                dones=dones_arr,
            )
            return

        actions = [
            [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(action_discrete_all.shape[1])]
            for e in range(n_envs)
        ]

    for e in range(n_envs):
        trainer.store_transition(
            states=states[e],
            actions=actions[e],
            rewards=np.asarray(rewards[e], dtype=np.float32),
            next_states=next_states[e],
            done=bool(dones[e]) or bool(is_last_step),
        )


def _prefill_buffer(
    *,
    vecenv: SubprocVecEnv,
    trainer: Any,
    env0: Environ,
    target_size: int,
    n_steps: int,
    n_envs: int,
    n_agents: int,
    epsilon: float,
) -> int:
    states = vecenv.reset()
    total_param_dim = int(env0.total_param_dim)
    while len(trainer.buffer) < int(target_size):
        for step in range(int(n_steps)):
            actions, _, _ = _build_actions(
                trainer=trainer,
                states=states,
                epsilon=epsilon,
                n_envs=n_envs,
                n_agents=n_agents,
                total_param_dim=total_param_dim,
            )
            next_states, rewards, dones, _ = vecenv.step(actions)
            _store_transitions(
                trainer=trainer,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                is_last_step=(step == int(n_steps) - 1),
                n_envs=n_envs,
            )
            states = vecenv.reset() if bool(np.any(dones)) else next_states
            if len(trainer.buffer) >= int(target_size):
                break
    return len(trainer.buffer)


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch(torch, device)
    use_amp = bool(not args.no_amp) and str(device).startswith("cuda")
    if int(args.warmup) < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    env0 = Environ()
    n_steps = resolve_episode_steps(env0, args.steps)
    n_envs = int(args.num_envs)
    n_agents = int(env0.n_ch)
    state_dim = int(env0.state_dim)
    total_param_dim = int(env0.total_param_dim)

    trainer = _make_trainer(
        algo=str(args.algo),
        env0=env0,
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        lr_actor=float(args.lr_actor),
        lr_q=float(args.lr_q),
        use_amp=use_amp,
        max_grad_norm=float(args.max_grad_norm),
        loss_log_interval=int(args.loss_log_every),
        device=str(device),
    )
    compiled_modules: list[str] = []
    if bool(args.torch_compile):
        compiled_modules = _compile_trainer_modules(trainer, torch=torch, mode=str(args.compile_mode))
    if not bool(args.no_train_step_detail) and hasattr(trainer, "set_profiler"):
        trainer.set_profiler(timer := Timer(torch, str(device)))
    else:
        timer = Timer(torch, str(device))

    vecenv = SubprocVecEnv(
        n_envs,
        p_trans=make_fixed_p_trans(env0),
        start_method=str(args.start_method),
        seed=int(args.seed),
        include_info=False,
    )

    episodes: list[dict[str, Any]] = []
    train_updates = 0
    skipped_updates = 0
    warmup_updates = 0
    warmup_skipped_updates = 0
    epsilon = float(args.epsilon)

    def _run_episode(episode: int) -> tuple[dict[str, Any], int, int]:
        ep_start = time.perf_counter()
        with timer.section("reset"):
            states = vecenv.reset()
        if states.shape != (n_envs, n_agents, state_dim):
            raise RuntimeError(f"Unexpected state shape {states.shape}")

        reward_sum = 0.0
        steps_done = 0
        episode_train_updates = 0
        episode_skipped_updates = 0
        for step in range(int(n_steps)):
            with timer.section("action_select_gpu", sync_cuda=True):
                actions, _, _ = _build_actions(
                    trainer=trainer,
                    states=states,
                    epsilon=epsilon,
                    n_envs=n_envs,
                    n_agents=n_agents,
                    total_param_dim=total_param_dim,
                )

            with timer.section("env_step_send"):
                vecenv.step_async(actions)

            if (step + 1) % int(max(1, args.learn_every)) == 0:
                for _ in range(int(max(1, args.updates_per_learn))):
                    with timer.section("train_step_gpu", sync_cuda=True):
                        loss_info = trainer.train_step()
                    if loss_info is None:
                        episode_skipped_updates += 1
                    elif int(loss_info.get("skipped", 0)) > 0:
                        episode_skipped_updates += 1
                    else:
                        episode_train_updates += 1

            with timer.section("env_step_wait"):
                next_states, rewards, dones, _ = vecenv.step_wait()

            with timer.section("store_transition"):
                _store_transitions(
                    trainer=trainer,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    is_last_step=(step == int(n_steps) - 1),
                    n_envs=n_envs,
                )

            states = next_states
            reward_sum += float(np.mean(rewards))
            steps_done += 1
            if bool(np.any(dones)):
                break

        with timer.section("metrics"):
            energy_arr, jump_arr, suc_arr = vecenv.get_metrics()
            _ = (float(np.sum(energy_arr)), float(np.sum(jump_arr)), float(np.sum(suc_arr)))

        timer.sync()
        return (
            {
                "episode": episode,
                "seconds": time.perf_counter() - ep_start,
                "steps": steps_done,
                "mean_reward_sum": reward_sum,
                "buffer_size": len(trainer.buffer),
            },
            episode_train_updates,
            episode_skipped_updates,
        )

    try:
        prefill_target = max(int(args.prefill), int(args.batch_size))
        prefill_start = time.perf_counter()
        prefill_size = _prefill_buffer(
            vecenv=vecenv,
            trainer=trainer,
            env0=env0,
            target_size=prefill_target,
            n_steps=n_steps,
            n_envs=n_envs,
            n_agents=n_agents,
            epsilon=epsilon,
        )
        timer.sync()
        prefill_seconds = time.perf_counter() - prefill_start

        warmup_seconds = 0.0
        for warmup_episode in range(int(args.warmup)):
            warmup_info, episode_train_updates, episode_skipped_updates = _run_episode(warmup_episode)
            warmup_seconds += float(warmup_info["seconds"])
            warmup_updates += int(episode_train_updates)
            warmup_skipped_updates += int(episode_skipped_updates)

        timer.sync()
        timer.clear()
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for episode in range(int(args.episodes)):
            episode_info, episode_train_updates, episode_skipped_updates = _run_episode(episode)
            episodes.append(episode_info)
            train_updates += int(episode_train_updates)
            skipped_updates += int(episode_skipped_updates)
    finally:
        vecenv.close()

    measured_total = sum(ep["seconds"] for ep in episodes)
    def _rows(items, *, denominator: float) -> list[dict[str, Any]]:
        rows = []
        for name, seconds in sorted(items, key=lambda item: item[1], reverse=True):
            rows.append(
                {
                    "stage": name,
                    "seconds": seconds,
                    "percent": (seconds / denominator * 100.0) if denominator else 0.0,
                    "calls": timer.counts.get(name, 0),
                    "ms_per_call": (seconds / max(1, timer.counts.get(name, 0)) * 1000.0),
                }
            )
        return rows

    outer_items = [(name, seconds) for name, seconds in timer.times.items() if not name.startswith("train_step.")]
    detail_items = [(name, seconds) for name, seconds in timer.times.items() if name.startswith("train_step.")]
    train_step_total = timer.times.get("train_step_gpu", 0.0)
    stage_rows = []
    for row in _rows(outer_items, denominator=measured_total):
        percent = row.pop("percent")
        stage_rows.append(
            {
                **row,
                "percent_of_measured_episode_wall": percent,
            }
        )
    train_step_detail_rows = []
    for row in _rows(detail_items, denominator=train_step_total):
        percent = row.pop("percent")
        train_step_detail_rows.append(
            {
                **row,
                "percent_of_train_step_gpu": percent,
            }
        )

    return {
        "config": {
            "algo": str(args.algo),
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "use_amp": use_amp,
            "episodes": int(args.episodes),
            "steps": int(n_steps),
            "num_envs": n_envs,
            "batch_size": int(args.batch_size),
            "learn_every": int(args.learn_every),
            "updates_per_learn": int(args.updates_per_learn),
            "start_method": str(args.start_method),
            "warmup": int(args.warmup),
            "loss_log_every": int(args.loss_log_every),
            "train_step_detail": not bool(args.no_train_step_detail),
            "torch_compile": bool(args.torch_compile),
            "compile_mode": str(args.compile_mode),
            "compiled_modules": compiled_modules,
        },
        "prefill": {
            "target": max(int(args.prefill), int(args.batch_size)),
            "buffer_size": int(prefill_size),
            "seconds": prefill_seconds,
        },
        "warmup": {
            "episodes": int(args.warmup),
            "seconds": warmup_seconds,
            "train_updates": warmup_updates,
            "skipped_updates": warmup_skipped_updates,
        },
        "episodes": episodes,
        "train_updates": train_updates,
        "skipped_updates": skipped_updates,
        "stage_rows": stage_rows,
        "train_step_detail_rows": train_step_detail_rows,
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1024 / 1024) if str(device).startswith("cuda") else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MP-DQN training stages")
    parser.add_argument("--algo", choices=["iql", "qmix", "vdn", "qplex"], default="qmix")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Unmeasured full training episodes before timing")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--prefill", type=int, default=64)
    parser.add_argument("--learn-every", type=int, default=1)
    parser.add_argument("--updates-per-learn", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-q", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--loss-log-every", type=int, default=1, help="0 disables per-update loss .item()")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-method", type=str, default="spawn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--torch-compile", action="store_true", help="Compile trainer modules for benchmark-only profiling")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--no-train-step-detail", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        json_out = args.json_out if args.json_out.is_absolute() else get_repo_root() / args.json_out
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
