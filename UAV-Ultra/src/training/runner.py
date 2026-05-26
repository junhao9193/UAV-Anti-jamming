"""Unified Stage 5 training runner."""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import random
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from src.algorithms import build_trainer
from src.algorithms.common.buffers import RolloutBuffer
from src.config import specs
from src.config.loader import (
    _deep_merge,
    env_run_summary,
    load_algo_config,
    load_env_config,
    load_experiment_preset,
)
from src.envs import Environ
from src.training.callbacks import CallbackManager, TrainHookContext, build_callbacks
from src.training.checkpoint import (
    load_callback_states,
    load_trainer_state_dict,
    resume_metadata,
    save_checkpoint,
)
from src.training.logging import reserve_output_dir, save_training_data
from src.training.metrics import aggregate_baseline_metrics, success_rate_from_suc
from src.training.schedules import epsilon_by_episode
from src.training.vec_env import SubprocVecEnv, make_fixed_p_trans


_DQN_ALGOS = {"iql", "vdn", "qmix", "qplex"}


@dataclass
class TrainingResult:
    trainer: Any
    metrics: dict[str, list[float]]
    output_dir: Path | None = None
    train_results: list[dict[str, float]] = dataclasses.field(default_factory=list)


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"
        self.errors = getattr(streams[0], "errors", "replace") if streams else "replace"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self._streams and self._streams[0].isatty())


_PRESET_LOG_STEMS = {
    "iql_baseline": "iql",
    "vdn_baseline": "vdn",
    "qmix_plain_baseline": "qmix",
    "qplex_baseline": "qplex",
    "mappo_baseline": "mappo",
    "qmix_wm_concurrent_baseline": "qmix_wm_concurrent",
    "qmix_wm_concurrent_jp_baseline": "qmix_wm_concurrent_jp",
    "qmix_wm_block_baseline": "qmix_wm_block",
    "qmix_wm_block_jp_baseline": "qmix_wm_block_jp",
    "qmix_wm_block_jp_cs_baseline": "qmix_wm_block_jp_cs",
}


def _safe_log_stem(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    stem = stem.strip("._-")
    return stem or "training"


def _default_run_log_name(
    *,
    algorithm: str,
    algo_cfg: Any,
    preset_info: dict[str, Any] | None,
) -> str:
    preset_name = None if preset_info is None else str(preset_info.get("name") or "")
    if preset_name:
        stem = _PRESET_LOG_STEMS.get(preset_name)
        if stem is None:
            stem = preset_name.removesuffix("_baseline")
    else:
        stem = str(algorithm)
    return f"{_safe_log_stem(stem)}_{int(algo_cfg.n_episode)}_seed{int(algo_cfg.seed)}.out"


class _RunLogTee:
    def __init__(self, output_dir: Path | None, log_name: str | None):
        self.output_dir = output_dir
        self.log_path = None if output_dir is None else output_dir / _safe_log_stem(log_name or "training.out")
        self._file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        if self.log_path is None:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._file)
        sys.stderr = _TeeStream(self._stderr, self._file)
        print(f"[training-runner] run_log={self.log_path}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.log_path is not None and exc_type is not None:
            print("[training-runner] run failed; traceback follows:", file=sys.stderr, flush=True)
            traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()
        return False


def _configure_process_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass


def _run_main_with_error_logging(argv: list[str] | None = None) -> None:
    _configure_process_logging()
    try:
        main(argv)
    except KeyboardInterrupt:
        print("[training-runner] interrupted by KeyboardInterrupt", file=sys.stderr, flush=True)
        raise
    except Exception:
        print("[training-runner] unhandled exception; full traceback follows:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        raise SystemExit(1)


def _default_env_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "defaults" / "env.yaml"


def _preset_metadata(preset: str | Path, ep: Any) -> dict[str, Any]:
    raw = str(preset)
    is_builtin_name = not any(ch in raw for ch in ("/", "\\", "."))
    return {
        "name": raw if is_builtin_name else Path(raw).stem,
        "path": str(ep.path),
        "sha256": str(ep.sha256),
        "description": str(ep.description),
        "source": str(ep.source),
    }


def resolve_episode_steps(env_cfg: Any, n_steps: int | None) -> int:
    if n_steps is None:
        n_steps = int(env_cfg.max_episode_steps)
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return n_steps


def _validate_positive(*, n_episode: int, n_steps: int | None, num_envs: int = 1) -> None:
    if int(n_episode) <= 0:
        raise ValueError(f"n_episode must be positive, got {n_episode}")
    if n_steps is not None and int(n_steps) <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if int(num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _reapply_loaded_callback_runtime_state(callbacks: CallbackManager, trainer: Any) -> None:
    for cb in callbacks:
        if getattr(cb, "name", "") == "critic_stable" and hasattr(cb, "apply_lr_scale"):
            cb.apply_lr_scale(trainer)


def _mean_numeric_fields(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            totals[key] = totals.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / max(1, counts[key]) for key in totals}


def _progress_from_callbacks(callbacks: CallbackManager, episode: int) -> dict[str, float | str]:
    fields: dict[str, float | str] = {}
    for cb in callbacks:
        phase_for_episode = getattr(cb, "_phase_for_episode", None)
        if callable(phase_for_episode):
            fields["phase"] = str(phase_for_episode(int(episode)))
        last_wm_result = getattr(cb, "last_wm_result", None)
        if last_wm_result:
            for key, value in dict(last_wm_result[-1]).items():
                if key.startswith("wm_"):
                    fields[key] = float(value)
        if getattr(cb, "name", "") == "jammer_prediction":
            warmup = max(1, int(getattr(cb, "warmup_episodes", 1)))
            use_feature = bool(getattr(cb, "use_feature", True))
            fields["fs"] = min(1.0, float(episode) / float(warmup)) if use_feature else 0.0
    return fields


def _should_log_episode(algo_cfg: Any, episode: int, n_episode: int) -> bool:
    every = int(max(1, getattr(algo_cfg, "loss_log_every", 1)))
    return episode == 0 or (episode + 1) % every == 0 or episode == n_episode - 1


def _format_progress_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _log_episode_progress(
    *,
    algorithm: str,
    episode: int,
    n_episode: int,
    reward_history: list[float],
    success_rate_history: list[float],
    energy_history: list[float],
    jump_history: list[float],
    train_summary: dict[str, float],
    extra_fields: dict[str, float | str] | None = None,
) -> None:
    avg_window = min(10, len(reward_history))
    fields: dict[str, float | str] = {
        "reward": float(reward_history[-1]),
        "avg10": float(np.mean(reward_history[-avg_window:])),
        "sr": float(success_rate_history[-1]),
        "energy": float(energy_history[-1]),
        "jump": float(jump_history[-1]),
    }
    if extra_fields:
        fields.update(extra_fields)
    for key in ("loss_q", "loss_actor", "loss_jammer", "wm_loss", "wm_L_VC", "wm_eta", "loss_pi", "loss_v"):
        if key in train_summary:
            fields[key] = float(train_summary[key])

    payload = " ".join(f"{key}={_format_progress_value(value)}" for key, value in fields.items())
    print(f"[{algorithm}] ep={episode + 1}/{n_episode} {payload}", flush=True)


def _configure_torch(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _select_dqn_actions(
    trainer: Any,
    states: np.ndarray,
    *,
    epsilon: float,
    base_param_dim: int,
    sensing_histories: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, np.ndarray]]]]:
    """Stage 8：JP enabled 时 ``sensing_histories`` 为 ``(n_envs, n_agents, H, C)``，
    per-agent 切片传给 ``select_action_batch``；off 时 None → 走原路径。
    """
    n_envs, n_agents, _ = states.shape
    action_discrete = np.zeros((n_envs, n_agents), dtype=np.int64)
    action_params = np.zeros((n_envs, n_agents, base_param_dim), dtype=np.float32)

    if hasattr(trainer, "agents") and all(hasattr(agent, "select_action_batch") for agent in trainer.agents):
        for i in range(n_agents):
            if sensing_histories is not None:
                ad_i, ap_i = trainer.agents[i].select_action_batch(
                    states[:, i, :], epsilon, sensing_history=sensing_histories[:, i],
                )
            else:
                ad_i, ap_i = trainer.agents[i].select_action_batch(states[:, i, :], epsilon)
            action_discrete[:, i] = np.asarray(ad_i, dtype=np.int64)
            action_params[:, i, :] = np.asarray(ap_i, dtype=np.float32)
    else:
        for e in range(n_envs):
            selected = trainer.select_actions([states[e, i, :] for i in range(n_agents)], epsilon)
            for i, (ad, ap) in enumerate(selected):
                action_discrete[e, i] = int(ad)
                # Stage 4 action heads intentionally emit only baseline power params.
                # Mobility deltas are appended later by policy_mobility; do not carry
                # any future extra params through this fallback silently.
                action_params[e, i, :] = np.asarray(ap, dtype=np.float32).reshape(-1)[:base_param_dim]

    actions = [
        [
            (int(action_discrete[e, i]), action_params[e, i, :].copy())
            for i in range(n_agents)
        ]
        for e in range(n_envs)
    ]
    return action_discrete, action_params, actions


def run_dqn_loop(
    *,
    algorithm: str,
    trainer: Any,
    vecenv: Any,
    env_cfg: Any,
    algo_cfg: Any,
    callbacks: CallbackManager,
    p_trans: np.ndarray | None,
) -> tuple[dict[str, list[float]], list[dict[str, float]]]:
    n_steps = resolve_episode_steps(env_cfg, algo_cfg.n_steps)
    n_episode = int(algo_cfg.n_episode)
    n_envs = int(algo_cfg.num_envs)
    n_agents = int(env_cfg.n_ch)
    base_param_dim = int(specs.total_param_dim(env_cfg))
    state_dim = int(specs.state_dim(env_cfg))

    reward_history: list[float] = []
    success_rate_history: list[float] = []
    energy_history: list[float] = []
    jump_history: list[float] = []
    train_results: list[dict[str, float]] = []

    global_step = 0
    for episode in range(n_episode):
        episode_train_results: list[dict[str, float]] = []
        states = vecenv.reset(p_trans)
        if states.shape != (n_envs, n_agents, state_dim):
            raise RuntimeError(
                f"Unexpected state shape {states.shape}; expected ({n_envs},{n_agents},{state_dim})"
            )
        # Stage 8：episode reset 后立刻初始化 JP sensing history + 设 feature_scale。
        # JP-off 时 no-op。必须在第一次 select 之前完成。
        callbacks.reset_jp_state(states, episode=episode)

        episode_reward = 0.0
        steps_done = 0
        epsilon = epsilon_by_episode(
            episode=episode,
            epsilon_start=float(algo_cfg.epsilon_start),
            epsilon_min=float(algo_cfg.epsilon_min),
            epsilon_decay=float(algo_cfg.epsilon_decay),
        )

        for step in range(n_steps):
            # (1) select：用 current_sensing_histories（JP-off 时 None → 走原路径）
            sensing_histories = callbacks.get_current_sensing_histories()
            action_discrete, action_params, actions = _select_dqn_actions(
                trainer,
                states,
                epsilon=epsilon,
                base_param_dim=base_param_dim,
                sensing_histories=sensing_histories,
            )
            env_actions = callbacks.on_action_selected(actions)

            # (2) step
            vecenv.step_async(env_actions)

            # (3) learn（Stage 5 锁定：在 step_async 与 step_wait 之间）
            if (global_step + 1) % int(max(1, algo_cfg.learn_every)) == 0:
                for _ in range(int(max(1, algo_cfg.updates_per_learn))):
                    result = callbacks.train_step(
                        TrainHookContext(trainer=trainer, episode=episode, step=global_step)
                    )
                    if result is not None:
                        train_results.append(result)
                        episode_train_results.append(result)

            next_states, rewards, dones, infos = vecenv.step_wait()
            is_last_step = step == n_steps - 1
            dones_to_store = np.asarray(dones, dtype=np.bool_) | bool(is_last_step)

            # (4) callback compute next_sensing_histories + jammer_targets（不 swap）
            callbacks.on_transition_batch(
                states=states,
                actions=env_actions,
                action_discrete=action_discrete,
                action_params=action_params,
                rewards=rewards,
                next_states=next_states,
                dones=dones_to_store,
                infos=infos,
            )
            # (5) store transition：current_sensing_histories（select 时） + next_*（这一步）
            jp_fields = callbacks.get_jp_buffer_fields()
            store_kwargs = dict(
                states=states,
                action_discrete=action_discrete,
                action_params=action_params,
                rewards=rewards,
                next_states=next_states,
                dones=dones_to_store.astype(np.float32),
            )
            if jp_fields is not None:
                store_kwargs.update(jp_fields)
            trainer.store_transition_batch(**store_kwargs)

            # (6) commit swap：current ← next，准备下一步 select
            callbacks.commit_jp_history_swap()

            states = next_states
            episode_reward += float(np.mean(rewards))
            steps_done += 1
            global_step += 1
            if bool(np.any(dones)):
                break

        energy_arr, jump_arr, suc_arr = vecenv.get_metrics()
        agg = aggregate_baseline_metrics(
            energy=energy_arr,
            jump=jump_arr,
            suc=suc_arr,
            steps_done=max(1, steps_done),
            n_envs=n_envs,
            n_agents=n_agents,
            n_des=int(specs.n_des(env_cfg)),
        )
        reward_history.append(float(episode_reward))
        success_rate_history.append(float(agg["success_rate"]))
        energy_history.append(float(agg["energy"]))
        jump_history.append(float(agg["jump"]))
        if _should_log_episode(algo_cfg, episode, n_episode):
            extra_fields = {"eps": float(epsilon)}
            extra_fields.update(_progress_from_callbacks(callbacks, episode))
            _log_episode_progress(
                algorithm=algorithm,
                episode=episode,
                n_episode=n_episode,
                reward_history=reward_history,
                success_rate_history=success_rate_history,
                energy_history=energy_history,
                jump_history=jump_history,
                train_summary=_mean_numeric_fields(episode_train_results),
                extra_fields=extra_fields,
            )

    return (
        {
            "reward": reward_history,
            "success_rate": success_rate_history,
            "energy": energy_history,
            "jump": jump_history,
        },
        train_results,
    )


def _run_dqn_training(
    *,
    algorithm: str,
    env_cfg: Any,
    algo_cfg: Any,
    env_overrides: dict[str, Any] | None,
    preset_info: dict[str, Any] | None,
    resume_from: Path | None,
    resume_info: dict[str, Any] | None,
    no_save: bool,
    output_root: Path | None,
    vecenv_factory: Callable[..., Any] = SubprocVecEnv,
) -> TrainingResult:
    device = "cuda" if (algo_cfg.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if algo_cfg.device == "auto" else str(algo_cfg.device)
    )
    _configure_torch(device)
    _seed_everything(int(algo_cfg.seed))

    output_dir = None if no_save else reserve_output_dir(algorithm, output_root=output_root)
    log_name = _default_run_log_name(algorithm=algorithm, algo_cfg=algo_cfg, preset_info=preset_info)
    log_path = None if output_dir is None else output_dir / log_name
    with _RunLogTee(output_dir, log_name):
        env0 = Environ(config=env_overrides)
        p_trans_fixed = make_fixed_p_trans(env0)
        trainer = build_trainer(algorithm, env_cfg=env_cfg, algo_cfg=algo_cfg, device=device)
        callbacks = build_callbacks(getattr(algo_cfg, "callbacks", []), env_cfg=env_cfg, algo_cfg=algo_cfg)
        callbacks.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=int(algo_cfg.num_envs))
        if resume_from is not None:
            checkpoint = load_trainer_state_dict(trainer, resume_from, algorithm, device=device)
            load_callback_states(list(callbacks), checkpoint.get("callbacks", {}), strict=True)
            _reapply_loaded_callback_runtime_state(callbacks, trainer)

        vecenv = vecenv_factory(
            int(algo_cfg.num_envs),
            p_trans=p_trans_fixed,
            start_method=str(algo_cfg.start_method),
            seed=int(algo_cfg.seed),
            env_overrides=env_overrides,
        )
        try:
            metrics, train_results = run_dqn_loop(
                algorithm=algorithm,
                trainer=trainer,
                vecenv=vecenv,
                env_cfg=env_cfg,
                algo_cfg=algo_cfg,
                callbacks=callbacks,
                p_trans=p_trans_fixed,
            )
        finally:
            vecenv.close()

        if output_dir is not None:
            run_config = {
                "algorithm": algorithm,
                "seed": int(algo_cfg.seed),
                "num_envs": int(algo_cfg.num_envs),
                "batch_size": int(algo_cfg.batch_size),
                "buffer_capacity": int(algo_cfg.buffer_capacity),
                "learn_every": int(algo_cfg.learn_every),
                "updates_per_learn": int(algo_cfg.updates_per_learn),
                "epsilon_start": float(algo_cfg.epsilon_start),
                "epsilon_min": float(algo_cfg.epsilon_min),
                "epsilon_decay": float(algo_cfg.epsilon_decay),
                "device": str(device),
                "start_method": str(algo_cfg.start_method),
                "callbacks": list(getattr(algo_cfg, "callbacks", [])),
                "preset": preset_info,
                "resume_from": resume_info,
                "log_file": str(log_path),
                "use_amp": bool(getattr(algo_cfg, "use_amp", False)),
                **env_run_summary(env_cfg, _default_env_yaml_path(), overrides=env_overrides),
            }
            n_steps = resolve_episode_steps(env_cfg, algo_cfg.n_steps)
            save_training_data(
                algorithm=algorithm,
                reward_history=metrics["reward"],
                success_rate_history=metrics["success_rate"],
                energy_history=metrics["energy"],
                jump_history=metrics["jump"],
                n_episode=int(algo_cfg.n_episode),
                n_steps=n_steps,
                run_config=run_config,
                output_root=output_root,
                data_dir=output_dir,
            )
            save_checkpoint(
                path=output_dir / f"{algorithm}_weights.pth",
                algorithm=algorithm,
                trainer=trainer,
                callbacks=list(callbacks),
                extra={"n_steps": n_steps},
            )
            print(f"[training-runner] output_dir={output_dir}", flush=True)

    return TrainingResult(trainer=trainer, metrics=metrics, output_dir=output_dir, train_results=train_results)


def _run_mappo_training(
    *,
    env_cfg: Any,
    algo_cfg: Any,
    env_overrides: dict[str, Any] | None,
    preset_info: dict[str, Any] | None,
    resume_from: Path | None,
    resume_info: dict[str, Any] | None,
    no_save: bool,
    output_root: Path | None,
) -> TrainingResult:
    device = "cuda" if (algo_cfg.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if algo_cfg.device == "auto" else str(algo_cfg.device)
    )
    _configure_torch(device)
    _seed_everything(int(algo_cfg.seed))

    output_dir = None if no_save else reserve_output_dir("mappo", output_root=output_root)
    log_name = _default_run_log_name(algorithm="mappo", algo_cfg=algo_cfg, preset_info=preset_info)
    log_path = None if output_dir is None else output_dir / log_name
    with _RunLogTee(output_dir, log_name):
        trainer = build_trainer("mappo", env_cfg=env_cfg, algo_cfg=algo_cfg, device=device)
        if resume_from is not None:
            checkpoint = load_trainer_state_dict(trainer, resume_from, "mappo", device=device)
            load_callback_states([], checkpoint.get("callbacks", {}), strict=True)
        env = Environ(config={"env_seed": int(algo_cfg.seed), **(env_overrides or {})})
        p_trans_fixed = make_fixed_p_trans(env)
        n_steps = resolve_episode_steps(env_cfg, algo_cfg.n_steps)
        n_agents = int(env_cfg.n_ch)
        cont_dim = int(specs.param_dim_per_action(env_cfg))

        reward_history: list[float] = []
        success_rate_history: list[float] = []
        energy_history: list[float] = []
        jump_history: list[float] = []
        train_results: list[dict[str, float]] = []

        n_episode = int(algo_cfg.n_episode)
        for _episode in range(n_episode):
            state = env.reset(p_trans_fixed)
            env.clear_reward()
            buffer = RolloutBuffer(n_agents=n_agents)
            episode_reward = 0.0
            steps_done = 0

            for _step in range(n_steps):
                obs_step = np.stack(state, axis=0).astype(np.float32)
                global_state = np.concatenate(state, axis=-1).astype(np.float32)
                global_step = np.tile(global_state, (n_agents, 1)).astype(np.float32)
                agent_ids = np.arange(n_agents, dtype=np.int64)

                actions = []
                act_discrete = np.zeros((n_agents,), dtype=np.int64)
                act_cont = np.zeros((n_agents, cont_dim), dtype=np.float32)
                log_probs = np.zeros((n_agents,), dtype=np.float32)
                values = np.zeros((n_agents,), dtype=np.float32)

                for i in range(n_agents):
                    res = trainer.act(obs_step[i], global_state, agent_id=i, deterministic=False)
                    params_full = np.zeros((int(specs.total_param_dim(env_cfg)),), dtype=np.float32)
                    start = int(res.action_discrete) * cont_dim
                    params_full[start : start + cont_dim] = res.action_cont
                    actions.append((int(res.action_discrete), params_full))
                    act_discrete[i] = int(res.action_discrete)
                    act_cont[i] = res.action_cont
                    log_probs[i] = float(res.log_prob)
                    values[i] = float(res.value)

                next_state, rewards, done, _info = env.step(actions)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(n_agents)
                done_step = np.full((n_agents,), float(done), dtype=np.float32)
                buffer.add(
                    obs=obs_step,
                    global_state=global_step,
                    agent_id=agent_ids,
                    action_discrete=act_discrete,
                    action_cont=act_cont,
                    log_prob=log_probs,
                    value=values,
                    reward=rewards,
                    done=done_step,
                )

                state = next_state
                episode_reward += float(np.mean(rewards))
                steps_done += 1
                if done:
                    break

            global_state_last = np.concatenate(state, axis=-1).astype(np.float32)
            global_last = np.tile(global_state_last, (n_agents, 1)).astype(np.float32)
            last_values = trainer.value(global_last, np.arange(n_agents, dtype=np.int64))
            returns, advantages = buffer.compute_returns_and_advantages(
                last_value=last_values,
                gamma=float(trainer.gamma),
                gae_lambda=float(trainer.gae_lambda),
            )
            update_info = trainer.update(buffer.as_batch(returns=returns, advantages=advantages))
            train_results.append(update_info)

            total_links = float(max(1, steps_done) * n_agents * int(specs.n_des(env_cfg)))
            reward_history.append(float(episode_reward))
            success_rate_history.append(success_rate_from_suc(float(env.rew_suc), total_links=total_links))
            energy_history.append(float(env.rew_energy) / total_links)
            jump_history.append(float(env.rew_jump) / total_links)
            if _should_log_episode(algo_cfg, _episode, n_episode):
                _log_episode_progress(
                    algorithm="mappo",
                    episode=_episode,
                    n_episode=n_episode,
                    reward_history=reward_history,
                    success_rate_history=success_rate_history,
                    energy_history=energy_history,
                    jump_history=jump_history,
                    train_summary=_mean_numeric_fields([update_info]),
                )

        metrics = {
            "reward": reward_history,
            "success_rate": success_rate_history,
            "energy": energy_history,
            "jump": jump_history,
        }

        if output_dir is not None:
            run_config = {
                "algorithm": "mappo",
                "seed": int(algo_cfg.seed),
                "lr": float(algo_cfg.lr),
                "gamma": float(algo_cfg.gamma),
                "gae_lambda": float(algo_cfg.gae_lambda),
                "clip_range": float(algo_cfg.clip_range),
                "ent_coef": float(algo_cfg.ent_coef),
                "vf_coef": float(algo_cfg.vf_coef),
                "update_epochs": int(algo_cfg.update_epochs),
                "minibatch_size": int(algo_cfg.minibatch_size),
                "max_grad_norm": float(algo_cfg.max_grad_norm),
                "device": str(device),
                "preset": preset_info,
                "resume_from": resume_info,
                "log_file": str(log_path),
                **env_run_summary(env_cfg, _default_env_yaml_path(), overrides=env_overrides),
            }
            save_training_data(
                algorithm="mappo",
                reward_history=metrics["reward"],
                success_rate_history=metrics["success_rate"],
                energy_history=metrics["energy"],
                jump_history=metrics["jump"],
                n_episode=int(algo_cfg.n_episode),
                n_steps=n_steps,
                run_config=run_config,
                output_root=output_root,
                data_dir=output_dir,
            )
            save_checkpoint(
                path=output_dir / "mappo_weights.pth",
                algorithm="mappo",
                trainer=trainer,
                callbacks=[],
                extra={"n_steps": n_steps},
            )
            print(f"[training-runner] output_dir={output_dir}", flush=True)

    return TrainingResult(trainer=trainer, metrics=metrics, output_dir=output_dir, train_results=train_results)


def run_training(
    algorithm: str,
    *,
    preset: str | Path | None = None,
    env_overrides: dict[str, Any] | None = None,
    algo_overrides: dict[str, Any] | None = None,
    use_amp: bool | None = None,
    resume_from: str | Path | None = None,
    no_save: bool = False,
    output_root: str | Path | None = None,
    vecenv_factory: Callable[..., Any] = SubprocVecEnv,
) -> TrainingResult:
    algorithm = str(algorithm).lower()
    preset_info: dict[str, Any] | None = None
    if preset is not None:
        ep = load_experiment_preset(preset)
        if ep.algorithm != algorithm:
            raise ValueError(
                f"preset algorithm {ep.algorithm!r} does not match requested algorithm {algorithm!r}"
            )
        env_overrides = _deep_merge(ep.env, env_overrides or {})
        algo_overrides = _deep_merge(ep.algo, algo_overrides or {})
        preset_info = _preset_metadata(preset, ep)
    if use_amp is not None:
        algo_overrides = _deep_merge(algo_overrides or {}, {"use_amp": bool(use_amp)})
    if algorithm == "mappo" and algo_overrides is not None and "use_amp" in algo_overrides:
        raise ValueError("MAPPO does not support AMP override; remove use_amp / --amp / --no-amp")
    resume_path = None if resume_from is None else Path(resume_from).expanduser().resolve()
    resume_info = None if resume_path is None else resume_metadata(resume_path)
    env_cfg = load_env_config(overrides=env_overrides)
    algo_cfg = load_algo_config(algorithm, overrides=algo_overrides, env_cfg=env_cfg)

    if algorithm in _DQN_ALGOS:
        _validate_positive(
            n_episode=int(algo_cfg.n_episode),
            n_steps=algo_cfg.n_steps,
            num_envs=int(algo_cfg.num_envs),
        )
        return _run_dqn_training(
            algorithm=algorithm,
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            env_overrides=env_overrides,
            preset_info=preset_info,
            resume_from=resume_path,
            resume_info=resume_info,
            no_save=bool(no_save),
            output_root=None if output_root is None else Path(output_root),
            vecenv_factory=vecenv_factory,
        )
    if algorithm == "mappo":
        _validate_positive(n_episode=int(algo_cfg.n_episode), n_steps=algo_cfg.n_steps)
        return _run_mappo_training(
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            env_overrides=env_overrides,
            preset_info=preset_info,
            resume_from=resume_path,
            resume_info=resume_info,
            no_save=bool(no_save),
            output_root=None if output_root is None else Path(output_root),
        )
    raise ValueError(f"Stage 5 training runner does not support algorithm {algorithm!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 5 UAV-Ultra training runner")
    parser.add_argument("algorithm", choices=sorted(_DQN_ALGOS | {"mappo"}))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-method", type=str, default=None)
    parser.add_argument("--callback", action="append", dest="callbacks", default=None)
    parser.add_argument("--preset", type=str, default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", dest="use_amp", default=None)
    amp_group.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.episodes is not None:
        overrides["n_episode"] = int(args.episodes)
    if args.steps is not None:
        overrides["n_steps"] = int(args.steps)
    if args.num_envs is not None:
        overrides["num_envs"] = int(args.num_envs)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.device is not None:
        overrides["device"] = str(args.device)
    if args.start_method is not None:
        overrides["start_method"] = str(args.start_method)
    if args.callbacks is not None:
        overrides["callbacks"] = list(args.callbacks)

    result = run_training(
        args.algorithm,
        preset=args.preset,
        algo_overrides=overrides,
        use_amp=args.use_amp,
        resume_from=args.resume,
        no_save=bool(args.no_save),
    )
    print(f"Training completed for {args.algorithm}. Episodes: {len(result.metrics['reward'])}")
    if result.output_dir is not None:
        print(f"Output: {result.output_dir}")


if __name__ == "__main__":
    _run_main_with_error_logging()


__all__ = ["TrainingResult", "resolve_episode_steps", "run_dqn_loop", "run_training"]
