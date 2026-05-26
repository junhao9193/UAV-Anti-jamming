"""Unified Stage 5 training runner."""

from __future__ import annotations

import argparse
import dataclasses
import random
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
from src.training.checkpoint import save_checkpoint
from src.training.logging import save_training_data
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
    no_save: bool,
    output_root: Path | None,
    vecenv_factory: Callable[..., Any] = SubprocVecEnv,
) -> TrainingResult:
    device = "cuda" if (algo_cfg.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if algo_cfg.device == "auto" else str(algo_cfg.device)
    )
    _configure_torch(device)
    _seed_everything(int(algo_cfg.seed))

    env0 = Environ(config=env_overrides)
    p_trans_fixed = make_fixed_p_trans(env0)
    trainer = build_trainer(algorithm, env_cfg=env_cfg, algo_cfg=algo_cfg, device=device)
    callbacks = build_callbacks(getattr(algo_cfg, "callbacks", []), env_cfg=env_cfg, algo_cfg=algo_cfg)
    callbacks.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=int(algo_cfg.num_envs))

    vecenv = vecenv_factory(
        int(algo_cfg.num_envs),
        p_trans=p_trans_fixed,
        start_method=str(algo_cfg.start_method),
        seed=int(algo_cfg.seed),
        env_overrides=env_overrides,
    )
    try:
        metrics, train_results = run_dqn_loop(
            trainer=trainer,
            vecenv=vecenv,
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            callbacks=callbacks,
            p_trans=p_trans_fixed,
        )
    finally:
        vecenv.close()

    output_dir = None
    if not no_save:
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
            **env_run_summary(env_cfg, _default_env_yaml_path(), overrides=env_overrides),
        }
        n_steps = resolve_episode_steps(env_cfg, algo_cfg.n_steps)
        _, _, output_dir = save_training_data(
            algorithm=algorithm,
            reward_history=metrics["reward"],
            success_rate_history=metrics["success_rate"],
            energy_history=metrics["energy"],
            jump_history=metrics["jump"],
            n_episode=int(algo_cfg.n_episode),
            n_steps=n_steps,
            run_config=run_config,
            output_root=output_root,
        )
        save_checkpoint(
            path=output_dir / f"{algorithm}_weights.pth",
            algorithm=algorithm,
            trainer=trainer,
            callbacks=list(callbacks),
            extra={"n_steps": n_steps},
        )

    return TrainingResult(trainer=trainer, metrics=metrics, output_dir=output_dir, train_results=train_results)


def _run_mappo_training(
    *,
    env_cfg: Any,
    algo_cfg: Any,
    env_overrides: dict[str, Any] | None,
    preset_info: dict[str, Any] | None,
    no_save: bool,
    output_root: Path | None,
) -> TrainingResult:
    device = "cuda" if (algo_cfg.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if algo_cfg.device == "auto" else str(algo_cfg.device)
    )
    _configure_torch(device)
    _seed_everything(int(algo_cfg.seed))

    trainer = build_trainer("mappo", env_cfg=env_cfg, algo_cfg=algo_cfg, device=device)
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

    for _episode in range(int(algo_cfg.n_episode)):
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

    metrics = {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }

    output_dir = None
    if not no_save:
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
            **env_run_summary(env_cfg, _default_env_yaml_path(), overrides=env_overrides),
        }
        _, _, output_dir = save_training_data(
            algorithm="mappo",
            reward_history=metrics["reward"],
            success_rate_history=metrics["success_rate"],
            energy_history=metrics["energy"],
            jump_history=metrics["jump"],
            n_episode=int(algo_cfg.n_episode),
            n_steps=n_steps,
            run_config=run_config,
            output_root=output_root,
        )
        save_checkpoint(
            path=output_dir / "mappo_weights.pth",
            algorithm="mappo",
            trainer=trainer,
            callbacks=[],
            extra={"n_steps": n_steps},
        )

    return TrainingResult(trainer=trainer, metrics=metrics, output_dir=output_dir, train_results=train_results)


def run_training(
    algorithm: str,
    *,
    preset: str | Path | None = None,
    env_overrides: dict[str, Any] | None = None,
    algo_overrides: dict[str, Any] | None = None,
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
        no_save=bool(args.no_save),
    )
    print(f"Training completed for {args.algorithm}. Episodes: {len(result.metrics['reward'])}")
    if result.output_dir is not None:
        print(f"Output: {result.output_dir}")


if __name__ == "__main__":
    main()


__all__ = ["TrainingResult", "resolve_episode_steps", "run_dqn_loop", "run_training"]
