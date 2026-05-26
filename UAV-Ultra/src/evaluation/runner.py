"""Unified Stage 6 evaluation runner."""

from __future__ import annotations

import argparse
import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.algorithms import build_evaluator, build_trainer
from src.algorithms.heuristic.policies import normalize_power_mode
from src.config import specs
from src.config.loader import env_run_summary, load_algo_config, load_env_config
from src.envs import Environ
from src.training.callbacks import CallbackManager, build_callbacks
from src.training.checkpoint import (
    load_callback_states,
    load_trainer_state_dict,
)
from src.training.logging import save_training_data
from src.training.metrics import aggregate_baseline_metrics
from src.training.vec_env import SubprocVecEnv, make_fixed_p_trans


_LEARNING_ALGOS = {"iql", "vdn", "qmix", "qplex", "mappo"}
_EVAL_ALGOS = _LEARNING_ALGOS | {"heuristic"}
_DEFAULT_POLICY = "greedy_sensing"
_DEFAULT_POWER_MODE = "quality_adaptive"


@dataclass
class EvaluationResult:
    trainer: Any | None
    metrics: dict[str, list[float]]
    output_dir: Path | None = None
    callback_states: dict[str, dict] | None = None


def _default_env_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "defaults" / "env.yaml"


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _resolve_device(device: str) -> str:
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_episode_steps(env_cfg: Any, steps: int | None) -> int:
    n_steps = int(env_cfg.max_episode_steps) if steps is None else int(steps)
    if n_steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}")
    return n_steps


def _validate_positive(*, episodes: int, steps: int | None, num_envs: int) -> None:
    if int(episodes) <= 0:
        raise ValueError(f"episodes must be positive, got {episodes!r}")
    if steps is not None and int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}")
    if int(num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs!r}")


def _validate_cross_algo_args(
    algorithm: str,
    *,
    checkpoint_path: str | Path | None,
    policy_name: str,
    power_mode: str,
    callback_overrides: list[str] | None,
) -> None:
    if algorithm == "heuristic":
        if checkpoint_path is not None:
            raise ValueError("heuristic evaluation does not accept checkpoint_path")
        if callback_overrides:
            raise ValueError("callbacks are only valid for qmix evaluation")
        return

    if checkpoint_path is None:
        raise ValueError(f"{algorithm} evaluation requires checkpoint_path")
    if policy_name != _DEFAULT_POLICY:
        raise ValueError("--policy / policy_name is only valid for heuristic evaluation")
    if power_mode != _DEFAULT_POWER_MODE:
        raise ValueError("--power-mode / power_mode is only valid for heuristic evaluation")
    if algorithm != "qmix" and callback_overrides:
        raise ValueError("callbacks are only valid for qmix evaluation")


def _select_vec_actions(
    evaluator: Any,
    states: np.ndarray,
    *,
    algorithm: str,
) -> list[list[tuple[int, np.ndarray]]]:
    n_envs = int(states.shape[0])
    actions: list[list[tuple[int, np.ndarray]]] = []
    for env_id in range(n_envs):
        env_states = [states[env_id, i, :] for i in range(int(states.shape[1]))]
        if algorithm == "mappo":
            global_state = np.concatenate(env_states, axis=-1).astype(np.float32)
            env_actions = evaluator.select_actions(env_states, global_state=global_state)
        else:
            env_actions = evaluator.select_actions(env_states)
        actions.append(env_actions)
    return actions


def _run_eval_loop(
    *,
    algorithm: str,
    evaluator: Any,
    vecenv: Any,
    env_cfg: Any,
    callbacks: CallbackManager,
    p_trans: np.ndarray | None,
    episodes: int,
    steps: int,
) -> dict[str, list[float]]:
    reward_history: list[float] = []
    success_rate_history: list[float] = []
    energy_history: list[float] = []
    jump_history: list[float] = []

    n_envs = int(vecenv.n_envs)
    n_agents = int(env_cfg.n_ch)
    n_des = int(specs.n_des(env_cfg))

    for _episode in range(int(episodes)):
        states = vecenv.reset(p_trans)
        episode_reward = 0.0
        steps_done = 0

        for _step in range(int(steps)):
            actions = _select_vec_actions(evaluator, states, algorithm=algorithm)
            actions = callbacks.on_action_selected(actions)
            next_states, rewards, dones, _infos = vecenv.step(actions)
            states = next_states
            episode_reward += float(np.mean(rewards))
            steps_done += 1
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
            n_des=n_des,
        )
        reward_history.append(float(episode_reward))
        success_rate_history.append(float(agg["success_rate"]))
        energy_history.append(float(agg["energy"]))
        jump_history.append(float(agg["jump"]))

    return {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }


def _set_eval_mode(trainer: Any | None) -> None:
    if trainer is None:
        return
    for agent in getattr(trainer, "agents", []) or []:
        agent.actor.eval()
        agent.q_net.eval()
        agent.target_actor.eval()
        agent.target_q_net.eval()
    if hasattr(trainer, "mixer"):
        trainer.mixer.eval()
    if hasattr(trainer, "target_mixer"):
        trainer.target_mixer.eval()
    if hasattr(trainer, "actor"):
        trainer.actor.eval()
    if hasattr(trainer, "critic"):
        trainer.critic.eval()


def run_evaluation(
    algorithm: str,
    *,
    checkpoint_path: str | Path | None = None,
    env_overrides: dict | None = None,
    algo_overrides: dict | None = None,
    episodes: int = 100,
    steps: int | None = None,
    num_envs: int = 32,
    seed: int = 0,
    device: str = "auto",
    start_method: str = "spawn",
    deterministic: bool = True,
    policy_name: str = _DEFAULT_POLICY,
    power_mode: str = _DEFAULT_POWER_MODE,
    callback_overrides: list[str] | None = None,
    no_save: bool = False,
    output_root: str | Path | None = None,
    vecenv_factory: Callable[..., Any] = SubprocVecEnv,
) -> EvaluationResult:
    algorithm = str(algorithm).lower()
    if algorithm not in _EVAL_ALGOS:
        raise ValueError(f"unknown evaluation algorithm {algorithm!r}; valid: {sorted(_EVAL_ALGOS)}")
    _validate_positive(episodes=int(episodes), steps=steps, num_envs=int(num_envs))
    _validate_cross_algo_args(
        algorithm,
        checkpoint_path=checkpoint_path,
        policy_name=str(policy_name),
        power_mode=str(power_mode),
        callback_overrides=callback_overrides,
    )

    resolved_device = _resolve_device(device)
    _seed_everything(int(seed))
    env_cfg = load_env_config(overrides=env_overrides)
    env0 = Environ(config=env_overrides)
    p_trans_fixed = make_fixed_p_trans(env0)
    n_steps = _resolve_episode_steps(env_cfg, steps)

    trainer = None
    callback_states: dict[str, dict] | None = None
    callbacks = CallbackManager([])

    if algorithm == "heuristic":
        normalized_power_mode = normalize_power_mode(str(policy_name), str(power_mode))
        evaluator = build_evaluator(
            "heuristic",
            env_cfg=env_cfg,
            policy_name=str(policy_name),
            seed=int(seed),
            power_mode=normalized_power_mode,
        )
        output_prefix = f"heuristic_{policy_name}_{normalized_power_mode}"
        source_algorithm = None
        algo_cfg = None
    else:
        overrides = dict(algo_overrides or {})
        overrides["seed"] = int(seed)
        overrides["device"] = str(device)
        if algorithm in {"iql", "vdn", "qmix", "qplex"}:
            overrides["num_envs"] = int(num_envs)
            overrides["start_method"] = str(start_method)
        if algorithm == "qmix" and callback_overrides is not None:
            overrides["callbacks"] = list(callback_overrides)

        algo_cfg = load_algo_config(algorithm, overrides=overrides, env_cfg=env_cfg)
        trainer = build_trainer(algorithm, env_cfg=env_cfg, algo_cfg=algo_cfg, device=resolved_device)
        checkpoint = load_trainer_state_dict(
            trainer,
            checkpoint_path,
            algorithm,
            device=resolved_device,
            strict=True,
            load_optimizers=False,
        )
        _set_eval_mode(trainer)

        if algorithm == "qmix":
            callbacks = build_callbacks(getattr(algo_cfg, "callbacks", []), env_cfg=env_cfg, algo_cfg=algo_cfg)
            callbacks.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=int(num_envs))
            load_callback_states(callbacks.callbacks, checkpoint.get("callbacks", {}), strict=True)
            callback_states = callbacks.state_dict()
        else:
            callback_states = {}

        evaluator_kwargs = {"deterministic": bool(deterministic)} if algorithm == "mappo" else {}
        evaluator = build_evaluator(
            algorithm,
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            trainer=trainer,
            **evaluator_kwargs,
        )
        output_prefix = algorithm
        source_algorithm = str(checkpoint.get("algorithm", "unknown"))

    vecenv = vecenv_factory(
        int(num_envs),
        p_trans=p_trans_fixed,
        start_method=str(start_method),
        seed=int(seed),
        env_overrides=env_overrides,
    )
    try:
        metrics = _run_eval_loop(
            algorithm=algorithm,
            evaluator=evaluator,
            vecenv=vecenv,
            env_cfg=env_cfg,
            callbacks=callbacks,
            p_trans=p_trans_fixed,
            episodes=int(episodes),
            steps=n_steps,
        )
    finally:
        vecenv.close()

    output_dir = None
    if not no_save:
        run_config = {
            "algorithm": output_prefix,
            "mode": algorithm,
            "weights": None if checkpoint_path is None else str(checkpoint_path),
            "source_algorithm": source_algorithm,
            "policy_name": str(policy_name) if algorithm == "heuristic" else None,
            "requested_power_mode": str(power_mode) if algorithm == "heuristic" else None,
            "power_mode": normalized_power_mode if algorithm == "heuristic" else None,
            "seed": int(seed),
            "num_envs": int(num_envs),
            "start_method": str(start_method),
            "device": str(resolved_device),
            "deterministic": bool(deterministic),
            "callbacks": list(getattr(algo_cfg, "callbacks", [])) if algo_cfg is not None else [],
            "evaluation_only": True,
            **env_run_summary(env_cfg, _default_env_yaml_path(), overrides=env_overrides),
        }
        _, _, output_dir = save_training_data(
            algorithm=output_prefix,
            reward_history=metrics["reward"],
            success_rate_history=metrics["success_rate"],
            energy_history=metrics["energy"],
            jump_history=metrics["jump"],
            n_episode=int(episodes),
            n_steps=n_steps,
            run_config=run_config,
            output_root=None if output_root is None else Path(output_root),
            artifact_kind="eval",
        )

    return EvaluationResult(
        trainer=trainer,
        metrics=metrics,
        output_dir=output_dir,
        callback_states=callback_states,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 6 UAV-Ultra evaluation runner")
    parser.add_argument("algorithm", choices=sorted(_EVAL_ALGOS))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-method", choices=["spawn", "fork", "forkserver"], default="spawn")
    parser.add_argument("--device", type=str, default="auto")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--callback", action="append", dest="callbacks", default=None)
    parser.add_argument(
        "--policy",
        choices=["random", "greedy_sensing", "max_csi", "min_interference"],
        default=None,
    )
    parser.add_argument(
        "--power-mode",
        choices=["quality_adaptive", "fixed_mid", "fixed_low", "random"],
        default=None,
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-root", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    algorithm = str(args.algorithm).lower()
    if algorithm != "heuristic" and args.policy is not None:
        parser.error("--policy is only valid for heuristic evaluation")
    if algorithm != "heuristic" and args.power_mode is not None:
        parser.error("--power-mode is only valid for heuristic evaluation")
    if algorithm == "heuristic" and args.checkpoint is not None:
        parser.error("--checkpoint is not valid for heuristic evaluation")
    if algorithm != "qmix" and args.callbacks:
        parser.error("--callback is only valid for qmix evaluation")

    result = run_evaluation(
        algorithm,
        checkpoint_path=args.checkpoint,
        episodes=int(args.episodes),
        steps=args.steps,
        num_envs=int(args.num_envs),
        seed=int(args.seed),
        device=str(args.device),
        start_method=str(args.start_method),
        deterministic=bool(args.deterministic),
        policy_name=args.policy or _DEFAULT_POLICY,
        power_mode=args.power_mode or _DEFAULT_POWER_MODE,
        callback_overrides=args.callbacks,
        no_save=bool(args.no_save),
        output_root=args.output_root,
    )
    print(f"Evaluation completed for {algorithm}. Episodes: {len(result.metrics['reward'])}")
    if result.output_dir is not None:
        print(f"Output: {result.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationResult",
    "run_evaluation",
    "build_arg_parser",
    "_run_eval_loop",
]
