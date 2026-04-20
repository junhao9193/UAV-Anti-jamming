"""Unified evaluation protocol for MP-DQN-family and heuristic baselines.

This script evaluates fixed policies without learning or exploration.  Use it for
paper main tables so learned baselines and heuristic baselines share the same
statistics protocol.

Examples from `UAV-Jammer-RL/`:
  python -m Main.evaluate.evaluate_mpdqn --mode mpdqn --weights ../Draw/experiment-data/mpdqn_qmix_xxx/mpdqn_qmix_weights.pth
  python -m Main.evaluate.evaluate_mpdqn --mode heuristic --heuristic-policy greedy_sensing
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from envs import Environ
from Main.common import (
    SubprocVecEnv,
    env_run_config,
    make_fixed_p_trans,
    resolve_env_config_path,
    save_training_data,
    validate_positive_run_args,
)
from tqdm.auto import trange


def _load_mpdqn_agents(weights_path: str | Path, *, env0: Environ, device: str):
    import torch

    from algorithms.mpdqn.agent import MPDQNAgent

    ckpt = torch.load(str(weights_path), map_location=device)
    agents_sd = ckpt.get("agents", None)
    if not isinstance(agents_sd, list) or len(agents_sd) == 0:
        raise ValueError(f"Checkpoint does not contain MP-DQN agents: {weights_path}")
    if len(agents_sd) != int(env0.n_ch):
        raise ValueError(f"Checkpoint agents={len(agents_sd)} != env n_ch={env0.n_ch}")
    agent_cfg = ckpt.get("agent_config", None)
    if isinstance(agent_cfg, dict):
        expected = {
            "state_dim": int(env0.state_dim),
            "n_actions": int(env0.action_dim),
            "param_dim": int(env0.param_dim_per_action),
        }
        for key, expected_val in expected.items():
            if key in agent_cfg and int(agent_cfg[key]) != int(expected_val):
                raise ValueError(
                    f"Checkpoint agent_config[{key}]={agent_cfg[key]} does not match "
                    f"current env {key}={expected_val}. Check --config-path / env.yaml."
                )

    agents = []
    for sd in agents_sd:
        agent = MPDQNAgent(
            state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            buffer_capacity=1,
            batch_size=1,
            device=str(device),
            use_amp=False,
        )
        agent.actor.load_state_dict(sd["actor"])
        agent.q_net.load_state_dict(sd["q_net"])
        agent.target_actor.load_state_dict(sd.get("target_actor", sd["actor"]))
        agent.target_q_net.load_state_dict(sd.get("target_q_net", sd["q_net"]))
        agent.actor.eval()
        agent.q_net.eval()
        agents.append(agent)
    return agents, ckpt


def evaluate_policy(
    *,
    mode: str = "mpdqn",
    weights: str | None = None,
    heuristic_policy: str = "greedy_sensing",
    power_mode: str = "quality_adaptive",
    algorithm_name: str | None = None,
    n_episode: int = 100,
    n_steps: int = 1000,
    num_envs: int = 32,
    device: str | None = None,
    save_data: bool = True,
    start_method: str = "spawn",
    seed: int = 0,
    config_path: str | None = None,
) -> dict[str, list[float]]:
    import torch

    from algorithms.heuristic import HeuristicDims, build_heuristic_policy, normalize_power_mode

    mode = str(mode).lower()
    validate_positive_run_args(n_episode=n_episode, n_steps=n_steps, num_envs=num_envs)
    if mode == "mpdqn" and weights is None:
        raise ValueError("--weights is required when --mode mpdqn")
    if mode not in {"mpdqn", "heuristic"}:
        raise ValueError(f"Unknown mode: {mode!r}; expected 'mpdqn' or 'heuristic'")
    requested_power_mode = str(power_mode)
    if mode == "heuristic":
        power_mode = normalize_power_mode(heuristic_policy, power_mode)
    config_path = resolve_env_config_path(config_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env0 = Environ(config_path=config_path)
    p_trans_fixed = make_fixed_p_trans(env0)

    ckpt: dict[str, Any] | None = None
    agents = None
    policies = None
    if mode == "mpdqn":
        agents, ckpt = _load_mpdqn_agents(weights, env0=env0, device=str(device))
        if algorithm_name is None:
            algorithm_name = f"eval_{ckpt.get('algorithm', 'mpdqn_policy')}"
    elif mode == "heuristic":
        dims = HeuristicDims(
            n_channel=int(env0.n_channel),
            n_des=int(env0.n_des),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
        )
        policies = [
            build_heuristic_policy(heuristic_policy, dims, seed=int(seed) + 1009 * i, power_mode=power_mode)
            for i in range(int(env0.n_ch))
        ]
        if algorithm_name is None:
            algorithm_name = f"eval_baseline_heuristic_{heuristic_policy}_{power_mode}"

    vecenv = SubprocVecEnv(
        int(num_envs),
        config_path=config_path,
        p_trans=p_trans_fixed,
        start_method=str(start_method),
        seed=int(seed),
    )

    reward_history: list[float] = []
    success_rate_history: list[float] = []
    energy_history: list[float] = []
    jump_history: list[float] = []

    n_envs = int(num_envs)
    n_agents = int(env0.n_ch)

    pbar = trange(int(n_episode), desc=f"Evaluating({algorithm_name})", unit="ep", ascii=True)
    try:
        for _episode in pbar:
            states = vecenv.reset()
            episode_reward = 0.0
            steps_done = 0

            for _step in range(int(n_steps)):
                if mode == "mpdqn":
                    action_discrete_all = np.zeros((n_envs, n_agents), dtype=np.int32)
                    action_params_all = np.zeros((n_envs, n_agents, int(env0.total_param_dim)), dtype=np.float32)
                    for i in range(n_agents):
                        ad, ap = agents[i].select_action_batch(states[:, i, :], epsilon=0.0)
                        action_discrete_all[:, i] = ad
                        action_params_all[:, i, :] = ap
                    actions = [
                        [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(n_agents)]
                        for e in range(n_envs)
                    ]
                else:
                    actions = []
                    for e in range(n_envs):
                        env_actions = []
                        for i in range(n_agents):
                            env_actions.append(policies[i].select_action(states[e, i, :]))
                        actions.append(env_actions)

                next_states, rewards, dones, infos = vecenv.step(actions)
                states = next_states
                episode_reward += float(np.mean(rewards))
                steps_done += 1

            steps_done = max(1, int(steps_done))
            total_links = float(steps_done * n_envs * n_agents * int(env0.n_des))
            energy_arr, jump_arr, suc_arr = vecenv.get_metrics()
            avg_energy = float(np.sum(energy_arr)) / total_links
            avg_jump = float(np.sum(jump_arr)) / total_links
            avg_suc_per_link = float(np.sum(suc_arr)) / total_links
            success_rate = float(np.clip((avg_suc_per_link + 3.0) / 4.0, 0.0, 1.0))

            reward_history.append(episode_reward)
            success_rate_history.append(success_rate)
            energy_history.append(avg_energy)
            jump_history.append(avg_jump)

            recent_window = min(100, len(reward_history))
            pbar.set_postfix(
                {
                    "avg_r": f"{float(np.mean(reward_history[-recent_window:])):.3f}",
                    "sr": f"{float(np.mean(success_rate_history[-recent_window:])):.3f}",
                    "envs": str(n_envs),
                }
            )
    finally:
        vecenv.close()

    if save_data:
        save_training_data(
            algorithm=str(algorithm_name),
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=int(n_episode),
            n_steps=int(n_steps),
            trainer=None,
            run_config={
                "algorithm": str(algorithm_name),
                "mode": str(mode),
                "weights": None if weights is None else str(weights),
                "source_algorithm": None if ckpt is None else str(ckpt.get("algorithm", "unknown")),
                "heuristic_policy": str(heuristic_policy),
                "requested_power_mode": requested_power_mode,
                "power_mode": str(power_mode),
                "seed": int(seed),
                "num_envs": int(num_envs),
                "start_method": str(start_method),
                "device": str(device),
                "epsilon": 0.0 if mode == "mpdqn" else None,
                "evaluation_only": True,
                **env_run_config(env0, config_path),
            },
            artifact_kind="eval",
        )

    return {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified evaluation for MP-DQN-family and heuristic baselines")
    parser.add_argument("--mode", type=str, default="mpdqn", choices=["mpdqn", "heuristic"])
    parser.add_argument("--weights", type=str, default=None, help="MP-DQN-family checkpoint path for --mode mpdqn")
    parser.add_argument("--algorithm-name", type=str, default=None, help="Override output algorithm prefix")
    parser.add_argument("--heuristic-policy", type=str, default="greedy_sensing", choices=["random", "greedy_sensing", "max_csi", "min_interference"])
    parser.add_argument("--power-mode", type=str, default="quality_adaptive", choices=["quality_adaptive", "fixed_mid", "fixed_low", "random"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-path", type=str, default=None, help="Env YAML config path (default: configs/env.yaml)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_policy(
        mode=str(args.mode),
        weights=args.weights,
        heuristic_policy=str(args.heuristic_policy),
        power_mode=str(args.power_mode),
        algorithm_name=args.algorithm_name,
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        device=args.device,
        save_data=not bool(args.no_save),
        start_method=str(args.start_method),
        seed=int(args.seed),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
