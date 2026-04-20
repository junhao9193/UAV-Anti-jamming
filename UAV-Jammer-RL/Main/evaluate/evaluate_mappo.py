"""Unified fixed-horizon evaluation for MAPPO checkpoints.

Examples from `UAV-Jammer-RL/`:
  python -m Main.evaluate.evaluate_mappo --weights ../Draw/experiment-data/mappo/mappo_weights.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def _load_mappo_agent(weights_path: str | Path, *, env0: Environ, device: str):
    import torch

    from algorithms.mappo import MAPPOAgent

    ckpt = torch.load(str(weights_path), map_location=device)
    if "actor" not in ckpt or "critic" not in ckpt:
        raise ValueError(f"Checkpoint does not look like a MAPPO checkpoint: {weights_path}")

    agent_cfg = ckpt.get("agent_config", None)
    if not isinstance(agent_cfg, dict):
        raise ValueError(f"MAPPO checkpoint missing agent_config: {weights_path}")

    expected = {
        "obs_dim": int(env0.state_dim),
        "n_actions": int(env0.action_dim),
        "cont_dim": int(env0.param_dim_per_action),
        "n_agents": int(env0.n_ch),
        "global_state_dim": int(env0.state_dim * env0.n_ch),
    }
    for key, expected_val in expected.items():
        if key in agent_cfg and int(agent_cfg[key]) != int(expected_val):
            raise ValueError(
                f"Checkpoint agent_config[{key}]={agent_cfg[key]} does not match "
                f"current env {key}={expected_val}. Check --config-path / env.yaml."
            )

    agent = MAPPOAgent(
        obs_dim=int(expected["obs_dim"]),
        n_actions=int(expected["n_actions"]),
        cont_dim=int(expected["cont_dim"]),
        n_agents=int(expected["n_agents"]),
        global_state_dim=int(expected["global_state_dim"]),
        device=str(device),
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()
    return agent, ckpt


def evaluate_mappo(
    *,
    weights: str,
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

    validate_positive_run_args(n_episode=n_episode, n_steps=n_steps, num_envs=num_envs)
    if not weights:
        raise ValueError("--weights is required for MAPPO evaluation")
    config_path = resolve_env_config_path(config_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env0 = Environ(config_path=config_path)
    p_trans_fixed = make_fixed_p_trans(env0)
    agent, ckpt = _load_mappo_agent(weights, env0=env0, device=str(device))
    if algorithm_name is None:
        algorithm_name = f"eval_{ckpt.get('algorithm', 'mappo')}"

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
    cont_dim = int(env0.param_dim_per_action)

    pbar = trange(int(n_episode), desc=f"Evaluating({algorithm_name})", unit="ep", ascii=True)
    try:
        for _episode in pbar:
            states = vecenv.reset()  # (E,N,S)
            episode_reward = 0.0
            steps_done = 0

            for _step in range(int(n_steps)):
                actions = []
                for e in range(n_envs):
                    obs_e = states[e]
                    global_state_e = np.concatenate(obs_e, axis=-1).astype(np.float32)
                    env_actions = []
                    for i in range(n_agents):
                        res = agent.act(obs_e[i], global_state_e, agent_id=i, deterministic=True)
                        params_full = np.zeros((int(env0.total_param_dim),), dtype=np.float32)
                        start = int(res.action_discrete) * cont_dim
                        params_full[start : start + cont_dim] = res.action_cont
                        env_actions.append((int(res.action_discrete), params_full))
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
                "mode": "mappo",
                "weights": str(weights),
                "source_algorithm": str(ckpt.get("algorithm", "mappo")),
                "seed": int(seed),
                "num_envs": int(num_envs),
                "start_method": str(start_method),
                "device": str(device),
                "deterministic": True,
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
    parser = argparse.ArgumentParser(description="Unified fixed-horizon evaluation for MAPPO")
    parser.add_argument("--weights", type=str, required=True, help="MAPPO checkpoint path")
    parser.add_argument("--algorithm-name", type=str, default=None, help="Override output algorithm prefix")
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
    evaluate_mappo(
        weights=str(args.weights),
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
