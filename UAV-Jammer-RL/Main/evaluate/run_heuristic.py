"""
Evaluate non-learning heuristic baselines.

Run from `UAV-Jammer-RL/`:
  python -m Main.evaluate.run_heuristic --policy greedy_sensing
  python -m Main.evaluate.run_heuristic --policy random --power-mode fixed_mid
"""
from __future__ import division

import argparse

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


def run_heuristic(
    policy_name: str = "greedy_sensing",
    n_episode: int = 300,
    n_steps: int = 1000,
    num_envs: int = 32,
    device: str | None = None,
    save_data: bool = True,
    start_method: str = "spawn",
    seed: int = 0,
    power_mode: str = "quality_adaptive",
    config_path: str | None = None,
):
    from algorithms.heuristic import HeuristicDims, build_heuristic_policy, normalize_power_mode

    del device  # kept for CLI symmetry with train scripts

    validate_positive_run_args(n_episode=n_episode, n_steps=n_steps, num_envs=num_envs)
    np.random.seed(int(seed))
    requested_power_mode = str(power_mode)
    power_mode = normalize_power_mode(policy_name, power_mode)
    config_path = resolve_env_config_path(config_path)

    env0 = Environ(config_path=config_path)
    p_trans_fixed = make_fixed_p_trans(env0)
    vecenv = SubprocVecEnv(
        int(num_envs),
        config_path=config_path,
        p_trans=p_trans_fixed,
        start_method=str(start_method),
        seed=int(seed),
    )

    dims = HeuristicDims(
        n_channel=int(env0.n_channel),
        n_des=int(env0.n_des),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
    )
    policies = [
        build_heuristic_policy(policy_name, dims, seed=int(seed) + 1009 * i, power_mode=power_mode)
        for i in range(int(env0.n_ch))
    ]

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []

    n_envs = int(num_envs)
    n_agents = int(env0.n_ch)

    pbar = trange(n_episode, desc=f"Evaluating({policy_name})", unit="ep", ascii=True)
    try:
        for episode in pbar:
            states = vecenv.reset()
            episode_reward = 0.0
            steps_done = 0

            for step in range(int(n_steps)):
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
            total_energy = float(np.sum(energy_arr))
            total_jump = float(np.sum(jump_arr))
            total_suc = float(np.sum(suc_arr))

            avg_energy = total_energy / total_links
            avg_jump = total_jump / total_links
            avg_suc_per_link = total_suc / total_links
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
            algorithm=f"baseline_heuristic_{policy_name}_{power_mode}",
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=n_episode,
            n_steps=n_steps,
            trainer=None,
            run_config={
                "algorithm": f"baseline_heuristic_{policy_name}_{power_mode}",
                "policy_name": str(policy_name),
                "requested_power_mode": requested_power_mode,
                "power_mode": str(power_mode),
                "seed": int(seed),
                "num_envs": int(num_envs),
                "start_method": str(start_method),
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
    parser = argparse.ArgumentParser(description="Evaluate heuristic baselines")
    parser.add_argument("--policy", type=str, default="greedy_sensing", choices=["random", "greedy_sensing", "max_csi", "min_interference"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Unused, kept for CLI symmetry")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--power-mode", type=str, default="quality_adaptive", choices=["quality_adaptive", "fixed_mid", "fixed_low", "random"])
    parser.add_argument("--config-path", type=str, default=None, help="Env YAML config path (default: configs/env.yaml)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 60)
    print(f"Running heuristic baseline: {args.policy}")
    print("=" * 60)
    run_heuristic(
        policy_name=str(args.policy),
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        device=args.device,
        save_data=not bool(args.no_save),
        start_method=str(args.start_method),
        seed=int(args.seed),
        power_mode=str(args.power_mode),
        config_path=args.config_path,
    )
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
