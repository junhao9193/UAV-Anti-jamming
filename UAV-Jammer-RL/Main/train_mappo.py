"""
MAPPO training script (shared actor + centralized critic).

Run from `UAV-Jammer-RL/`:
  python -m Main.train_mappo
"""
from __future__ import division

import argparse

import numpy as np

from envs import Environ
from Main.common import make_fixed_p_trans, save_training_data
from tqdm.auto import trange


def train_mappo(
    n_episode: int = 1500,
    n_steps: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    update_epochs: int = 10,
    minibatch_size: int = 256,
    max_grad_norm: float = 0.5,
    device: str | None = None,
    save_data: bool = True,
    seed: int = 0,
):
    """Train MAPPO with a shared hybrid actor and centralized value function."""
    import torch

    from algorithms.mappo import MAPPOAgent, RolloutBuffer

    def _configure_torch(dev: str) -> None:
        if dev.startswith("cuda"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_torch(str(device))

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env = Environ()
    p_trans_fixed = make_fixed_p_trans(env)
    env.set_p(p_trans_fixed)

    n_agents = int(env.n_ch)
    obs_dim = int(env.state_dim)
    global_state_dim = int(env.state_dim * env.n_ch)
    cont_dim = int(env.param_dim_per_action)

    agent = MAPPOAgent(
        obs_dim=obs_dim,
        n_actions=int(env.action_dim),
        cont_dim=cont_dim,
        n_agents=n_agents,
        global_state_dim=global_state_dim,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_range=float(clip_range),
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        lr=float(lr),
        update_epochs=int(update_epochs),
        minibatch_size=min(int(minibatch_size), int(n_steps) * n_agents),
        max_grad_norm=float(max_grad_norm),
        device=str(device),
    )

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []

    pbar = trange(n_episode, desc="Training(MAPPO)", unit="ep", ascii=True)
    for episode in pbar:
        state = env.reset(p_trans_fixed)
        env.clear_reward()
        buffer = RolloutBuffer(n_agents=n_agents)
        episode_reward = 0.0
        steps_done = 0

        for step in range(int(n_steps)):
            obs_step = np.stack(state, axis=0).astype(np.float32)  # (N,S)
            global_state = np.concatenate(state, axis=-1).astype(np.float32)  # (Ds,)
            global_step = np.tile(global_state, (n_agents, 1)).astype(np.float32)
            agent_ids = np.arange(n_agents, dtype=np.int64)

            actions = []
            act_discrete = np.zeros((n_agents,), dtype=np.int64)
            act_cont = np.zeros((n_agents, cont_dim), dtype=np.float32)
            log_probs = np.zeros((n_agents,), dtype=np.float32)
            values = np.zeros((n_agents,), dtype=np.float32)

            for i in range(n_agents):
                res = agent.act(obs_step[i], global_state, agent_id=i, deterministic=False)
                params_full = np.zeros((int(env.total_param_dim),), dtype=np.float32)
                start = int(res.action_discrete) * cont_dim
                params_full[start : start + cont_dim] = res.action_cont

                actions.append((int(res.action_discrete), params_full))
                act_discrete[i] = int(res.action_discrete)
                act_cont[i] = res.action_cont
                log_probs[i] = float(res.log_prob)
                values[i] = float(res.value)

            next_state, rewards, done, info = env.step(actions)
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
        last_values = agent.value(global_last, np.arange(n_agents, dtype=np.int64))

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=last_values,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )
        batch = buffer.as_batch(returns=returns, advantages=advantages)
        update_info = agent.update(batch)

        steps_done = max(1, int(steps_done))
        total_links = float(steps_done * int(env.n_ch) * int(env.n_des))
        avg_energy = float(env.rew_energy) / total_links
        avg_jump = float(env.rew_jump) / total_links
        avg_suc_per_link = float(env.rew_suc) / total_links
        success_rate = float(np.clip((avg_suc_per_link + 3.0) / 4.0, 0.0, 1.0))

        reward_history.append(episode_reward)
        success_rate_history.append(success_rate)
        energy_history.append(avg_energy)
        jump_history.append(avg_jump)

        recent_window = min(100, len(reward_history))
        avg_reward = float(np.mean(reward_history[-recent_window:]))
        avg_sr = float(np.mean(success_rate_history[-recent_window:]))
        pbar.set_postfix(
            {
                "avg_r": f"{avg_reward:.3f}",
                "sr": f"{avg_sr:.3f}",
                "loss_pi": f"{update_info['loss_pi']:.3f}",
                "loss_v": f"{update_info['loss_v']:.3f}",
                "ent": f"{update_info['entropy']:.3f}",
            }
        )

    if save_data:
        save_training_data(
            algorithm="mappo",
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=n_episode,
            n_steps=n_steps,
            trainer=agent,
        )

    return agent, {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--rollout-steps",
        dest="steps",
        type=int,
        help="Compatibility alias for --steps",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics/weights")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting MAPPO Training")
    print("=" * 60)
    train_mappo(
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        lr=float(args.lr),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        max_grad_norm=float(args.max_grad_norm),
        device=args.device,
        save_data=not bool(args.no_save),
        seed=int(args.seed),
    )
    print("Training completed!")
