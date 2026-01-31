"""
MP-DQN (QMIX) 联合训练脚本
全局联合训练：team reward + joint replay + mixing network
"""
from __future__ import division

import argparse

import numpy as np

from envs import Environ
from Main.common import SubprocVecEnv, make_fixed_p_trans, save_training_data
from tqdm.auto import trange


def train_mpdqn_qmix(
    n_episode: int = 1500,
    n_steps: int = 1000,
    num_envs: int = 32,
    batch_size: int = 256,
    buffer_capacity: int = 200_000,
    learn_every: int = 4,
    updates_per_learn: int = 1,
    lr_actor: float = 1e-3,
    lr_q: float = 1e-3,
    lr_mixer: float | None = None,
    max_grad_norm: float = 10.0,
    use_amp: bool = True,
    device: str | None = None,
    save_data: bool = True,
    start_method: str = "spawn",
):
    """MP-DQN (QMIX) 全局联合训练：team reward + joint replay + mixing network."""
    # NOTE: delay torch imports so that SubprocVecEnv workers don't import torch/CUDA on spawn.
    import torch

    from algorithms.mpdqn.qmix.trainer import MPDQNQMIXTrainer

    def _configure_torch(dev: str) -> None:
        if dev.startswith("cuda"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    env0 = Environ()
    p_trans_fixed = make_fixed_p_trans(env0)
    vecenv = SubprocVecEnv(
        int(num_envs),
        p_trans=p_trans_fixed,
        start_method=str(start_method),
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_torch(device)
    use_amp = bool(use_amp) and device.startswith("cuda")

    trainer = MPDQNQMIXTrainer(
        n_agents=int(env0.n_ch),
        state_dim=int(env0.state_dim),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        global_state_dim=int(env0.state_dim * env0.n_ch),
        buffer_capacity=int(buffer_capacity),
        batch_size=int(batch_size),
        lr_actor=float(lr_actor),
        lr_q=float(lr_q),
        lr_mixer=(float(lr_mixer) if lr_mixer is not None else None),
        use_amp=use_amp,
        max_grad_norm=float(max_grad_norm),
        device=device,
    )

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []

    n_envs = int(num_envs)
    n_agents = int(env0.n_ch)
    state_dim = int(env0.state_dim)

    pbar = trange(n_episode, desc="Training(MP-DQN-QMIX)", unit="ep", ascii=True)
    try:
        for episode in pbar:
            states = vecenv.reset()  # (E,N,S)
            if states.shape != (n_envs, n_agents, state_dim):
                raise RuntimeError(f"Unexpected state shape: {states.shape}, expected ({n_envs},{n_agents},{state_dim})")

            episode_reward = 0.0
            loss_q_sum = 0.0
            loss_actor_sum = 0.0
            loss_count = 0
            steps_done = 0

            for step in range(int(n_steps)):
                action_discrete_all = np.zeros((n_envs, n_agents), dtype=np.int32)
                action_params_all = np.zeros((n_envs, n_agents, int(env0.total_param_dim)), dtype=np.float32)

                for i in range(n_agents):
                    ad, ap = trainer.agents[i].select_action_batch(states[:, i, :], epsilon)
                    action_discrete_all[:, i] = ad
                    action_params_all[:, i, :] = ap

                actions = [
                    [
                        (int(action_discrete_all[e, i]), action_params_all[e, i, :])
                        for i in range(n_agents)
                    ]
                    for e in range(n_envs)
                ]

                # Overlap: start env stepping, then train on GPU while workers simulate.
                vecenv.step_async(actions)

                if (step + 1) % int(max(1, learn_every)) == 0:
                    for _ in range(int(max(1, updates_per_learn))):
                        loss_info = trainer.train_step()
                        if loss_info is not None:
                            loss_q_sum += float(loss_info["loss_q"])
                            loss_actor_sum += float(loss_info["loss_actor"])
                            loss_count += 1

                next_states, rewards, dones, infos = vecenv.step_wait()  # next_states:(E,N,S), rewards:(E,N)

                is_last_step = (step == int(n_steps) - 1)
                for e in range(n_envs):
                    trainer.store_transition(
                        states=states[e],
                        actions=actions[e],
                        rewards=np.asarray(rewards[e], dtype=np.float32),
                        next_states=next_states[e],
                        done=bool(dones[e]) or is_last_step,
                    )

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

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            reward_history.append(episode_reward)
            success_rate_history.append(success_rate)
            energy_history.append(avg_energy)
            jump_history.append(avg_jump)

            recent_window = min(100, len(reward_history))
            avg_reward = float(np.mean(reward_history[-recent_window:]))
            avg_sr = float(np.mean(success_rate_history[-recent_window:]))
            postfix = {
                "avg_r": f"{avg_reward:.3f}",
                "sr": f"{avg_sr:.3f}",
                "eps": f"{epsilon:.3f}",
                "envs": str(n_envs),
            }
            if loss_count > 0:
                postfix["loss_q"] = f"{(loss_q_sum / loss_count):.3f}"
                postfix["loss_a"] = f"{(loss_actor_sum / loss_count):.3f}"
            pbar.set_postfix(postfix)
    finally:
        vecenv.close()

    if save_data:
        save_training_data(
            algorithm="mpdqn_qmix",
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=n_episode,
            n_steps=n_steps,
            trainer=trainer,
        )

    return trainer, {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MP-DQN (QMIX)")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--updates-per-learn", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-q", type=float, default=1e-3)
    parser.add_argument("--lr-mixer", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting MP-DQN QMIX Training")
    print("=" * 60)
    trainer, metrics = train_mpdqn_qmix(
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        learn_every=int(args.learn_every),
        updates_per_learn=int(args.updates_per_learn),
        lr_actor=float(args.lr_actor),
        lr_q=float(args.lr_q),
        lr_mixer=args.lr_mixer,
        max_grad_norm=float(args.max_grad_norm),
        use_amp=not bool(args.no_amp),
        device=args.device,
        save_data=not bool(args.no_save),
        start_method=str(args.start_method),
    )
    print("Training completed!")
