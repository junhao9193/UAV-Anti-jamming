"""
MP-DQN (IQL) 独立训练脚本
每个 agent 独立学习，无全局协作
"""
from __future__ import division

import numpy as np

from envs import Environ
from algorithms.mpdqn.agent import MPDQNAgent
from Main.common import make_fixed_p_trans, save_training_data
from tqdm.auto import trange


def train_mpdqn_iql(n_episode=1500, n_steps=1000, save_data=True):
    """MP-DQN (IQL) 训练主函数（每个 agent 独立学习）。"""
    env = Environ()
    p_trans_fixed = make_fixed_p_trans(env)

    agents = [
        MPDQNAgent(
            state_dim=env.state_dim,
            n_actions=env.action_dim,
            param_dim=env.param_dim_per_action,
        )
        for _ in range(env.n_ch)
    ]

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []

    pbar = trange(n_episode, desc="Training(MP-DQN-IQL)", unit="ep", ascii=True)
    for episode in pbar:
        state = env.reset(p_trans_fixed)
        env.clear_reward()
        episode_reward = 0.0
        loss_q_sum = 0.0
        loss_actor_sum = 0.0
        loss_count = 0
        steps_done = 0

        for step in range(n_steps):
            actions = []

            for i in range(env.n_ch):
                action_discrete, action_params = agents[i].select_action(state[i], epsilon)
                actions.append((action_discrete, action_params))

            next_state, rewards, done, info = env.step(actions)

            for i in range(env.n_ch):
                agents[i].store(
                    state=state[i],
                    action_discrete=actions[i][0],
                    action_params=actions[i][1],
                    reward=float(rewards[i]),
                    next_state=next_state[i],
                    done=bool(done),
                )
                loss_info = agents[i].train_step()
                if loss_info is not None:
                    loss_q_sum += float(loss_info["loss_q"])
                    loss_actor_sum += float(loss_info["loss_actor"])
                    loss_count += 1

            state = next_state
            episode_reward += np.mean(rewards)
            steps_done += 1

            if done:
                break

        steps_done = max(1, int(steps_done))
        total_links = float(steps_done * int(env.n_ch) * int(env.n_des))

        avg_energy = float(env.rew_energy) / total_links
        avg_jump = float(env.rew_jump) / total_links
        avg_suc_per_link = float(env.rew_suc) / total_links

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
        }
        if loss_count > 0:
            postfix["loss_q"] = f"{(loss_q_sum / loss_count):.3f}"
            postfix["loss_a"] = f"{(loss_actor_sum / loss_count):.3f}"
        pbar.set_postfix(postfix)

    if save_data:
        save_training_data(
            algorithm="mpdqn_iql",
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=n_episode,
            n_steps=n_steps,
        )

    return agents, {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Starting MP-DQN IQL Training")
    print("=" * 60)
    agents, metrics = train_mpdqn_iql(n_episode=1500, n_steps=1000, save_data=True)
    print("Training completed!")
