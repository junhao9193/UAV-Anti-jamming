from __future__ import division

import numpy as np

from envs import Environ
from tqdm.auto import trange


def train_mpdqn(n_episode=1500, n_steps=1000):
    """MP-DQN 训练主函数（信道离散 + 功率连续）"""
    env = Environ()

    from algorithms import MPDQNAgent

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

    pbar = trange(n_episode, desc="Training", unit="ep")
    for episode in pbar:
        state = env.reset(env.generate_p_trans())
        episode_reward = 0.0
        loss_q_sum = 0.0
        loss_actor_sum = 0.0
        loss_count = 0

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

            if done:
                break

        # 衰减 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        reward_history.append(episode_reward)

        recent_window = min(100, len(reward_history))
        avg_reward = float(np.mean(reward_history[-recent_window:]))
        postfix = {
            "avg_r": f"{avg_reward:.3f}",
            "eps": f"{epsilon:.3f}",
        }
        if loss_count > 0:
            postfix["loss_q"] = f"{(loss_q_sum / loss_count):.3f}"
            postfix["loss_a"] = f"{(loss_actor_sum / loss_count):.3f}"
        pbar.set_postfix(postfix)

    return agents, reward_history


def test():
    """简单测试环境是否可运行"""
    env = Environ()
    state = env.reset(env.generate_p_trans())

    for _ in range(10):
        actions = [space.sample() for space in env.action_space]
        state, reward, done, info = env.step(actions)
        if done:
            break

    print("Test OK, state_dim:", env.state_dim, "action_dim:", env.action_dim)


if __name__ == "__main__":
    train_mpdqn()
