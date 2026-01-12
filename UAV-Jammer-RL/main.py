from __future__ import division

import numpy as np

from envs import Environ
from algorithms import Agent


def train(n_episode=1500, n_steps=1000, max_epi_num=200):
    """DDRQN 训练主函数"""
    env = Environ()

    # 在外部创建 Agent（解耦后的设计）
    agents = [
        Agent(i, env.state_dim, env.action_dim, max_epi_num=max_epi_num, max_epi_len=n_steps)
        for i in range(env.n_ch)
    ]

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    reward_history = []

    for episode in range(n_episode):
        state = env.reset(env.generate_p_trans())
        hiddens = [None] * env.n_ch
        episode_reward = 0

        for step in range(n_steps):
            actions = []
            new_hiddens = []

            # 每个 agent 选择动作
            for i in range(env.n_ch):
                action, new_hidden = agents[i].get_action(state[i], hiddens[i], epsilon)
                actions.append(action)
                new_hiddens.append(new_hidden)

            # 环境执行动作
            next_state, rewards, done, info = env.step(actions)

            # 存储经验并训练
            for i in range(env.n_ch):
                agents[i].remember(state[i], actions[i], rewards[i], next_state[i])
                agents[i].train(hiddens[i])

            state = next_state
            hiddens = new_hiddens
            episode_reward += np.mean(rewards)

            if done:
                break

        # 衰减 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        reward_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode + 1}/{n_episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")

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


def get_params(agents):
    """获取所有 agent 的参数"""
    return [agent.get_params() for agent in agents]


def load_params(agents, params):
    """加载所有 agent 的参数"""
    for i, agent in enumerate(agents):
        agent.load_params(params[i])


if __name__ == "__main__":
    test()
