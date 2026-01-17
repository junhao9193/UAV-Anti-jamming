from __future__ import division

import numpy as np

from envs import Environ
from tqdm.auto import trange


def train_mappo(n_episode=2000, rollout_steps=256):
    """MAPPO (centralized critic, parameter-sharing) training."""
    env = Environ()

    from algorithms import MAPPOAgent
    from algorithms.mappo import RolloutBuffer

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
        minibatch_size=min(256, rollout_steps * n_agents),
    )

    reward_history = []

    pbar = trange(n_episode, desc="Training(MAPPO)", unit="ep", ascii=True)
    for episode in pbar:
        state = env.reset(env.generate_p_trans())
        buffer = RolloutBuffer(n_agents=n_agents)

        episode_reward = 0.0

        for _ in range(int(rollout_steps)):
            obs_step = np.stack(state, axis=0).astype(np.float32)  # (n_agents, obs_dim)
            global_state = np.concatenate(state, axis=-1).astype(np.float32)  # (global_dim,)
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
            if done:
                break

        # Bootstrap values for GAE
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

        reward_history.append(episode_reward)
        recent_window = min(100, len(reward_history))
        avg_reward = float(np.mean(reward_history[-recent_window:]))

        pbar.set_postfix(
            {
                "avg_r": f"{avg_reward:.3f}",
                "loss_pi": f"{update_info['loss_pi']:.3f}",
                "loss_v": f"{update_info['loss_v']:.3f}",
                "ent": f"{update_info['entropy']:.3f}",
            }
        )

    return agent, reward_history


def train_mpdqn(n_episode=1500, n_steps=1000):
    """MP-DQN (QMIX) 全局联合训练：team reward + joint replay + mixing network."""
    env = Environ()

    from algorithms import MPDQNQMIXTrainer

    trainer = MPDQNQMIXTrainer(
        n_agents=int(env.n_ch),
        state_dim=int(env.state_dim),
        n_actions=int(env.action_dim),
        param_dim=int(env.param_dim_per_action),
        global_state_dim=int(env.state_dim * env.n_ch),
    )

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    reward_history = []

    pbar = trange(n_episode, desc="Training(MP-DQN-QMIX)", unit="ep", ascii=True)
    for episode in pbar:
        state = env.reset(env.generate_p_trans())
        episode_reward = 0.0
        loss_q_sum = 0.0
        loss_actor_sum = 0.0
        loss_count = 0

        for step in range(n_steps):
            actions = trainer.select_actions(state, epsilon)

            next_state, rewards, done, info = env.step(actions)

            trainer.store_transition(
                states=state,
                actions=actions,
                rewards=np.asarray(rewards, dtype=np.float32),
                next_states=next_state,
                done=bool(done),
            )
            loss_info = trainer.train_step()
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

    return trainer, reward_history


def train_mpdqn_iql(n_episode=1500, n_steps=1000):
    """MP-DQN (IQL) 训练主函数（每个 agent 独立学习）。"""
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

    pbar = trange(n_episode, desc="Training(MP-DQN-IQL)", unit="ep", ascii=True)
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
    train_mappo()
