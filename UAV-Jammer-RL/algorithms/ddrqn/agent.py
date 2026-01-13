from __future__ import division

import random

import numpy as np
import torch

from algorithms.ddrqn.model import DDRQN
from algorithms.ddrqn.replay_memory import ReplayMemory

class Agent:
    def __init__(self, i, state_dim, action_dim, max_epi_num=50, max_epi_len=300):
        self.name = 'agent%d' % i
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.learn_step_counter = 0
        self.replace_target_iter = 100
        self.eval_net = DDRQN(state_dim, action_dim)
        self.target_net = DDRQN(state_dim, action_dim)
        self.N_action = action_dim
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=2e-3)  # 创建一个标准来测量输入x和目标y中每个元素之间的均方误差

    def remember(self, state, action, reward, next_state):
        state = np.array(state)
        next_state = np.array(next_state)
        self.buffer.remember(state, action, reward, next_state)

    def train(self, hidden, batch_size=32):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.buffer.is_available():
            memo = self.buffer.sample(batch_size)
            obs_list = []
            action_list = []
            reward_list = []
            obs_next_list = []
            for i in range(len(memo)):
                obs_list.append(memo[i][0])
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
                obs_next_list.append(memo[i][3])
            obs_list = torch.FloatTensor((np.array(obs_list)).reshape(-1, 1, self.eval_net.input))
            obs_next_list = torch.FloatTensor((np.array(obs_next_list)).reshape(-1, 1, self.eval_net.input))
            q_eval, _ = self.eval_net.forward(obs_list, hidden)
            q_next, _ = self.target_net.forward(obs_next_list, hidden)
            q_eval4next, _ = self.eval_net.forward(obs_next_list, hidden)
            q_eval4next_s = np.squeeze(q_eval4next.detach().numpy())

            q_next_s = q_next.squeeze()

            q_target = q_eval.clone()
            batch_index = np.arange(len(memo), dtype=np.int32)
            max_act4next = np.argmax(q_eval4next_s, axis=-1)
            max_postq = q_next_s[batch_index, max_act4next]

            for t in range(len(memo) - 1):
                q_target[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_postq[t]

            T = len(memo) - 1
            q_target[T, 0, action_list[T]] = reward_list[T]
            loss = self.loss_fn(q_eval, q_target)

            self.optimizer.zero_grad() #梯度初始化为零
            loss.backward() # 反向传播求梯度
            self.optimizer.step() # 更新所有参数

    def get_action(self, obs, hidden, epsilon):
        obs = np.array(obs)
        obs = obs.reshape(-1, 1, self.eval_net.input)
        obs = torch.FloatTensor(obs)#转换为一个向量
        if random.random() > epsilon:
            q, new_hidden = self.eval_net.forward(obs, hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.eval_net.forward(obs, hidden)
            action = random.randint(0, self.N_action-1)
        return action, new_hidden
    
    def get_params(self):
        return (self.eval_net.state_dict(), self.target_net.state_dict())
    
    def load_params(self, params):
        self.eval_net.load_state_dict(params[0])
        self.target_net.load_state_dict(params[0])
