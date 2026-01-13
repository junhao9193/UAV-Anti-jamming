from __future__ import division

from collections import deque

import numpy as np

class ReplayMemory(object):
    def __init__(self, max_epi_num=50, max_epi_len=300):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=2000) 
        self.sample_length = 32

    def reset(self):
        self.memory.clear()

    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    def sample(self, batch_size=32):
        if self.is_available():
            epi_index = np.random.choice(range(len(self.memory)), batch_size, replace=False)
            data = [self.memory[i] for i in epi_index]
            return data
        else:
            raise ValueError('Memory is not available')

    def size(self):
        return len(self.memory)

    def is_available(self):
        return len(self.memory) >= self.sample_length


__all__ = ["ReplayMemory"]
