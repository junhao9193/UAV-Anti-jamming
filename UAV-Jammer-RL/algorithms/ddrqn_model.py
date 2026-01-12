from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class DDRQN(nn.Module):
    def __init__(self, state, action):
        super(DDRQN, self).__init__()
        #self.ENV = Environ()
        self.lstm_i_dim = state    # input dimension of LSTM
        self.lstm_h_dim = 128    # output dimension of LSTM
        self.lstm_N_layer = 2   # number of layers of LSTM
        self.input = state   #网络总输入:每个无人机簇的状态，8个快衰落（2个成员x4个信道）、6个信道（6个成员）、6个功率（6个成员）、1个干扰机信道状态
        self.output = action     #网络总输出
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(self.input+self.lstm_h_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.output)

    def forward(self, x, hidden):
        h1, new_hidden = self.lstm(x, hidden)
        h2 = F.relu(self.fc1(torch.cat((x, h1), dim=2)))
        h3 = F.relu(self.fc2(h2))
        return self.fc3(h3), new_hidden
