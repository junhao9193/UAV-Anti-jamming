from __future__ import division

import numpy as np

class UAVchannels:
    def __init__(self, n_uav, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_uav = 1.5  # uav antenna height
        self.fc = 2  # 载频2GHz
        self.BS_position = BS_position
        self.n_uav = n_uav
        self.n_channel = n_channel

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    #无人机之间的位置路径损耗
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1**2 + d2**2 + d3**2) + 0.001
        PL_los = 103.8 + 20.9*np.log10(distance*1e-3)
        return PL_los

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)) + 1j * np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))

class Jammerchannels:
    def __init__(self, n_jammer, n_uav, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_jammer = 1.5  # jammer antenna height
        self.h_uav = 1.5  # uav antenna height
        self.BS_position = BS_position
        self.n_jammer = n_jammer
        self.n_uav = n_uav
        self.n_channel = n_channel

    def update_positions(self, positions, uav_positions):
        self.positions = positions
        self.uav_positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.uav_positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.uav_positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.uav_positions[j])

    #position A表示干扰机的位置 position B表示无人机的位置
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1**2 + d2**2 + d3**2) + 0.001
        PL_los = 103.8 + 20.9*np.log10(distance*1e-3)
        return PL_los

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)) +
                              1j * np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))
