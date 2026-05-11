import numpy as np


def _position_array(positions):
    arr = np.asarray(positions, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr.reshape(-1, 3)


def pathloss_matrix(tx_positions, rx_positions):
    tx = _position_array(tx_positions)
    rx = _position_array(rx_positions)
    diff = tx[:, np.newaxis, :] - rx[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=-1) + 0.001
    return 103.8 + 20.9 * np.log10(distance * 1e-3)


def pathloss_between(position_A, position_B):
    return float(pathloss_matrix([position_A], [position_B])[0, 0])


class UAVchannels:
    def __init__(self, n_uav, n_channel, BS_position):
        del n_channel, BS_position
        # Current pathloss model depends only on 3D transmitter-receiver distance.
        self.n_uav = n_uav

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = pathloss_matrix(self.positions, self.positions)

    #无人机之间的位置路径损耗
    def get_path_loss(self, position_A, position_B):
        return pathloss_between(position_A, position_B)

class Jammerchannels:
    def __init__(self, n_jammer, n_uav, n_channel, BS_position):
        del n_channel, BS_position
        # Current pathloss model depends only on 3D transmitter-receiver distance.
        self.n_jammer = n_jammer
        self.n_uav = n_uav

    def update_positions(self, positions, uav_positions):
        self.positions = positions
        self.uav_positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = pathloss_matrix(self.positions, self.uav_positions)

    #position A表示干扰机的位置 position B表示无人机的位置
    def get_path_loss(self, position_A, position_B):
        return pathloss_between(position_A, position_B)
