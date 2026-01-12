from __future__ import division

class UAV:
    def __init__(self, start_position, start_direction, start_velocity, start_p):
        self.position = start_position
        self.direction = start_direction
        self.velocity = start_velocity
        self.p = start_p    # 无人机与地平面的夹角度
        self.uav_velocity = []
        self.uav_direction = []
        self.uav_p = []
        self.destinations = []
        self.connections = []

class Jammer:
    def __init__(self, start_position, start_direction, velocity, start_p):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.p = start_p # 无人机与地平面的夹角度
        self.jammer_velocity = []
        self.jammer_direction = []
        self.jammer_p = []

class RP:
    def __init__(self, start_position):
        self.position = start_position
        self.connections = []
