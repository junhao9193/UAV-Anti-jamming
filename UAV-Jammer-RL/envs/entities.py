class UAV:
    def __init__(self, start_position, start_direction, start_velocity, start_p):
        self.position = start_position
        self.direction = start_direction
        self.velocity = start_velocity
        self.p = start_p    # 无人机与地平面的夹角度
        self.mean_velocity = start_velocity
        self.mean_direction = start_direction
        self.mean_p = start_p
        self.destinations = []
        self.connections = []

class Jammer:
    def __init__(self, start_position, start_direction, velocity, start_p):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.p = start_p # 无人机与地平面的夹角度
        self.mean_velocity = velocity
        self.mean_direction = start_direction
        self.mean_p = start_p

class RP:
    def __init__(self, start_position):
        self.position = start_position
        self.connections = []
