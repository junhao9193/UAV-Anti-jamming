"""无人机与接收点实体。

本模块直接对应旧实现 `UAV-Jammer-RL/envs/entities.py` 中的 `UAV` 与 `RP`。
Stage 1 的目标是先把“数据容器”搬到新包里，不改变任何属性名或初始化语义。
运动更新、链路更新和观测装配仍属于后续 `envs/` 模块，不放在实体类里。
"""


class UAV:
    """簇头无人机实体。

    旧环境把 UAV 当作可变数据容器使用：`core.py` 会直接读写 `position`、
    `direction`、`velocity`、`p`、`destinations` 和 `connections`。因此这里保留
    普通 class 与公开属性，不改成带属性保护的对象，避免后续迁移环境时改变行为。
    """

    def __init__(self, start_position, start_direction, start_velocity, start_p):
        # 当前位置。旧实现通常使用三维坐标列表/数组，Stage 1 不复制或转换，保持引用语义。
        self.position = start_position

        # 当前水平运动方向；具体角度单位和更新公式属于 mobility 模块。
        self.direction = start_direction

        # 当前速度。旧 Gauss-Markov 运动会直接更新该字段。
        self.velocity = start_velocity

        # 无人机与地平面的夹角。字段名沿用旧版 `p`，避免环境迁移时出现重命名噪声。
        self.p = start_p

        # Gauss-Markov 运动模型的长期均值参数，初始化时等于初始状态。
        self.mean_velocity = start_velocity
        self.mean_direction = start_direction
        self.mean_p = start_p

        # 该簇头服务的目的节点集合。旧环境会在 reset/拓扑更新时直接替换或追加。
        self.destinations = []

        # 当前链路连接记录。具体含义由 link_budget/环境编排阶段维护。
        self.connections = []


class RP:
    """接收点实体。

    `RP` 在旧代码里只保存位置和连接记录。这里保持同样的极简形态，让 Stage 3
    拆分链路预算和拓扑逻辑时可以直接复用。
    """

    def __init__(self, start_position):
        # 接收点位置。保持旧实现的引用语义，不在实体层做类型转换。
        self.position = start_position

        # 与该接收点相关的连接记录，由环境/链路模块维护。
        self.connections = []
