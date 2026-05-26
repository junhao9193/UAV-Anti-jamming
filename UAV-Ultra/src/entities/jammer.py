"""干扰机实体。

本模块只保存干扰机的物理状态。干扰策略、Markov 转移和 reactive bias 属于
后续 `envs/jammer_model.py`，不应放入实体类，避免实体层依赖环境逻辑。
"""


class Jammer:
    """干扰机实体。

    构造参数和公开属性保持与旧 `envs.entities.Jammer` 一致。旧版参数名中第三个
    参数叫 `velocity`，这里也保留该名字，方便对照旧代码与 golden-master trace。
    """

    def __init__(self, start_position, start_direction, velocity, start_p):
        # 当前位置。旧实现通常传入三维坐标列表/数组；这里不复制，保持旧引用语义。
        self.position = start_position

        # 当前运动方向；具体更新由 mobility 或 jammer_model 阶段处理。
        self.direction = start_direction

        # 当前速度。字段名和旧实现一致。
        self.velocity = velocity

        # 干扰机与地平面的夹角。沿用旧字段名 `p`。
        self.p = start_p

        # Gauss-Markov 运动模型的长期均值参数，初始化时等于初始状态。
        self.mean_velocity = velocity
        self.mean_direction = start_direction
        self.mean_p = start_p
