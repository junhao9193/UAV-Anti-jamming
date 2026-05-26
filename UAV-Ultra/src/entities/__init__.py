"""纯领域实体导出入口。

`entities/` 是依赖图里的底层：不依赖配置、环境、算法或 PyTorch。这里仅导出旧
环境中已经存在的实体类，方便后续 `envs/` 通过一个稳定入口使用它们。
"""

from .jammer import Jammer
from .uav import RP, UAV

__all__ = ["UAV", "Jammer", "RP"]
