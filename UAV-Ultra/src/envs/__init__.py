"""环境层公开 API：``Environ`` 编排器与 8 个子模块。

依赖图位置（REFACTOR.md Constraint 2）：``envs → entities + config``。
本包**不**得 import ``torch`` / ``src.algorithms`` / ``src.training``。
"""

from src.envs.environment import Environ

__all__ = ["Environ"]
