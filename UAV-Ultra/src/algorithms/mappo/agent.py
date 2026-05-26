"""MAPPO ``ActResult`` dataclass。

MAPPO trainer 直接持 actor + critic + optimizer，不需要重的 Agent 包装。本文件仅提供
``ActResult`` 数据类型，供 trainer.act 与 evaluator.select_actions 返回。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActResult:
    """MAPPO ``act()`` 单次结果。"""

    action_discrete: int
    action_cont: np.ndarray
    log_prob: float
    value: float


__all__ = ["ActResult"]
