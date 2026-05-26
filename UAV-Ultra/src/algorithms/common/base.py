"""算法层抽象基类（Stage 4 Plan API contracts）。

依赖图位置：``algorithms → config + entities``。本模块只定义协议，运行时不依赖
``torch`` 之外的算法实现。

三个核心 Protocol：
- ``Agent``：单 agent 视图（用于 IQL/VDN/QMIX/QPLEX 中 `trainer.agents[i]` 字段）。
- ``Trainer``：训练步入口；``train_step`` 返回 dict（损失值等指标）。
- ``EvalPolicy``：评估期动作选择；``select_actions(states) -> actions``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class TrainerStepResult:
    """``Trainer.train_step()`` 的标准返回类型。

    具体算法可携带额外字段（如 mixer loss），dict 形式保留向后兼容。
    """

    losses: Dict[str, float]
    extra: Dict[str, Any] | None = None


@runtime_checkable
class Agent(Protocol):
    """单 agent 视图。`MPDQNAgent` 实现该协议。"""

    actor: Any
    q_net: Any
    target_actor: Any
    target_q_net: Any
    actor_opt: Any
    q_opt: Any


@runtime_checkable
class Trainer(Protocol):
    """训练步抽象。具体算法在 ``train_step()`` 内组织一次更新。"""

    def train_step(self) -> Optional[Dict[str, float]]: ...


@runtime_checkable
class EvalPolicy(Protocol):
    """评估期策略：仅负责动作选择，不创建 env / 不算指标 / 不写日志。"""

    def select_actions(self, states: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]: ...


__all__ = ["Agent", "Trainer", "EvalPolicy", "TrainerStepResult"]
