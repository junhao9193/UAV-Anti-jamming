"""算法共享层：base / registry / networks / agents / buffers / optim / value_decomp。

依赖图位置：``algorithms/common → config + entities``。本包**不**得 import
``src.envs`` / ``src.training`` / ``src.evaluation``。
"""

from src.algorithms.common.base import (
    Agent,
    EvalPolicy,
    Trainer,
    TrainerStepResult,
)
from src.algorithms.common.registry import (
    build_evaluator,
    build_trainer,
    get_evaluator_cls,
    get_trainer_cls,
    register,
    registered_names,
)

__all__ = [
    "Agent",
    "EvalPolicy",
    "Trainer",
    "TrainerStepResult",
    "register",
    "registered_names",
    "get_trainer_cls",
    "get_evaluator_cls",
    "build_trainer",
    "build_evaluator",
]
