"""QPLEX 子包入口。"""

from src.algorithms.common.registry import register
from src.algorithms.qplex.evaluator import QPLEXEvalPolicy
from src.algorithms.qplex.trainer import QPLEXTrainer

register("qplex", QPLEXTrainer, QPLEXEvalPolicy)

__all__ = ["QPLEXTrainer", "QPLEXEvalPolicy"]
