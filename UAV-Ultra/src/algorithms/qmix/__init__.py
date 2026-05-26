"""QMIX 子包入口。"""

from src.algorithms.common.registry import register
from src.algorithms.qmix.evaluator import QMIXEvalPolicy
from src.algorithms.qmix.trainer import QMIXTrainer

register("qmix", QMIXTrainer, QMIXEvalPolicy)

__all__ = ["QMIXTrainer", "QMIXEvalPolicy"]
