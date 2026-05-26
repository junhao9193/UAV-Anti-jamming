"""VDN 子包入口。"""

from src.algorithms.common.registry import register
from src.algorithms.vdn.evaluator import VDNEvalPolicy
from src.algorithms.vdn.trainer import VDNTrainer

register("vdn", VDNTrainer, VDNEvalPolicy)

__all__ = ["VDNTrainer", "VDNEvalPolicy"]
