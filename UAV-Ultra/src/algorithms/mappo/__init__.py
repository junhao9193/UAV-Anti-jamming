"""MAPPO 子包入口。"""

from src.algorithms.common.registry import register
from src.algorithms.mappo.agent import ActResult
from src.algorithms.mappo.evaluator import MAPPOEvalPolicy
from src.algorithms.mappo.trainer import MAPPOTrainer

register("mappo", MAPPOTrainer, MAPPOEvalPolicy)

__all__ = ["MAPPOTrainer", "MAPPOEvalPolicy", "ActResult"]
