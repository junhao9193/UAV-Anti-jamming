"""IQL 子包入口：注册 (IQLTrainer, IQLEvalPolicy)。"""

from src.algorithms.common.registry import register
from src.algorithms.iql.agent import MPDQNAgent  # re-export
from src.algorithms.iql.evaluator import IQLEvalPolicy
from src.algorithms.iql.trainer import IQLTrainer

register("iql", IQLTrainer, IQLEvalPolicy)

__all__ = ["IQLTrainer", "IQLEvalPolicy", "MPDQNAgent"]
