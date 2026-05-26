"""Heuristic 子包入口：trainer=None（plan locked decision #2）。"""

from src.algorithms.common.registry import register
from src.algorithms.heuristic.evaluator import HeuristicEvalPolicy
from src.algorithms.heuristic.policies import (
    GreedySensingPolicy,
    HeuristicDims,
    MaxCSIPolicy,
    MinInterferencePolicy,
    RandomHoppingPolicy,
    ScoreBasedPolicy,
    build_heuristic_policy,
)

register("heuristic", None, HeuristicEvalPolicy)

__all__ = [
    "HeuristicEvalPolicy",
    "HeuristicDims",
    "RandomHoppingPolicy",
    "ScoreBasedPolicy",
    "GreedySensingPolicy",
    "MaxCSIPolicy",
    "MinInterferencePolicy",
    "build_heuristic_policy",
]
