"""算法层入口（Stage 4）。

公开：
- ``ALGO_NAMES``：6 个注册名（iql / qmix / vdn / qplex / mappo / heuristic）。
- ``build_trainer(name, *, env_cfg, algo_cfg=None, ...)``：heuristic 返回 None。
- ``build_evaluator(name, *, env_cfg, algo_cfg=None, trainer=None, ...)``。

子包 import 时通过 ``common.registry.register(...)`` 自动登记。
"""

import importlib
import importlib.util

from src.algorithms.common import build_evaluator, build_trainer, registered_names

ALGO_NAMES: tuple[str, ...] = ("iql", "qmix", "vdn", "qplex", "mappo", "heuristic")


def _ensure_subpackages_imported() -> None:
    """惰性 import 各算法子包以触发 register() 调用。

    若某个子包尚未实现（Stage 4 子阶段分批落地），跳过；但若子包存在
    且其内部 import 出错，错误必须原样暴露，避免把真实实现错误伪装成
    "unknown algorithm"。
    """
    for name in ALGO_NAMES:
        module_name = f"src.algorithms.{name}"
        if importlib.util.find_spec(module_name) is None:
            # 当前子阶段未实现该算法子包，是允许的；后续阶段会补上。
            continue
        importlib.import_module(module_name)


_ensure_subpackages_imported()


__all__ = ["ALGO_NAMES", "build_trainer", "build_evaluator", "registered_names"]
