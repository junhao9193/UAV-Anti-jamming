"""算法注册表（Stage 4 Plan API contracts §「Unified factories」）。

合约（locked decisions #1 + #2）：
- ``register(name, trainer_cls, evaluator_cls)``：trainer_cls 可为 ``None``（heuristic）。
- ``get_trainer_cls(name) -> type | None``；``get_evaluator_cls(name) -> type``。
- ``build_trainer(name, *, env_cfg, algo_cfg, ...) -> Trainer | None``：
  heuristic 返回 ``None``；其它算法 ``algo_cfg`` 必须提供。
- ``build_evaluator(name, *, env_cfg, algo_cfg=None, trainer=None, ...) -> EvalPolicy``：
  5 个学习算法需提供 trainer；heuristic 只需 env_cfg。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

# 注册表本体。子包 import 时调用 `register(...)` 填充。
_REGISTRY: Dict[str, Tuple[Optional[type], type]] = {}


def register(name: str, trainer_cls: Optional[type], evaluator_cls: type) -> None:
    """登记算法。``trainer_cls=None`` 仅允许 heuristic 走该路径。"""
    if not isinstance(name, str) or not name:
        raise ValueError(f"register: name must be non-empty str, got {name!r}")
    if name in _REGISTRY:
        raise ValueError(f"register: '{name}' already registered")
    if trainer_cls is None and name != "heuristic":
        raise ValueError("register: trainer_cls=None is only allowed for 'heuristic'")
    if trainer_cls is not None and not isinstance(trainer_cls, type):
        raise TypeError(f"register: trainer_cls must be a class or None, got {trainer_cls!r}")
    if not isinstance(evaluator_cls, type):
        raise TypeError(f"register: evaluator_cls must be a class, got {evaluator_cls!r}")
    _REGISTRY[name] = (trainer_cls, evaluator_cls)


def registered_names() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def get_trainer_cls(name: str) -> Optional[type]:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown algorithm {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name][0]


def get_evaluator_cls(name: str) -> type:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown algorithm {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name][1]


def build_trainer(
    name: str,
    *,
    env_cfg: Any,
    algo_cfg: Any = None,
    device: str | None = None,
    **kwargs: Any,
) -> Any:
    """构造 trainer 实例。

    - ``name == "heuristic"``：始终返回 ``None``（无可学参；契约明示）。
    - 其它 name：``algo_cfg`` 不可为 None。
    """
    if name == "heuristic":
        return None
    cls = get_trainer_cls(name)
    if cls is None:
        raise ValueError(
            f"build_trainer: '{name}' has no trainer registered (use heuristic for None-trainer path)"
        )
    if algo_cfg is None:
        raise ValueError(
            f"build_trainer: algo_cfg is required for '{name}' (only 'heuristic' allows None)"
        )
    return cls(env_cfg=env_cfg, algo_cfg=algo_cfg, device=device, **kwargs)


def build_evaluator(
    name: str,
    *,
    env_cfg: Any,
    algo_cfg: Any = None,
    trainer: Any = None,
    **kwargs: Any,
) -> Any:
    """构造 evaluator 实例。

    - 5 个学习算法需提供 trainer；
    - heuristic 仅需 env_cfg（trainer/algo_cfg 可省略）。
    """
    cls = get_evaluator_cls(name)
    if name == "heuristic":
        return cls(env_cfg=env_cfg, **kwargs)
    if trainer is None:
        raise ValueError(
            f"build_evaluator: '{name}' requires trainer (only 'heuristic' allows None)"
        )
    return cls(env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer, **kwargs)


__all__ = [
    "register",
    "registered_names",
    "get_trainer_cls",
    "get_evaluator_cls",
    "build_trainer",
    "build_evaluator",
]
