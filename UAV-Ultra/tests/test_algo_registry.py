"""算法注册表契约测试（Stage 4 plan §「API contracts」）。

4A 仅验证 registry 自身：
- ``ALGO_NAMES`` 完整 6 名；
- ``register/get_*/build_*`` 签名与 heuristic 特例（trainer=None）；
- 未知名 raise ValueError；
- 5 个学习算法 ``build_trainer(algo_cfg=None)`` 拒绝。

注：4B-4F 各算法子包尚未实现时，``registered_names()`` 为空 —— 那是预期。
"""

from __future__ import annotations

import pytest

import src.algorithms.common.registry as registry_mod
from src.algorithms import ALGO_NAMES
from src.algorithms.common import (
    build_evaluator,
    build_trainer,
    get_evaluator_cls,
    get_trainer_cls,
    register,
    registered_names,
)


@pytest.fixture(autouse=True)
def _restore_registry_after_test():
    """Registry tests mutate global state; restore it so later Stage 4 tests stay isolated."""
    snapshot = dict(registry_mod._REGISTRY)
    yield
    registry_mod._REGISTRY.clear()
    registry_mod._REGISTRY.update(snapshot)


def test_algo_names_canonical_set():
    """ALGO_NAMES 必须是 6 个固定名字。"""
    assert set(ALGO_NAMES) == {"iql", "qmix", "vdn", "qplex", "mappo", "heuristic"}
    assert len(ALGO_NAMES) == 6


def test_register_rejects_invalid_inputs():
    class _T: ...

    with pytest.raises(ValueError, match="non-empty str"):
        register("", None, object)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trainer_cls"):
        register("__bad_trainer__", "not_a_class", object)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evaluator_cls"):
        register("__bad_eval__", _T, "not_a_class")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only allowed"):
        register("__bad_none_trainer__", None, object)


def test_register_duplicate_name_rejected():
    class _T: ...
    class _E: ...
    register("__dup_test__", _T, _E)
    with pytest.raises(ValueError, match="already registered"):
        register("__dup_test__", _T, _E)


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        get_trainer_cls("nope_not_real")
    with pytest.raises(ValueError, match="unknown algorithm"):
        get_evaluator_cls("nope_not_real")


def test_build_trainer_heuristic_returns_none_without_algo_cfg():
    """heuristic 是约定的「无 trainer」入口，``build_trainer`` 始终返回 None。"""
    result = build_trainer("heuristic", env_cfg=None, algo_cfg=None)
    assert result is None


def test_build_trainer_learning_algo_requires_algo_cfg():
    """5 个学习算法的 build_trainer 在 algo_cfg=None 时必须报错。

    即便子包未实现（不会走到 cls 构造），也应该先于该错误抛出 algo_cfg=None 错误，
    或者抛 unknown algorithm 错误（取决于子包是否注册）—— 两者都是合理拒绝。
    """
    for name in ("iql", "qmix", "vdn", "qplex", "mappo"):
        with pytest.raises(ValueError):
            build_trainer(name, env_cfg=None, algo_cfg=None)


def test_registered_names_returns_sorted_tuple():
    names = registered_names()
    assert isinstance(names, tuple)
    assert list(names) == sorted(names)
