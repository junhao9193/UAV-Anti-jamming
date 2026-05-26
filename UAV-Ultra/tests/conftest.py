"""pytest 共享 fixtures。

提供 ``baseline_import`` fixture：临时把 baseline worktree 加入 ``sys.path``，
import 指定模块；测试结束自动清理 ``sys.path`` 与 ``sys.modules`` 中受污染的顶层 key。

实现策略遵循 Stage 4 plan 修订 #4：
- 用 ``importlib.import_module`` 而非 ``spec_from_file_location``，支持 baseline 内部的
  ``from algorithms... / from envs...`` 绝对 import。
- teardown 清理 sys.modules 中**精确顶层 key** ``{"algorithms", "envs", "Main"}`` 与
  **前缀匹配** ``("algorithms.", "envs.", "Main.")`` 的所有子模块，避免 baseline 模块
  污染 ``src.*`` 命名空间。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


_BASELINE_ROOT = Path("/home/jh/projects/UAV-baseline/UAV-Jammer-RL")
_BASELINE_TOP_KEYS = {"algorithms", "envs", "Main"}
_BASELINE_PREFIXES = ("algorithms.", "envs.", "Main.")


@pytest.fixture()
def baseline_import():
    """返回 ``import_fn(module_path) -> ModuleType``，使用后自动清理。"""
    if not _BASELINE_ROOT.exists():
        pytest.skip(f"baseline worktree missing: {_BASELINE_ROOT}")

    sys_path_added = False
    baseline_root_str = str(_BASELINE_ROOT)
    snapshot_keys = set(sys.modules.keys())

    if baseline_root_str not in sys.path:
        sys.path.insert(0, baseline_root_str)
        sys_path_added = True

    def _import_fn(module_path: str) -> ModuleType:
        return importlib.import_module(module_path)

    try:
        yield _import_fn
    finally:
        # 清理 sys.modules：删除测试期间新加入的、归属 baseline 顶层包的 key
        new_keys = set(sys.modules.keys()) - snapshot_keys
        for key in list(new_keys):
            if key in _BASELINE_TOP_KEYS or key.startswith(_BASELINE_PREFIXES):
                sys.modules.pop(key, None)
        # 清理 sys.path
        if sys_path_added and baseline_root_str in sys.path:
            sys.path.remove(baseline_root_str)
