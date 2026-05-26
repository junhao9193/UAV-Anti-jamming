"""``algorithms/`` 依赖纯度（REFACTOR.md Constraint 2 + plan locked decision #10）。

AST 扫描 ``src/algorithms/**/*.py``：禁止 import
``src.envs`` / ``src.training`` / ``src.evaluation``。
允许 ``src.config`` / ``src.entities`` / ``src.algorithms`` 自身。
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALGO_DIR = _REPO_ROOT / "UAV-Ultra" / "src" / "algorithms"

FORBIDDEN_PREFIXES = (
    "src.envs",
    "src.training",
    "src.evaluation",
)


def _module_names_in_file(py_path: Path) -> list[str]:
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_algorithms_does_not_import_envs_training_evaluation():
    py_files = sorted(_ALGO_DIR.rglob("*.py"))
    assert py_files, f"no python files under {_ALGO_DIR}"
    for py in py_files:
        names = _module_names_in_file(py)
        for n in names:
            for pre in FORBIDDEN_PREFIXES:
                assert not n.startswith(pre), (
                    f"{py.relative_to(_REPO_ROOT)}: forbidden import {n!r}"
                )
