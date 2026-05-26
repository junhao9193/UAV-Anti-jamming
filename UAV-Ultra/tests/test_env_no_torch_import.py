"""``src.envs.*`` 的 import 纯度测试。

REFACTOR.md Constraint 2 铁律一：``envs/`` **不得 import torch**（环境是纯物理仿真）。

双重把关：
1. AST 扫描每个 envs/*.py 的 import 节点，禁止 ``torch.*``。
2. 子进程加载完整 ``src.envs`` 包后，``sys.modules`` 不得出现 ``torch``。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = _REPO_ROOT / "UAV-Ultra"
_ENVS_DIR = _PACKAGE_DIR / "src" / "envs"

FORBIDDEN_EXACT = ("torch",)
FORBIDDEN_PREFIXES = ("src.algorithms", "src.training")


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


def _envs_python_files() -> list[Path]:
    return sorted(p for p in _ENVS_DIR.iterdir() if p.is_file() and p.suffix == ".py")


def test_envs_modules_do_not_import_torch_statically():
    for py in _envs_python_files():
        names = _module_names_in_file(py)
        for n in names:
            for exact in FORBIDDEN_EXACT:
                assert n != exact and not n.startswith(exact + "."), (
                    f"{py.name}: forbidden static import {n!r}"
                )
            for pre in FORBIDDEN_PREFIXES:
                assert not n.startswith(pre), (
                    f"{py.name}: forbidden static import {n!r}"
                )


def test_runtime_import_envs_does_not_pull_torch():
    """子进程加载完整 envs 包后，sys.modules 不得出现 torch。"""
    code = (
        "import sys\n"
        "import src.envs  # noqa: F401\n"
        "import src.envs.channel  # noqa: F401\n"
        "import src.envs.jammer_model  # noqa: F401\n"
        "import src.envs.mobility  # noqa: F401\n"
        "import src.envs.link_budget  # noqa: F401\n"
        "import src.envs.sensing  # noqa: F401\n"
        "import src.envs.observation  # noqa: F401\n"
        "import src.envs.action_space  # noqa: F401\n"
        "import src.envs.reward  # noqa: F401\n"
        "import src.envs.environment  # noqa: F401\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "if bad:\n"
        "    print('FORBIDDEN', bad)\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_PACKAGE_DIR),
    )
    assert result.returncode == 0, (
        f"runtime import pulled torch.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
