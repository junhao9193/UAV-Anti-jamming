"""Stage 2 import purity 测试。

``config/`` 是依赖图的叶子层（``config → 无依赖``）；``schema.py`` 与 ``specs.py`` 不得
import ``src.envs.*`` / ``src.algorithms.*`` / ``torch`` —— 这是 algorithms
层独立拿到维度规格的前提。

双重把关：
1. 静态 AST 扫描两个源文件的 import 节点；
2. 子进程运行期 ``import src.config.specs`` 后检查 ``sys.modules``。
"""

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = _REPO_ROOT / "UAV-Ultra"
_CONFIG_DIR = _PACKAGE_DIR / "src" / "config"

FORBIDDEN_PREFIXES = ("src.envs", "src.algorithms")
FORBIDDEN_EXACT = ("torch",)


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


def _assert_no_forbidden(names: list[str], where: str) -> None:
    for n in names:
        for pre in FORBIDDEN_PREFIXES:
            assert not n.startswith(pre), f"{where}: forbidden import {n!r}"
        for exact in FORBIDDEN_EXACT:
            assert n != exact and not n.startswith(exact + "."), (
                f"{where}: forbidden import {n!r}"
            )


def test_schema_py_no_forbidden_imports():
    names = _module_names_in_file(_CONFIG_DIR / "schema.py")
    _assert_no_forbidden(names, "schema.py")


def test_specs_py_no_forbidden_imports():
    names = _module_names_in_file(_CONFIG_DIR / "specs.py")
    _assert_no_forbidden(names, "specs.py")


def test_runtime_import_specs_does_not_pull_forbidden_modules():
    """在干净子进程中 import specs，确认 sys.modules 不出现禁用模块。"""
    code = (
        "import sys\n"
        "import src.config.specs  # noqa: F401\n"
        "bad = []\n"
        "if 'torch' in sys.modules:\n"
        "    bad.append('torch')\n"
        "bad += [m for m in sys.modules if m.startswith('src.envs')]\n"
        "bad += [m for m in sys.modules if m.startswith('src.algorithms')]\n"
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
        f"runtime import pulled forbidden modules.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
