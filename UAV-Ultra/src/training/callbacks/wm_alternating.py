"""Deprecated alias module for ``wm_concurrent``.

Stage 7 重命名 ``wm_alternating`` → ``wm_concurrent``。本文件作为 shim 保留，
旧的 ``from src.training.callbacks.wm_alternating import WMAlternatingCallback``
仍可工作但触发 ``FutureWarning``。
"""

from __future__ import annotations

import warnings

from src.training.callbacks.wm_concurrent import (
    WMConcurrentCallback as WMAlternatingCallback,
    _diff_cfg,
    _vc_eta,
)

warnings.warn(
    "src.training.callbacks.wm_alternating is deprecated; "
    "use src.training.callbacks.wm_concurrent instead",
    FutureWarning,
    stacklevel=2,
)


__all__ = ["WMAlternatingCallback", "_diff_cfg", "_vc_eta"]
