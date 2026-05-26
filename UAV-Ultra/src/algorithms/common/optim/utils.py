"""共享优化工具：grad clip / 硬同步 target / AMP context。

抽 baseline `mpdqn/agent.py` 与 `mpdqn/qmix/trainer_greedy_actor.py` 多处重复段。
"""

from __future__ import annotations

import contextlib
from typing import Iterable

import torch
from torch import amp as torch_amp


def hard_sync_target(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """硬同步：``target.load_state_dict(source.state_dict())``。"""
    target.load_state_dict(source.state_dict())


def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """按 ``max_norm > 0`` 调用 torch grad clip；否则空操作返回 0。"""
    params = list(parameters)
    if max_norm <= 0.0 or not params:
        return torch.zeros((), dtype=torch.float32)
    return torch.nn.utils.clip_grad_norm_(params, float(max_norm))


@contextlib.contextmanager
def amp_autocast(enabled: bool):
    """统一封装 ``torch.amp.autocast("cuda", enabled=...)``。CPU 路径直接 yield。"""
    if enabled:
        with torch_amp.autocast("cuda", enabled=True):
            yield
    else:
        yield


__all__ = ["hard_sync_target", "clip_grad_norm", "amp_autocast"]
