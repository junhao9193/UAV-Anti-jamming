"""IQL agent 入口：薄重导出 ``MPDQNAgent``（plan locked decision #4）。

baseline IQL trainer 持 N 个独立 ``MPDQNAgent``，本文件提供算法包内引用入口；
真实实现保留在 ``src.algorithms.common.agents.mpdqn_agent``，与 VDN/QMIX/QPLEX
统一契约（plan locked #6）。
"""

from __future__ import annotations

from src.algorithms.common.agents.mpdqn_agent import MPDQNAgent

__all__ = ["MPDQNAgent"]
