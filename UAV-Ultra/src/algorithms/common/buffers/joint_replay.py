"""联合 replay buffer：IQL（per-agent reward）与 QMIX/VDN/QPLEX（global scalar reward）
两种模式由 ``per_agent_reward: bool`` 参数化区分。

合并自 baseline `algorithms/mpdqn/iql/joint_replay_buffer.py` 与
`algorithms/mpdqn/qmix/joint_replay_buffer.py`（数据布局相同，只有 reward 维度不同）。

形状契约（Stage 4 Plan Trainer Batch Contract 表 A）：
- state:           (B, N, S)
- action_discrete: (B, N)        int64
- action_params:   (B, N, A*P)
- reward:          (B, N) if per_agent_reward else (B,)
- next_state:      (B, N, S)
- done:            (B,)          float32

Stage 8 扩展：``track_jammer=True`` 时额外存 ``sensing_history / next_sensing_history /
jammer_target`` 三个 JP 字段（baseline `joint_replay_buffer.py:297-331`）：
- sensing_history:      (B, N, H, C)
- next_sensing_history: (B, N, H, C)
- jammer_target:        (B, C)   ★ 单一团队 target，aux loss 内广播到每 agent
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple, Union

import numpy as np


class JointReplayBuffer:
    """联合多 agent transition replay。

    Args:
        capacity: 最大容量。
        per_agent_reward:
            - ``True``：每条 transition 的 reward 是 ``(N,)`` 向量（IQL 用）。
            - ``False``：每条 transition 的 reward 是 ``float`` 标量（VDN/QMIX/QPLEX 用，
              团队均值奖励）。
        track_jammer: Stage 8 — 是否额外存 sensing_history / next_sensing_history /
            jammer_target 三个 JP 字段。True 时 ``add`` 必传三参数；False 时三参数必为 None。
    """

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        per_agent_reward: bool,
        track_jammer: bool = False,
    ):
        self.capacity = int(capacity)
        self.per_agent_reward = bool(per_agent_reward)
        self.track_jammer = bool(track_jammer)
        self._buffer: Deque[Tuple] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: bool,
        *,
        sensing_history: Optional[np.ndarray] = None,
        next_sensing_history: Optional[np.ndarray] = None,
        jammer_target: Optional[np.ndarray] = None,
    ) -> None:
        if self.per_agent_reward:
            reward_stored = np.asarray(reward, dtype=np.float32)
            if reward_stored.ndim != 1:
                raise ValueError(
                    f"per_agent_reward=True requires (N,) reward vector, got shape {reward_stored.shape}"
                )
        else:
            reward_stored = float(reward)

        # Stage 8：JP 字段双向 strict
        if self.track_jammer:
            missing = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"track_jammer=True requires JP fields, missing: {missing}"
                )
            jp_tuple = (
                np.asarray(sensing_history, dtype=np.float32),
                np.asarray(next_sensing_history, dtype=np.float32),
                np.asarray(jammer_target, dtype=np.float32),
            )
        else:
            present = [
                name
                for name, val in (
                    ("sensing_history", sensing_history),
                    ("next_sensing_history", next_sensing_history),
                    ("jammer_target", jammer_target),
                )
                if val is not None
            ]
            if present:
                raise ValueError(
                    f"track_jammer=False does not accept JP fields, got: {present}"
                )
            jp_tuple = ()

        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action_discrete, dtype=np.int64),
                np.asarray(action_params, dtype=np.float32),
                reward_stored,
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                *jp_tuple,
            )
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        idx = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        rows = [self._buffer[i] for i in idx]

        if self.track_jammer:
            unzipped = list(zip(*rows))
            (
                states,
                actions,
                params,
                rewards,
                next_states,
                dones,
                sensing_histories,
                next_sensing_histories,
                jammer_targets,
            ) = unzipped
        else:
            states, actions, params, rewards, next_states, dones = zip(*rows)

        out: Dict[str, np.ndarray] = {
            "state": np.asarray(states, dtype=np.float32),
            "action_discrete": np.asarray(actions, dtype=np.int64),
            "action_params": np.asarray(params, dtype=np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),  # (B, N) 或 (B,)
            "next_state": np.asarray(next_states, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
        }
        if self.track_jammer:
            out["sensing_history"] = np.asarray(sensing_histories, dtype=np.float32)
            out["next_sensing_history"] = np.asarray(next_sensing_histories, dtype=np.float32)
            out["jammer_target"] = np.asarray(jammer_targets, dtype=np.float32)
        return out


__all__ = ["JointReplayBuffer"]
