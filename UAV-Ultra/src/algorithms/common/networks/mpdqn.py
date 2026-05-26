"""MP-DQN 参数化动作网络（baseline `algorithms/mpdqn/model.py:5-84` 的直接迁移）。

设计约束（Stage 4 plan locked decision #9）：**子模块属性名必须保持 baseline 不变**
（``net.0``、``net.1`` 序列分支；``state_encoder.0``、``q_head.1`` 等），否则
``load_state_dict(strict=True)`` 回归测试会立即失败。

公开类：
- ``MPDQNActor(state_dim, n_actions, param_dim, hidden_dim=128)``
  forward: ``(B, state_dim) -> (B, n_actions, param_dim)``，输出经 Sigmoid 归一化到 [0, 1]。
- ``MPDQNQNetwork(state_dim, n_actions, param_dim, hidden_dim=128, q_hidden_dim=128)``
  forward: ``(state: (B, state_dim), action_params: (B, n_actions, param_dim)) -> (B, n_actions)``。
  实现多通道 (multi-pass) 评估：state 编码 + one-hot 拼接 + per-action 参数 → 每动作 Q。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MPDQNActor(nn.Module):
    """Baseline-equivalent MP-DQN actor; 子模块属性名 ``net`` 与 baseline 完全一致。"""

    def __init__(self, state_dim: int, n_actions: int, param_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)

        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_actions * self.param_dim),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.view(state.shape[0], self.n_actions, self.param_dim)


class MPDQNQNetwork(nn.Module):
    """Baseline-equivalent MP-DQN Q-network；子模块属性名 ``state_encoder`` / ``q_head``。"""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        hidden_dim: int = 128,
        q_hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )

        q_input_dim = int(hidden_dim) + self.n_actions + self.param_dim
        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, int(q_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(q_hidden_dim), 1),
        )

    def forward(self, state: torch.Tensor, action_params: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        features = self.state_encoder(state)  # (B, hidden_dim)

        features = features.unsqueeze(1).expand(-1, self.n_actions, -1)  # (B, A, hidden_dim)
        action_onehot = (
            torch.eye(self.n_actions, device=state.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        x = torch.cat([features, action_onehot, action_params], dim=2)  # (B, A, hidden + A + P)
        x = x.reshape(batch_size * self.n_actions, -1)
        q = self.q_head(x).view(batch_size, self.n_actions)
        return q


__all__ = ["MPDQNActor", "MPDQNQNetwork"]
