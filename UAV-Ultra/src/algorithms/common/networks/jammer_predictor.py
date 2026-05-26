"""Stage 8 jammer-prediction networks.

对齐 baseline ``algorithms/mpdqn/qmix/trainer_jammer_prediction.py:18-130``：

- ``JammerPredictionHead``：3-layer MLP，``self.net`` 子模块名与 baseline 同。
- ``JammerAwareMPDQNActor``：与 ``MPDQNActor`` 结构相同，但第一层 ``Linear`` 的
  ``in_features = state_dim + n_channel``；forward 接 augmented state。
- ``JammerAwareMPDQNQNetwork``：与 ``MPDQNQNetwork`` 结构相同，但
  ``state_encoder`` 第一层 ``in_features = state_dim + n_channel``。

**state_dict 字段名严格匹配 baseline** —— 否则 ``load_state_dict(strict=True)`` 回归会立刻失败。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class JammerPredictionHead(nn.Module):
    """Predict next-slot jammed-channel probabilities from recent sensing history.

    Baseline 对齐：``algorithms/mpdqn/qmix/trainer_jammer_prediction.py:18-42``。
    BCE-with-logits 由调用方算；本类输出 raw logits。
    """

    def __init__(self, *, history_len: int, n_channel: int, hidden_dim: int = 64):
        super().__init__()
        self.history_len = int(history_len)
        self.n_channel = int(n_channel)
        in_dim = self.history_len * self.n_channel
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_channel),
        )

    def forward(self, sensing_history: torch.Tensor) -> torch.Tensor:
        if sensing_history.ndim != 3:
            raise ValueError(f"sensing_history must be (B,W,K), got {tuple(sensing_history.shape)}")
        if (
            int(sensing_history.shape[1]) != self.history_len
            or int(sensing_history.shape[2]) != self.n_channel
        ):
            raise ValueError(
                "sensing_history shape mismatch: expected "
                f"(*,{self.history_len},{self.n_channel}), got {tuple(sensing_history.shape)}"
            )
        return self.net(sensing_history.reshape(sensing_history.shape[0], -1))


class JammerAwareMPDQNActor(nn.Module):
    """MP-DQN actor that consumes ``state ++ jammer_feature`` (augmented input).

    Baseline 对齐：``trainer_jammer_prediction.py:45-74``。子模块属性名 ``self.net``。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        n_channel: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.n_channel = int(n_channel)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.n_channel, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_actions * self.param_dim),
            nn.Sigmoid(),
        )

    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        out = self.net(augmented_state)
        return out.view(augmented_state.shape[0], self.n_actions, self.param_dim)


class JammerAwareMPDQNQNetwork(nn.Module):
    """MP-DQN Q-network that consumes ``state ++ jammer_feature`` (augmented input).

    Baseline 对齐：``trainer_jammer_prediction.py:77-128``。子模块属性名
    ``self.state_encoder`` + ``self.q_head``，与 plain ``MPDQNQNetwork`` 命名一致。
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        n_channel: int,
        hidden_dim: int = 128,
        q_hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.n_channel = int(n_channel)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.n_channel, int(hidden_dim)),
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

    def forward(self, augmented_state: torch.Tensor, action_params: torch.Tensor) -> torch.Tensor:
        batch_size = augmented_state.shape[0]
        features = self.state_encoder(augmented_state)
        features = features.unsqueeze(1).expand(-1, self.n_actions, -1)
        action_onehot = (
            torch.eye(self.n_actions, device=augmented_state.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        x = torch.cat([features, action_onehot, action_params], dim=2)
        x = x.reshape(batch_size * self.n_actions, -1)
        q = self.q_head(x).view(batch_size, self.n_actions)
        return q


__all__ = [
    "JammerPredictionHead",
    "JammerAwareMPDQNActor",
    "JammerAwareMPDQNQNetwork",
]
