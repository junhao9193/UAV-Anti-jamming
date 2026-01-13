from __future__ import division

import torch
import torch.nn as nn


class MPDQNActor(nn.Module):
    """
    Actor network for MP-DQN:
    Given state s, outputs continuous parameters for *all* discrete actions.

    Output is normalized to [0, 1] via Sigmoid. Environment scales to physical ranges.
    """

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
    """
    Q network for MP-DQN:
    Q(s, a, x_a) where a is discrete action and x_a is its continuous parameter vector.

    For multi-pass evaluation, `forward` returns Q-values for *all* discrete actions,
    using the corresponding parameters produced for each action.
    """

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
        """
        Args:
            state: (batch, state_dim)
            action_params: (batch, n_actions, param_dim) continuous params in [0,1]

        Returns:
            q_values: (batch, n_actions)
        """
        batch_size = state.shape[0]
        features = self.state_encoder(state)  # (batch, hidden_dim)

        features = features.unsqueeze(1).expand(-1, self.n_actions, -1)  # (batch, n_actions, hidden_dim)
        action_onehot = torch.eye(self.n_actions, device=state.device).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([features, action_onehot, action_params], dim=2)  # (batch, n_actions, hidden+n_actions+param)
        x = x.reshape(batch_size * self.n_actions, -1)
        q = self.q_head(x).view(batch_size, self.n_actions)
        return q


__all__ = ["MPDQNActor", "MPDQNQNetwork"]
