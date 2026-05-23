import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.register_buffer("action_eye", torch.eye(self.n_actions, dtype=torch.float32), persistent=False)

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
        action_onehot = self.action_eye.to(dtype=state.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([features, action_onehot, action_params], dim=2)  # (batch, n_actions, hidden+n_actions+param)
        x = x.reshape(batch_size * self.n_actions, -1)
        q = self.q_head(x).view(batch_size, self.n_actions)
        return q


class BatchedIndependentLinear(nn.Module):
    """N independent Linear layers evaluated as one batched operation.

    Parameters keep the agent axis first: weight=(N,out,in), bias=(N,out).
    Adam remains element-wise over these tensors, so using one optimizer preserves
    independent per-agent Adam state while avoiding N separate tiny module calls.
    """

    def __init__(self, n_agents: int, in_features: int, out_features: int):
        super().__init__()
        self.n_agents = int(n_agents)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.n_agents, self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.n_agents, self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.n_agents):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.n_agents or x.shape[-1] != self.in_features:
            raise ValueError(
                f"x must end with (N={self.n_agents}, in={self.in_features}), got {tuple(x.shape)}"
            )
        return torch.einsum("...ni,noi->...no", x, self.weight) + self.bias

    def forward_agent(self, agent_idx: int, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight[int(agent_idx)], self.bias[int(agent_idx)])

    def copy_agent_from_linear(self, agent_idx: int, linear: nn.Linear) -> None:
        with torch.no_grad():
            self.weight[int(agent_idx)].copy_(linear.weight)
            self.bias[int(agent_idx)].copy_(linear.bias)


class BatchedIndependentMPDQNActor(nn.Module):
    """Independent MP-DQN actors evaluated together over the agent axis."""

    def __init__(self, n_agents: int, state_dim: int, n_actions: int, param_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = BatchedIndependentLinear(self.n_agents, self.state_dim, self.hidden_dim)
        self.fc2 = BatchedIndependentLinear(self.n_agents, self.hidden_dim, self.hidden_dim)
        self.fc3 = BatchedIndependentLinear(self.n_agents, self.hidden_dim, self.n_actions * self.param_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != self.n_agents or state.shape[2] != self.state_dim:
            raise ValueError(f"state must be (B,{self.n_agents},{self.state_dim}), got {tuple(state.shape)}")
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))
        return out.view(state.shape[0], self.n_agents, self.n_actions, self.param_dim)

    def forward_agent(self, agent_idx: int, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1.forward_agent(agent_idx, state))
        x = F.relu(self.fc2.forward_agent(agent_idx, x))
        out = torch.sigmoid(self.fc3.forward_agent(agent_idx, x))
        return out.view(state.shape[0], self.n_actions, self.param_dim)

    def agent_state_dict(self, agent_idx: int) -> dict[str, torch.Tensor]:
        i = int(agent_idx)
        return {
            "net.0.weight": self.fc1.weight[i].detach().clone(),
            "net.0.bias": self.fc1.bias[i].detach().clone(),
            "net.2.weight": self.fc2.weight[i].detach().clone(),
            "net.2.bias": self.fc2.bias[i].detach().clone(),
            "net.4.weight": self.fc3.weight[i].detach().clone(),
            "net.4.bias": self.fc3.bias[i].detach().clone(),
        }

    def load_agent_state_dict(self, agent_idx: int, state_dict: dict[str, torch.Tensor]) -> None:
        i = int(agent_idx)
        with torch.no_grad():
            self.fc1.weight[i].copy_(state_dict["net.0.weight"])
            self.fc1.bias[i].copy_(state_dict["net.0.bias"])
            self.fc2.weight[i].copy_(state_dict["net.2.weight"])
            self.fc2.bias[i].copy_(state_dict["net.2.bias"])
            self.fc3.weight[i].copy_(state_dict["net.4.weight"])
            self.fc3.bias[i].copy_(state_dict["net.4.bias"])


class BatchedIndependentMPDQNQNetwork(nn.Module):
    """Independent MP-DQN Q networks evaluated together over the agent axis."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        param_dim: int,
        hidden_dim: int = 128,
        q_hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.hidden_dim = int(hidden_dim)
        self.q_hidden_dim = int(q_hidden_dim)

        self.state_fc1 = BatchedIndependentLinear(self.n_agents, self.state_dim, self.hidden_dim)
        self.state_fc2 = BatchedIndependentLinear(self.n_agents, self.hidden_dim, self.hidden_dim)
        q_input_dim = self.hidden_dim + self.n_actions + self.param_dim
        self.q_fc1 = BatchedIndependentLinear(self.n_agents, q_input_dim, self.q_hidden_dim)
        self.q_fc2 = BatchedIndependentLinear(self.n_agents, self.q_hidden_dim, 1)
        self.register_buffer("action_eye", torch.eye(self.n_actions, dtype=torch.float32), persistent=False)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.state_fc1(state))
        return F.relu(self.state_fc2(x))

    def forward(self, state: torch.Tensor, action_params: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1] != self.n_agents or state.shape[2] != self.state_dim:
            raise ValueError(f"state must be (B,{self.n_agents},{self.state_dim}), got {tuple(state.shape)}")
        if (
            action_params.ndim != 4
            or action_params.shape[1] != self.n_agents
            or action_params.shape[2] != self.n_actions
            or action_params.shape[3] != self.param_dim
        ):
            raise ValueError(
                "action_params must be "
                f"(B,{self.n_agents},{self.n_actions},{self.param_dim}), got {tuple(action_params.shape)}"
            )

        batch_size = state.shape[0]
        features = self._encode_state(state)  # (B,N,H)
        features = features.unsqueeze(2).expand(-1, -1, self.n_actions, -1)  # (B,N,A,H)
        action_onehot = self.action_eye.to(dtype=state.dtype).view(1, 1, self.n_actions, self.n_actions)
        action_onehot = action_onehot.expand(batch_size, self.n_agents, -1, -1)
        x = torch.cat([features, action_onehot, action_params], dim=3)  # (B,N,A,H+A+P)

        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.n_actions, self.n_agents, -1)
        x = F.relu(self.q_fc1(x))
        q = self.q_fc2(x).view(batch_size, self.n_actions, self.n_agents)
        return q.permute(0, 2, 1).contiguous()  # (B,N,A)

    def forward_agent(self, agent_idx: int, state: torch.Tensor, action_params: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        features = F.relu(self.state_fc1.forward_agent(agent_idx, state))
        features = F.relu(self.state_fc2.forward_agent(agent_idx, features))
        features = features.unsqueeze(1).expand(-1, self.n_actions, -1)
        action_onehot = self.action_eye.to(device=state.device, dtype=state.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([features, action_onehot, action_params], dim=2)
        x = x.reshape(batch_size * self.n_actions, -1)
        x = F.relu(self.q_fc1.forward_agent(agent_idx, x))
        q = self.q_fc2.forward_agent(agent_idx, x).view(batch_size, self.n_actions)
        return q

    def agent_state_dict(self, agent_idx: int) -> dict[str, torch.Tensor]:
        i = int(agent_idx)
        return {
            "state_encoder.0.weight": self.state_fc1.weight[i].detach().clone(),
            "state_encoder.0.bias": self.state_fc1.bias[i].detach().clone(),
            "state_encoder.2.weight": self.state_fc2.weight[i].detach().clone(),
            "state_encoder.2.bias": self.state_fc2.bias[i].detach().clone(),
            "q_head.0.weight": self.q_fc1.weight[i].detach().clone(),
            "q_head.0.bias": self.q_fc1.bias[i].detach().clone(),
            "q_head.2.weight": self.q_fc2.weight[i].detach().clone(),
            "q_head.2.bias": self.q_fc2.bias[i].detach().clone(),
        }

    def load_agent_state_dict(self, agent_idx: int, state_dict: dict[str, torch.Tensor]) -> None:
        i = int(agent_idx)
        with torch.no_grad():
            self.state_fc1.weight[i].copy_(state_dict["state_encoder.0.weight"])
            self.state_fc1.bias[i].copy_(state_dict["state_encoder.0.bias"])
            self.state_fc2.weight[i].copy_(state_dict["state_encoder.2.weight"])
            self.state_fc2.bias[i].copy_(state_dict["state_encoder.2.bias"])
            self.q_fc1.weight[i].copy_(state_dict["q_head.0.weight"])
            self.q_fc1.bias[i].copy_(state_dict["q_head.0.bias"])
            self.q_fc2.weight[i].copy_(state_dict["q_head.2.weight"])
            self.q_fc2.bias[i].copy_(state_dict["q_head.2.bias"])


__all__ = [
    "BatchedIndependentLinear",
    "BatchedIndependentMPDQNActor",
    "BatchedIndependentMPDQNQNetwork",
    "MPDQNActor",
    "MPDQNQNetwork",
]
