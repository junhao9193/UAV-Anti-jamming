from __future__ import division

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HeuristicPolicy(Protocol):
    def select_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]: ...


@dataclass(frozen=True)
class HeuristicDims:
    n_channel: int
    n_des: int
    n_actions: int
    param_dim: int


def encode_channel_assignment(channels: np.ndarray, n_channel: int) -> int:
    action = 0
    base = 1
    for ch in np.asarray(channels, dtype=np.int64).reshape(-1):
        action += int(ch) * int(base)
        base *= int(n_channel)
    return int(action)


def build_flat_action(discrete_action: int, power_norm: np.ndarray, dims: HeuristicDims) -> tuple[int, np.ndarray]:
    params = np.zeros((int(dims.n_actions) * int(dims.param_dim),), dtype=np.float32)
    start = int(discrete_action) * int(dims.param_dim)
    params[start : start + int(dims.param_dim)] = np.clip(np.asarray(power_norm, dtype=np.float32), 0.0, 1.0)
    return int(discrete_action), params


def normalize_power_mode(policy_name: str, power_mode: str) -> str:
    """Normalize/validate policy-specific power modes.

    `quality_adaptive` is a sensible default for score-based heuristics, but
    random hopping has no channel-quality notion.  For random hopping we map the
    default `quality_adaptive` to a fixed mid-power baseline so
    `--policy random` works out of the box.
    """
    name = str(policy_name).lower()
    mode = str(power_mode).lower()
    valid_policies = {"random", "greedy_sensing", "max_csi", "min_interference"}
    if name not in valid_policies:
        raise ValueError(f"Unknown heuristic policy: {policy_name!r}; expected one of {sorted(valid_policies)}")
    if name == "random":
        if mode == "quality_adaptive":
            return "fixed_mid"
        if mode not in {"fixed_mid", "fixed_low", "random"}:
            raise ValueError(
                f"Policy 'random' supports power_mode in "
                f"{{fixed_mid, fixed_low, random}}, got {power_mode!r}"
            )
        return mode

    if mode == "random":
        raise ValueError(
            f"Policy {policy_name!r} is score-based and does not support "
            "`power_mode=random`; use quality_adaptive, fixed_mid, or fixed_low."
        )
    if mode not in {"quality_adaptive", "fixed_mid", "fixed_low"}:
        raise ValueError(
            f"Policy {policy_name!r} supports power_mode in "
            f"{{quality_adaptive, fixed_mid, fixed_low}}, got {power_mode!r}"
        )
    return mode


class RandomHoppingPolicy:
    def __init__(self, dims: HeuristicDims, *, power_mode: str = "fixed_mid", seed: int = 0):
        self.dims = dims
        self.power_mode = str(power_mode)
        self.rng = np.random.default_rng(int(seed))

    def _sample_power(self) -> np.ndarray:
        if self.power_mode == "fixed_mid":
            return np.full((self.dims.param_dim,), 0.5, dtype=np.float32)
        if self.power_mode == "fixed_low":
            return np.full((self.dims.param_dim,), 0.25, dtype=np.float32)
        if self.power_mode == "random":
            return self.rng.uniform(0.0, 1.0, size=(self.dims.param_dim,)).astype(np.float32)
        raise ValueError(f"Unknown power_mode: {self.power_mode}")

    def select_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        channels = self.rng.integers(0, self.dims.n_channel, size=(self.dims.n_des,), dtype=np.int64)
        discrete = encode_channel_assignment(channels, self.dims.n_channel)
        return build_flat_action(discrete, self._sample_power(), self.dims)


class ScoreBasedPolicy:
    def __init__(
        self,
        dims: HeuristicDims,
        *,
        csi_weight: float,
        sensing_weight: float,
        reuse_penalty: float,
        power_mode: str = "quality_adaptive",
    ):
        self.dims = dims
        self.csi_weight = float(csi_weight)
        self.sensing_weight = float(sensing_weight)
        self.reuse_penalty = float(reuse_penalty)
        self.power_mode = str(power_mode)

    def _split_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        csi_len = int(self.dims.n_des * self.dims.n_channel)
        if obs.size < csi_len + int(self.dims.n_channel):
            raise ValueError(
                f"Observation too short for heuristic policy: got {obs.size}, expected >= {csi_len + self.dims.n_channel}"
            )
        csi = obs[:csi_len].reshape(self.dims.n_des, self.dims.n_channel)
        sensing = obs[csi_len : csi_len + self.dims.n_channel]
        return csi, sensing

    def _power_from_quality(self, selected_quality: np.ndarray) -> np.ndarray:
        if self.power_mode == "fixed_mid":
            return np.full((self.dims.param_dim,), 0.5, dtype=np.float32)
        if self.power_mode == "fixed_low":
            return np.full((self.dims.param_dim,), 0.25, dtype=np.float32)
        if self.power_mode != "quality_adaptive":
            raise ValueError(f"Unknown power_mode: {self.power_mode}")

        # In current env observation, smaller CSI (pathloss-normalized) means a better link.
        # Use quality = -CSI so larger means better, then allocate lower power on better links.
        quality = np.asarray(selected_quality, dtype=np.float32)
        power = 0.6 - 0.30 * np.clip(quality, -1.0, 1.0)
        return np.clip(power, 0.10, 0.95).astype(np.float32)

    def _score_matrix(self, csi: np.ndarray, sensing: np.ndarray) -> np.ndarray:
        quality = -np.asarray(csi, dtype=np.float32)
        occupancy = np.asarray(sensing, dtype=np.float32).reshape(1, -1)
        return self.csi_weight * quality - self.sensing_weight * occupancy

    def select_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        csi, sensing = self._split_obs(obs)
        score = self._score_matrix(csi, sensing)

        chosen_channels = []
        chosen_quality = []
        used = set()
        for j in range(self.dims.n_des):
            score_j = score[j].copy()
            if self.reuse_penalty > 0.0 and used:
                for ch in used:
                    score_j[int(ch)] -= float(self.reuse_penalty)
            ch = int(np.argmax(score_j))
            chosen_channels.append(ch)
            chosen_quality.append(-float(csi[j, ch]))
            used.add(ch)

        channels_arr = np.asarray(chosen_channels, dtype=np.int64)
        power_norm = self._power_from_quality(np.asarray(chosen_quality, dtype=np.float32))
        discrete = encode_channel_assignment(channels_arr, self.dims.n_channel)
        return build_flat_action(discrete, power_norm, self.dims)


class GreedySensingPolicy(ScoreBasedPolicy):
    def __init__(self, dims: HeuristicDims, *, power_mode: str = "quality_adaptive"):
        super().__init__(
            dims,
            csi_weight=1.0,
            sensing_weight=0.9,
            reuse_penalty=0.15,
            power_mode=power_mode,
        )


class MaxCSIPolicy(ScoreBasedPolicy):
    def __init__(self, dims: HeuristicDims, *, power_mode: str = "quality_adaptive"):
        super().__init__(
            dims,
            csi_weight=1.0,
            sensing_weight=0.0,
            reuse_penalty=0.10,
            power_mode=power_mode,
        )


class MinInterferencePolicy(ScoreBasedPolicy):
    def __init__(self, dims: HeuristicDims, *, power_mode: str = "fixed_mid"):
        super().__init__(
            dims,
            csi_weight=0.25,
            sensing_weight=1.2,
            reuse_penalty=0.30,
            power_mode=power_mode,
        )


def build_heuristic_policy(policy_name: str, dims: HeuristicDims, *, seed: int = 0, power_mode: str = "quality_adaptive"):
    name = str(policy_name).lower()
    power_mode = normalize_power_mode(name, power_mode)
    if name == "random":
        return RandomHoppingPolicy(dims, power_mode=power_mode, seed=int(seed))
    if name == "greedy_sensing":
        return GreedySensingPolicy(dims, power_mode=power_mode)
    if name == "max_csi":
        return MaxCSIPolicy(dims, power_mode=power_mode)
    if name == "min_interference":
        return MinInterferencePolicy(dims, power_mode=power_mode)
    raise ValueError(f"Unknown heuristic policy: {policy_name}")


__all__ = [
    "HeuristicDims",
    "RandomHoppingPolicy",
    "GreedySensingPolicy",
    "MaxCSIPolicy",
    "MinInterferencePolicy",
    "build_heuristic_policy",
    "encode_channel_assignment",
    "build_flat_action",
    "normalize_power_mode",
]
