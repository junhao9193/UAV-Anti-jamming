"""IQL / VDN / QMIX / QPLEX one-step trainer regression against baseline.

These tests lock the Stage 4 trainer update loops, not just construction smoke:
copy baseline weights into the new trainer, store identical replay transitions,
sample with the same RNG seed, run one train_step, then compare losses and
state_dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.algorithms.iql.trainer import IQLTrainer
from src.algorithms.qmix.trainer import QMIXTrainer
from src.algorithms.qplex.trainer import QPLEXTrainer
from src.algorithms.vdn.trainer import VDNTrainer
from src.config import specs
from src.config.loader import load_algo_config, load_env_config


def _make_transitions(
    *,
    n_agents: int,
    state_dim: int,
    n_actions: int,
    total_param_dim: int,
    count: int,
) -> list[tuple[list[np.ndarray], list[tuple[int, np.ndarray]], np.ndarray, list[np.ndarray], bool]]:
    rng = np.random.RandomState(123)
    transitions = []
    for _ in range(count):
        states = [rng.randn(state_dim).astype(np.float32) for _ in range(n_agents)]
        next_states = [rng.randn(state_dim).astype(np.float32) for _ in range(n_agents)]
        actions = [
            (int(rng.randint(n_actions)), rng.rand(total_param_dim).astype(np.float32))
            for _ in range(n_agents)
        ]
        rewards = rng.randn(n_agents).astype(np.float32)
        transitions.append((states, actions, rewards, next_states, False))
    return transitions


def _store_transitions(trainer: Any, transitions: list[tuple]) -> None:
    for states, actions, rewards, next_states, done in transitions:
        trainer.store_transition(states, actions, rewards, next_states, done=done)


def _copy_agent_weights(old_trainer: Any, new_trainer: Any) -> None:
    for old_agent, new_agent in zip(old_trainer.agents, new_trainer.agents):
        new_agent.actor.load_state_dict(old_agent.actor.state_dict())
        new_agent.q_net.load_state_dict(old_agent.q_net.state_dict())
        new_agent.target_actor.load_state_dict(old_agent.target_actor.state_dict())
        new_agent.target_q_net.load_state_dict(old_agent.target_q_net.state_dict())


def _assert_module_state_close(label: str, old_module: torch.nn.Module, new_module: torch.nn.Module) -> None:
    old_state = old_module.state_dict()
    new_state = new_module.state_dict()
    assert set(new_state) == set(old_state), label
    for key, old_value in old_state.items():
        torch.testing.assert_close(
            new_state[key],
            old_value,
            rtol=0,
            atol=1e-7,
            msg=lambda msg, k=key: f"{label}.{k}: {msg}",
        )


def _assert_agents_close(name: str, old_trainer: Any, new_trainer: Any) -> None:
    for i, (old_agent, new_agent) in enumerate(zip(old_trainer.agents, new_trainer.agents)):
        _assert_module_state_close(f"{name}.agent{i}.actor", old_agent.actor, new_agent.actor)
        _assert_module_state_close(f"{name}.agent{i}.q_net", old_agent.q_net, new_agent.q_net)
        _assert_module_state_close(
            f"{name}.agent{i}.target_actor", old_agent.target_actor, new_agent.target_actor
        )
        _assert_module_state_close(
            f"{name}.agent{i}.target_q_net", old_agent.target_q_net, new_agent.target_q_net
        )


def _baseline_kwargs(env_cfg: Any, algo_cfg: Any, name: str) -> dict[str, Any]:
    n_agents = int(env_cfg.n_ch)
    state_dim = int(specs.state_dim(env_cfg))
    n_actions = int(specs.action_dim(env_cfg))
    param_dim = int(specs.param_dim_per_action(env_cfg))
    kwargs: dict[str, Any] = {
        "n_agents": n_agents,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "param_dim": param_dim,
        "buffer_capacity": int(algo_cfg.buffer_capacity),
        "batch_size": int(algo_cfg.batch_size),
        "gamma": float(algo_cfg.gamma),
        "lr_actor": float(algo_cfg.lr_actor),
        "lr_q": float(algo_cfg.lr_q),
        "target_update_interval": int(algo_cfg.target_update_interval),
        "use_amp": False,
        "max_grad_norm": float(algo_cfg.max_grad_norm),
        "device": "cpu",
    }
    if name != "iql":
        kwargs["global_state_dim"] = n_agents * state_dim
        kwargs["value_target_clip"] = float(algo_cfg.value_target_clip)
    if name in {"qmix", "qplex"}:
        kwargs["lr_mixer"] = float(algo_cfg.lr_mixer)
        kwargs["mixing_hidden_dim"] = int(algo_cfg.mixing_hidden_dim)
        kwargs["hypernet_hidden_dim"] = int(algo_cfg.hypernet_hidden_dim)
    if name == "qplex":
        kwargs["n_heads"] = int(algo_cfg.n_heads)
    return kwargs


@pytest.mark.parametrize(
    ("name", "baseline_module", "baseline_class", "new_class"),
    [
        ("iql", "algorithms.mpdqn.iql.trainer", "MPDQNJointIQLTrainer", IQLTrainer),
        ("vdn", "algorithms.mpdqn.vdn.trainer", "MPDQNVDNTrainer", VDNTrainer),
        ("qmix", "algorithms.mpdqn.qmix.trainer_greedy_actor", "MPDQNQMIXTrainer", QMIXTrainer),
        ("qplex", "algorithms.mpdqn.qplex.trainer", "MPDQNQPLEXTrainer", QPLEXTrainer),
    ],
)
def test_train_step_matches_baseline_after_one_update(
    baseline_import, name: str, baseline_module: str, baseline_class: str, new_class: type
):
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        name,
        overrides={
            "batch_size": 4,
            "buffer_capacity": 64,
            # Avoid target sync so the assertion isolates one critic + actor update.
            "target_update_interval": 999,
        },
    )
    baseline_cls = getattr(baseline_import(baseline_module), baseline_class)

    old_trainer = baseline_cls(**_baseline_kwargs(env_cfg, algo_cfg, name))
    new_trainer = new_class(env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")

    if name != "iql":
        new_trainer.mixer.load_state_dict(old_trainer.mixer.state_dict())
        new_trainer.target_mixer.load_state_dict(old_trainer.target_mixer.state_dict())
    _copy_agent_weights(old_trainer, new_trainer)

    n_agents = int(env_cfg.n_ch)
    state_dim = int(specs.state_dim(env_cfg))
    n_actions = int(specs.action_dim(env_cfg))
    total_param_dim = n_actions * int(specs.param_dim_per_action(env_cfg))
    transitions = _make_transitions(
        n_agents=n_agents,
        state_dim=state_dim,
        n_actions=n_actions,
        total_param_dim=total_param_dim,
        count=10,
    )
    _store_transitions(old_trainer, transitions)
    _store_transitions(new_trainer, transitions)

    np.random.seed(777)
    torch.manual_seed(777)
    old_result = old_trainer.train_step()
    np.random.seed(777)
    torch.manual_seed(777)
    new_result = new_trainer.train_step()

    assert new_result is not None
    assert old_result is not None
    for key in ("loss_q", "loss_actor"):
        assert new_result[key] == pytest.approx(old_result[key], rel=0, abs=1e-7)
    assert int(new_result.get("skipped", 0)) == int(old_result.get("skipped", 0))

    _assert_agents_close(name, old_trainer, new_trainer)
    if name != "iql":
        _assert_module_state_close(f"{name}.mixer", old_trainer.mixer, new_trainer.mixer)
        _assert_module_state_close(
            f"{name}.target_mixer", old_trainer.target_mixer, new_trainer.target_mixer
        )
        assert new_trainer.learn_steps == old_trainer.learn_steps == 1
    else:
        for old_agent, new_agent in zip(old_trainer.agents, new_trainer.agents):
            assert new_agent.learn_steps == old_agent.learn_steps == 1
