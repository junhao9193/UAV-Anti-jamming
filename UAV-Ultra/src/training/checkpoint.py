"""Checkpoint helpers for Stage 5+ training and Stage 6 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def trainer_state_dict(trainer: Any, *, algorithm: str) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {"algorithm": str(algorithm)}

    agents = getattr(trainer, "agents", None)
    if agents is not None:
        checkpoint["agents"] = []
        for agent in agents:
            agent_state = {
                "actor": agent.actor.state_dict(),
                "q_net": agent.q_net.state_dict(),
                "target_actor": agent.target_actor.state_dict(),
                "target_q_net": agent.target_q_net.state_dict(),
            }
            if hasattr(agent, "jammer_predictor"):
                agent_state["jammer_predictor"] = agent.jammer_predictor.state_dict()
            if hasattr(agent, "target_jammer_predictor"):
                agent_state["target_jammer_predictor"] = agent.target_jammer_predictor.state_dict()
            if hasattr(agent, "jammer_predictor_opt"):
                agent_state["jammer_predictor_opt"] = agent.jammer_predictor_opt.state_dict()
            checkpoint["agents"].append(agent_state)

        if agents:
            first = agents[0]
            cfg: dict[str, Any] = {
                "state_dim": int(first.state_dim),
                "n_actions": int(first.n_actions),
                "param_dim": int(first.param_dim),
            }
            # Stage 8：显式区分 JP-on/off 的字段（strict 双向校验）
            jp_enabled = hasattr(first, "jammer_predictor")
            cfg["jammer_prediction_enabled"] = bool(jp_enabled)
            if jp_enabled:
                cfg["n_channel"] = int(getattr(first, "n_channel", 0))
                cfg["jammer_history_len"] = int(getattr(first, "jammer_history_len", 0))
            checkpoint["agent_config"] = cfg

        if hasattr(trainer, "mixer") and getattr(trainer, "mixer") is not None:
            checkpoint["mixer"] = trainer.mixer.state_dict()
            if hasattr(trainer, "target_mixer") and getattr(trainer, "target_mixer") is not None:
                checkpoint["target_mixer"] = trainer.target_mixer.state_dict()
            checkpoint["mixer_config"] = {
                "mixer_class": trainer.mixer.__class__.__name__,
                "n_agents": int(getattr(trainer, "n_agents")),
                "global_state_dim": int(getattr(trainer, "global_state_dim")),
            }
    elif hasattr(trainer, "actor") and hasattr(trainer, "critic"):
        checkpoint["actor"] = trainer.actor.state_dict()
        checkpoint["critic"] = trainer.critic.state_dict()
        if hasattr(trainer, "actor_opt"):
            checkpoint["actor_opt"] = trainer.actor_opt.state_dict()
        if hasattr(trainer, "critic_opt"):
            checkpoint["critic_opt"] = trainer.critic_opt.state_dict()
        checkpoint["agent_config"] = {
            "obs_dim": int(trainer.obs_dim),
            "n_actions": int(trainer.n_actions),
            "cont_dim": int(trainer.cont_dim),
            "n_agents": int(trainer.n_agents),
            "global_state_dim": int(trainer.global_state_dim),
        }
    else:
        raise ValueError("Unsupported trainer type for checkpointing")

    return checkpoint


def save_checkpoint(
    *,
    path: Path,
    algorithm: str,
    trainer: Any,
    callbacks: Sequence[Any] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint = trainer_state_dict(trainer, algorithm=algorithm)
    checkpoint["callbacks"] = {cb.name: cb.state_dict() for cb in callbacks}
    if extra is not None:
        checkpoint["extra"] = dict(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(p))
    return p


def _load_checkpoint(
    checkpoint_or_path: Mapping[str, Any] | str | Path,
    *,
    device: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(checkpoint_or_path, Mapping):
        return checkpoint_or_path
    map_location = torch.device(device)
    try:
        return torch.load(str(checkpoint_or_path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_or_path), map_location=map_location)


def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(
            f"{context}: missing required checkpoint key(s) {missing}. "
            "checkpoint must be produced by the Stage 5+ training runner."
        )


def _check_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise ValueError(f"checkpoint {field}={actual!r} does not match current {field}={expected!r}")


def _validate_algorithm(checkpoint: Mapping[str, Any], *, algorithm: str, strict: bool) -> None:
    if strict:
        _require_keys(checkpoint, ["algorithm", "callbacks", "agent_config"], context="checkpoint")
    if "algorithm" in checkpoint:
        saved = str(checkpoint["algorithm"]).lower()
        expected = str(algorithm).lower()
        if strict and saved != expected:
            raise ValueError(f"checkpoint algorithm={saved!r} does not match requested {expected!r}")


def _load_dqn_trainer_state(
    trainer: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool,
    load_optimizers: bool = False,
) -> None:
    _require_keys(checkpoint, ["agents"], context="DQN checkpoint")
    agents_sd = checkpoint["agents"]
    if not isinstance(agents_sd, Sequence):
        raise ValueError("DQN checkpoint agents must be a sequence")
    if strict and len(agents_sd) != int(trainer.n_agents):
        raise ValueError(
            f"checkpoint agents={len(agents_sd)} does not match current n_agents={trainer.n_agents}"
        )

    trainer_jp_on = hasattr(trainer.agents[0], "jammer_predictor") if trainer.agents else False
    if strict:
        cfg = dict(checkpoint["agent_config"])
        expected = {
            "state_dim": int(trainer.state_dim),
            "n_actions": int(trainer.n_actions),
            "param_dim": int(trainer.param_dim),
        }
        for key, expected_value in expected.items():
            if key not in cfg:
                raise ValueError(
                    f"checkpoint agent_config missing {key!r}. "
                    "checkpoint must be produced by the Stage 5+ training runner."
                )
            _check_equal(cfg[key], expected_value, field=f"agent_config.{key}")
        # Stage 8：JP-on/off 双向 strict
        ckpt_jp_on = bool(cfg.get("jammer_prediction_enabled", False))
        if trainer_jp_on != ckpt_jp_on:
            raise ValueError(
                f"agent_config.jammer_prediction_enabled mismatch: "
                f"checkpoint={ckpt_jp_on}, trainer={trainer_jp_on}"
            )
        if trainer_jp_on:
            for key in ("n_channel", "jammer_history_len"):
                if key not in cfg:
                    raise ValueError(
                        f"JP-on checkpoint agent_config missing {key!r}; "
                        "must be produced by Stage 8+ training runner with JP enabled."
                    )
                _check_equal(
                    cfg[key], int(getattr(trainer.agents[0], key)),
                    field=f"agent_config.{key}",
                )

    for idx, (agent, state) in enumerate(zip(trainer.agents, agents_sd)):
        _require_keys(state, ["actor", "q_net", "target_actor", "target_q_net"], context=f"agent[{idx}]")
        agent.actor.load_state_dict(state["actor"], strict=strict)
        agent.q_net.load_state_dict(state["q_net"], strict=strict)
        agent.target_actor.load_state_dict(state["target_actor"], strict=strict)
        agent.target_q_net.load_state_dict(state["target_q_net"], strict=strict)
        if "jammer_predictor" in state:
            if strict and not hasattr(agent, "jammer_predictor"):
                raise ValueError(f"agent[{idx}] checkpoint contains jammer_predictor but trainer does not")
            if hasattr(agent, "jammer_predictor"):
                agent.jammer_predictor.load_state_dict(state["jammer_predictor"], strict=strict)
        if "target_jammer_predictor" in state:
            if strict and not hasattr(agent, "target_jammer_predictor"):
                raise ValueError(f"agent[{idx}] checkpoint contains target_jammer_predictor but trainer does not")
            if hasattr(agent, "target_jammer_predictor"):
                agent.target_jammer_predictor.load_state_dict(
                    state["target_jammer_predictor"],
                    strict=strict,
                )
        # Stage 8：optional jammer_predictor_opt 持久化（仅 load_optimizers=True 时恢复）
        if load_optimizers and "jammer_predictor_opt" in state:
            if hasattr(agent, "jammer_predictor_opt"):
                agent.jammer_predictor_opt.load_state_dict(state["jammer_predictor_opt"])

    if hasattr(trainer, "mixer") and getattr(trainer, "mixer") is not None:
        _require_keys(checkpoint, ["mixer", "target_mixer", "mixer_config"], context="mixer checkpoint")
        if strict:
            mixer_cfg = dict(checkpoint["mixer_config"])
            expected = {
                "mixer_class": trainer.mixer.__class__.__name__,
                "n_agents": int(trainer.n_agents),
                "global_state_dim": int(trainer.global_state_dim),
            }
            for key, expected_value in expected.items():
                if key not in mixer_cfg:
                    raise ValueError(
                        f"checkpoint mixer_config missing {key!r}. "
                        "checkpoint must be produced by the Stage 5+ training runner."
                    )
                _check_equal(mixer_cfg[key], expected_value, field=f"mixer_config.{key}")
        trainer.mixer.load_state_dict(checkpoint["mixer"], strict=strict)
        trainer.target_mixer.load_state_dict(checkpoint["target_mixer"], strict=strict)


def _load_mappo_trainer_state(
    trainer: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool,
    load_optimizers: bool,
) -> None:
    _require_keys(checkpoint, ["actor", "critic"], context="MAPPO checkpoint")
    if strict:
        cfg = dict(checkpoint["agent_config"])
        expected = {
            "obs_dim": int(trainer.obs_dim),
            "n_actions": int(trainer.n_actions),
            "cont_dim": int(trainer.cont_dim),
            "n_agents": int(trainer.n_agents),
            "global_state_dim": int(trainer.global_state_dim),
        }
        for key, expected_value in expected.items():
            if key not in cfg:
                raise ValueError(
                    f"checkpoint agent_config missing {key!r}. "
                    "checkpoint must be produced by the Stage 5+ training runner."
                )
            _check_equal(cfg[key], expected_value, field=f"agent_config.{key}")

    trainer.actor.load_state_dict(checkpoint["actor"], strict=strict)
    trainer.critic.load_state_dict(checkpoint["critic"], strict=strict)
    if load_optimizers:
        _require_keys(checkpoint, ["actor_opt", "critic_opt"], context="MAPPO optimizer checkpoint")
        trainer.actor_opt.load_state_dict(checkpoint["actor_opt"])
        trainer.critic_opt.load_state_dict(checkpoint["critic_opt"])


def load_trainer_state_dict(
    trainer: Any,
    checkpoint_or_path: Mapping[str, Any] | str | Path,
    algorithm: str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    load_optimizers: bool = False,
) -> Mapping[str, Any]:
    """Load trainer weights from a Stage 5+ checkpoint.

    DQN-family checkpoints intentionally do not persist the actor/Q/mixer
    optimizers. ``load_optimizers=True`` only restores the optional Stage 8
    jammer-predictor optimizer when present; otherwise it is a no-op on DQN.
    """
    algorithm = str(algorithm).lower()
    checkpoint = _load_checkpoint(checkpoint_or_path, device=device)
    _validate_algorithm(checkpoint, algorithm=algorithm, strict=strict)

    if hasattr(trainer, "agents"):
        _load_dqn_trainer_state(
            trainer,
            checkpoint,
            strict=strict,
            load_optimizers=bool(load_optimizers),
        )
    elif hasattr(trainer, "actor") and hasattr(trainer, "critic"):
        _load_mappo_trainer_state(
            trainer,
            checkpoint,
            strict=strict,
            load_optimizers=bool(load_optimizers),
        )
    else:
        raise ValueError("Unsupported trainer type for checkpoint loading")
    return checkpoint


_DEPRECATED_CALLBACK_KEY_ALIASES: dict[str, str] = {
    "wm_alternating": "wm_concurrent",
}


def _normalize_callback_state_keys(state: Mapping[str, Any]) -> dict[str, Any]:
    """Stage 7：把 checkpoint 里的旧 callback key（如 ``wm_alternating``）
    重命名为当前名（``wm_concurrent``）并触发 ``FutureWarning``。
    返回新 dict，不修改原对象。
    """
    import warnings

    out: dict[str, Any] = {}
    for k, v in state.items():
        target = _DEPRECATED_CALLBACK_KEY_ALIASES.get(k)
        if target is None:
            out[k] = v
            continue
        if target in out:
            raise ValueError(
                f"checkpoint contains both deprecated key {k!r} and current key {target!r}; "
                "remove one before reloading"
            )
        warnings.warn(
            f"checkpoint callback key {k!r} is deprecated; mapped to {target!r}",
            FutureWarning,
            stacklevel=3,
        )
        out[target] = v
    return out


def load_callback_states(
    callbacks: Sequence[Any],
    callback_state: Mapping[str, Any],
    *,
    strict: bool = True,
) -> None:
    callback_state = _normalize_callback_state_keys(callback_state)
    expected = {cb.name for cb in callbacks}
    provided = set(callback_state.keys())
    if strict and expected != provided:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ValueError(f"callback checkpoint key mismatch: missing={missing}, extra={extra}")
    for cb in callbacks:
        if cb.name in callback_state:
            cb.load_state_dict(callback_state[cb.name], strict=strict)


__all__ = [
    "load_callback_states",
    "load_trainer_state_dict",
    "save_checkpoint",
    "trainer_state_dict",
]
