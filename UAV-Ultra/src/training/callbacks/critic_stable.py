"""Critic-stability callback: soft target updates and LR scaling."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from src.training.callbacks.base import TrainHookContext, TrainingCallback

_DISABLED_HARD_TARGET_SYNC_INTERVAL = 10**9


def soft_update_module(target: torch.nn.Module, source: torch.nn.Module, *, tau: float) -> None:
    tau = float(tau)
    with torch.no_grad():
        for tgt, src in zip(target.parameters(), source.parameters()):
            tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)


def _optimizers_for_trainer(trainer: Any) -> Iterable[torch.optim.Optimizer]:
    for agent in getattr(trainer, "agents", []) or []:
        if hasattr(agent, "actor_opt"):
            yield agent.actor_opt
        if hasattr(agent, "q_opt"):
            yield agent.q_opt
        if hasattr(agent, "jammer_predictor_opt"):
            yield agent.jammer_predictor_opt
    if getattr(trainer, "mixer_opt", None) is not None:
        yield trainer.mixer_opt
    for name in ("actor_opt", "critic_opt"):
        if hasattr(trainer, name):
            yield getattr(trainer, name)


class CriticStableCallback(TrainingCallback):
    name = "critic_stable"

    def __init__(self, *, tau: float = 0.005, lr_scale: float = 1.0):
        self.tau = float(tau)
        self.lr_scale = float(lr_scale)
        self._base_lrs: dict[int, list[float]] = {}
        self._original_target_update_interval: int | None = None

    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        super().attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=n_envs)
        if hasattr(trainer, "target_update_interval"):
            current = int(getattr(trainer, "target_update_interval"))
            if self._original_target_update_interval is None:
                self._original_target_update_interval = current
            trainer.target_update_interval = _DISABLED_HARD_TARGET_SYNC_INTERVAL
        self.apply_lr_scale(trainer, self.lr_scale)

    def apply_lr_scale(self, trainer: Any, scale: Optional[float] = None) -> None:
        if scale is not None:
            self.lr_scale = float(scale)
        for opt in _optimizers_for_trainer(trainer):
            opt_id = id(opt)
            if opt_id not in self._base_lrs:
                self._base_lrs[opt_id] = [float(group["lr"]) for group in opt.param_groups]
            for group, base_lr in zip(opt.param_groups, self._base_lrs[opt_id]):
                group["lr"] = base_lr * self.lr_scale

    def apply_soft_target_update(self, trainer: Any) -> None:
        for agent in getattr(trainer, "agents", []) or []:
            soft_update_module(agent.target_actor, agent.actor, tau=self.tau)
            soft_update_module(agent.target_q_net, agent.q_net, tau=self.tau)
            if hasattr(agent, "target_jammer_predictor") and hasattr(agent, "jammer_predictor"):
                soft_update_module(agent.target_jammer_predictor, agent.jammer_predictor, tau=self.tau)
        if hasattr(trainer, "target_mixer") and hasattr(trainer, "mixer"):
            soft_update_module(trainer.target_mixer, trainer.mixer, tau=self.tau)

    def after_train_step(self, context: TrainHookContext, result: Optional[dict[str, float]]) -> None:
        if result is not None:
            self.apply_soft_target_update(context.trainer)

    def restore_target_sync(self, trainer: Any | None = None) -> None:
        target = trainer if trainer is not None else getattr(self, "trainer", None)
        if target is not None and self._original_target_update_interval is not None:
            target.target_update_interval = int(self._original_target_update_interval)

    def state_dict(self) -> dict:
        return {
            "tau": float(self.tau),
            "lr_scale": float(self.lr_scale),
            "original_target_update_interval": self._original_target_update_interval,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        allowed = {"tau", "lr_scale", "original_target_update_interval"}
        if strict and set(state) != allowed:
            raise ValueError(f"{self.name}: expected state keys {sorted(allowed)}, got {sorted(state)}")
        if "tau" in state:
            self.tau = float(state["tau"])
        if "lr_scale" in state:
            self.lr_scale = float(state["lr_scale"])
        if "original_target_update_interval" in state:
            value = state["original_target_update_interval"]
            self._original_target_update_interval = None if value is None else int(value)


__all__ = [
    "CriticStableCallback",
    "_DISABLED_HARD_TARGET_SYNC_INTERVAL",
    "soft_update_module",
]
