"""Callback protocol and canonical Stage 5+ callback builder."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence


CANONICAL_CALLBACK_ORDER: tuple[str, ...] = (
    "policy_mobility",
    "value_expansion",
    "wm_concurrent",
    "wm_block_alternating",
    "jammer_prediction",
    "critic_stable",
)
ALLOWED_CALLBACKS = set(CANONICAL_CALLBACK_ORDER)
# Stage 7：旧名 → 新名 alias。canonicalize 把 list 内字符串替换并触发 FutureWarning。
_CALLBACK_NAME_ALIASES: dict[str, str] = {
    "wm_alternating": "wm_concurrent",
}
_WM_TRAINING_CALLBACKS = {"wm_concurrent", "wm_block_alternating"}


@dataclass
class TrainHookContext:
    trainer: Any
    episode: int
    step: int


class TrainingCallback:
    """Base class for action-time and train-time hooks."""

    name = "callback"

    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        self.trainer = trainer
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.n_envs = int(n_envs)

    def on_action_selected(self, actions: Any) -> Any:
        return actions

    def on_transition_batch(
        self,
        *,
        states: Any,
        actions: Any,
        action_discrete: Any,
        action_params: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
        infos: Sequence[dict],
    ) -> None:
        return None

    def should_skip_q_update(self, context: TrainHookContext) -> bool:
        """Stage 7：若返回 True，CallbackManager 跳过整个 Q 训练分支（包括基类 ``trainer.train_step``
        与其它 callback 的 ``on_train_step``），result 直接设为 ``None``。``after_train_step`` 仍会被调用。
        """
        return False

    def on_train_step(self, context: TrainHookContext) -> Optional[dict[str, float]]:
        return None

    def on_aux_loss(self, trainer: Any, batch: Any, context: TrainHookContext) -> Optional[Any]:
        """Stage 8：返回标量 loss tensor，与 critic loss 同 batch + 同 backward。

        - ``trainer``: 当前 trainer 实例（callback 读 ``trainer.agents`` 等）。
        - ``batch``: 当前 critic_step 的 tensor batch dict（含可选 JP 字段）。
        - ``context``: TrainHookContext（含 episode/step）。

        callback **不**负责 optimizer step / target sync；trainer 主流程统一处理。
        """
        return None

    def after_train_step(self, context: TrainHookContext, result: Optional[dict[str, float]]) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        if strict and state:
            raise ValueError(f"{self.name}: unexpected callback state keys {sorted(state)}")


class CallbackManager:
    """Dispatch callbacks in canonical order.

    Train-time dispatch (Stage 7)::

        1. If any callback's ``should_skip_q_update(context)`` returns True,
           Q update is fully skipped: result = None.
        2. Otherwise, first non-None callback ``on_train_step`` result wins.
        3. If no callback short-circuited, ``trainer.train_step()`` runs.
        4. ``after_train_step`` is always called on every callback.
    """

    def __init__(self, callbacks: Sequence[TrainingCallback]):
        self.callbacks = list(callbacks)

    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        for cb in self.callbacks:
            cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=n_envs)

    def on_action_selected(self, actions: Any) -> Any:
        for cb in self.callbacks:
            actions = cb.on_action_selected(actions)
        return actions

    def on_transition_batch(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_transition_batch(**kwargs)

    def train_step(self, context: TrainHookContext) -> Optional[dict[str, float]]:
        """Run one trainer update; phase skip > first non-None > trainer.train_step().

        Stage 8：注入 ``trainer._aux_loss_fns = [cb.on_aux_loss for cb in callbacks]`` 供
        ``_collect_aux_losses`` 调用；并把 ``hook_context=context`` 透传到 trainer.train_step。
        """
        trainer = context.trainer
        trainer._aux_loss_fns = [cb.on_aux_loss for cb in self.callbacks]
        try:
            skip_decisions = [cb.should_skip_q_update(context) for cb in self.callbacks]
            skip = any(skip_decisions)
            result: Optional[dict[str, float]] = None
            if not skip:
                for cb in self.callbacks:
                    maybe = cb.on_train_step(context)
                    if maybe is not None:
                        result = maybe
                        break
                if result is None:
                    result = trainer.train_step(hook_context=context)
            for cb in self.callbacks:
                cb.after_train_step(context, result)
            return result
        finally:
            trainer._aux_loss_fns = []

    # ----------------------- Stage 8 JP helper API（runner 调用入口）-----------------------

    def _find_jp(self):
        from src.training.callbacks.jammer_prediction import JammerPredictionCallback
        for cb in self.callbacks:
            if isinstance(cb, JammerPredictionCallback):
                return cb
        return None

    def reset_jp_state(self, states: Any, *, episode: int) -> None:
        """JP-off no-op；JP-on 调 callback.reset_jp_state（reset history + 设 feature_scale）。"""
        jp = self._find_jp()
        if jp is not None:
            jp.reset_jp_state(states, episode=episode)

    def get_current_sensing_histories(self) -> Optional[Any]:
        """JP-off return None；JP-on return ``(n_envs, n_agents, H, C)``。"""
        jp = self._find_jp()
        return None if jp is None else jp.current_sensing_histories

    def get_jp_buffer_fields(self) -> Optional[dict[str, Any]]:
        """JP-off return None；JP-on return ``{'sensing_history','next_sensing_history','jammer_target'}``。

        ★ ``sensing_history`` 是 **当前 step select 时的 history**（current_*），与 baseline
        joint_replay_buffer.py 同 layout。``next_sensing_history`` 是这一步结束后的滚动结果。
        runner 必须在 ``commit_jp_history_swap`` **之前** 用这个字典调 store_transition_batch。
        """
        jp = self._find_jp()
        if jp is None:
            return None
        return {
            "sensing_history": jp.current_sensing_histories,
            "next_sensing_history": jp.next_sensing_histories,
            "jammer_target": jp.current_jammer_targets,
        }

    def commit_jp_history_swap(self) -> None:
        """JP-off no-op；JP-on: ``current ← next``，为下一步 select 准备。"""
        jp = self._find_jp()
        if jp is not None and jp.next_sensing_histories is not None:
            jp.current_sensing_histories = jp.next_sensing_histories.copy()

    def state_dict(self) -> dict[str, dict]:
        return {cb.name: cb.state_dict() for cb in self.callbacks}

    def __iter__(self):
        return iter(self.callbacks)


def canonicalize_callback_names(names: Sequence[str]) -> list[str]:
    """Map deprecated aliases to current names, then validate set + dependencies.

    Stage 7 changes:

    - ``wm_alternating`` is mapped to ``wm_concurrent`` with a ``FutureWarning``.
    - ``wm_concurrent`` and ``wm_block_alternating`` are mutually exclusive.
    - ``value_expansion`` requires exactly one of them.
    """
    raw = list(names)
    normalized: list[str] = []
    aliased: list[tuple[str, str]] = []
    for item in raw:
        if item in _CALLBACK_NAME_ALIASES:
            target = _CALLBACK_NAME_ALIASES[item]
            aliased.append((item, target))
            normalized.append(target)
        else:
            normalized.append(item)
    for src, dst in aliased:
        warnings.warn(
            f"callback name {src!r} is deprecated; use {dst!r}",
            FutureWarning,
            stacklevel=2,
        )
    unknown = sorted(set(normalized) - ALLOWED_CALLBACKS)
    if unknown:
        raise ValueError(f"unknown callback(s): {unknown}; valid: {sorted(ALLOWED_CALLBACKS)}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"callbacks must not contain duplicates: {raw!r}")
    cb_set = set(normalized)
    wm_present = cb_set & _WM_TRAINING_CALLBACKS
    if len(wm_present) > 1:
        raise ValueError(
            f"callbacks {sorted(wm_present)} are mutually exclusive; pick at most one of "
            f"{sorted(_WM_TRAINING_CALLBACKS)}"
        )
    if wm_present and "value_expansion" not in cb_set:
        raise ValueError(f"{next(iter(wm_present))!r} callback requires 'value_expansion'")
    if "value_expansion" in cb_set and not wm_present:
        raise ValueError(
            "value_expansion callback requires exactly one of "
            f"{sorted(_WM_TRAINING_CALLBACKS)}"
        )
    return [name for name in CANONICAL_CALLBACK_ORDER if name in cb_set]


def build_callbacks(names: Sequence[str], *, env_cfg: Any, algo_cfg: Any) -> CallbackManager:
    from src.training.callbacks.critic_stable import CriticStableCallback
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback
    from src.training.callbacks.policy_mobility import PolicyMobilityCallback
    from src.training.callbacks.value_expansion import ValueExpansionCallback
    from src.training.callbacks.wm_block_alternating import WMBlockAlternatingCallback
    from src.training.callbacks.wm_concurrent import WMConcurrentCallback

    ordered = canonicalize_callback_names(names)
    curriculum_active = "value_expansion" in ordered and "wm_concurrent" in ordered
    shared: dict[str, Any] = {}
    factories = {
        "policy_mobility": lambda: PolicyMobilityCallback(env_cfg=env_cfg),
        "value_expansion": lambda: ValueExpansionCallback(
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            shared=shared,
            curriculum_active=curriculum_active,
        ),
        "wm_concurrent": lambda: WMConcurrentCallback(
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            shared=shared,
            curriculum_active=curriculum_active,
        ),
        "wm_block_alternating": lambda: WMBlockAlternatingCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, shared=shared),
        "jammer_prediction": lambda: JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg),
        "critic_stable": lambda: CriticStableCallback(
            tau=float(getattr(algo_cfg, "critic_stable_tau", 0.005)),
            lr_scale=float(getattr(algo_cfg, "critic_stable_lr_scale", 1.0)),
            lr_decay_enabled=bool(getattr(algo_cfg, "critic_stable_lr_decay_enabled", False)),
            lr_decay_start_ep=int(getattr(algo_cfg, "critic_stable_lr_decay_start_ep", 1500)),
            lr_decay_end_ep=int(getattr(algo_cfg, "critic_stable_lr_decay_end_ep", 3000)),
            lr_decay_min=float(getattr(algo_cfg, "critic_stable_lr_decay_min", 0.1)),
        ),
    }
    return CallbackManager([factories[name]() for name in ordered])


__all__ = [
    "ALLOWED_CALLBACKS",
    "CANONICAL_CALLBACK_ORDER",
    "CallbackManager",
    "TrainHookContext",
    "TrainingCallback",
    "build_callbacks",
    "canonicalize_callback_names",
]
