"""Block-level alternating WM/QMIX training (block coordinate descent).

对齐 baseline ``Main/train/train_qmix_wm_alternating.py``：每
``wm_block_qmix_episodes`` ep 只训 Q（WM 冻结），下 ``wm_block_wm_episodes``
ep 只训 WM（agents/mixer 冻结）。两 phase 内 replay 与 sequence buffer 都
持续累积，action selection 始终是 ε-greedy。

复用 ``WMConcurrentCallback`` 的 L_VC 装配与 update 入口；本 callback 只
负责 phase 切换 + .train()/.eval() 管理 + Q phase 短路。
"""

from __future__ import annotations

from typing import Any, Optional

from src.training.callbacks.base import TrainHookContext
from src.training.callbacks.wm_concurrent import WMConcurrentCallback


class WMBlockAlternatingCallback(WMConcurrentCallback):
    name = "wm_block_alternating"

    def __init__(
        self,
        *,
        env_cfg: Any,
        algo_cfg: Any,
        shared: dict[str, Any] | None = None,
        seq_len: int = 4,
        lr: float = 1e-3,
    ):
        super().__init__(
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            shared=shared,
            seq_len=seq_len,
            lr=lr,
        )
        self.qmix_block_episodes = int(getattr(algo_cfg, "wm_block_qmix_episodes", 50))
        self.wm_block_episodes = int(getattr(algo_cfg, "wm_block_wm_episodes", 50))
        self._last_applied_phase: Optional[str] = None
        self.current_episode: Optional[int] = None
        self.phase_history: list[tuple[int, str]] = []  # (episode, phase) 首次进入新 phase 时追加

    # ------------------------------------------------------------------
    # phase 计算
    # ------------------------------------------------------------------
    def _phase_for_episode(self, episode: int) -> str:
        block_len = self.qmix_block_episodes + self.wm_block_episodes
        if block_len <= 0:
            return "qmix"
        pos = int(episode) % block_len
        return "qmix" if pos < self.qmix_block_episodes else "wm"

    def _apply_phase_modes(self, phase: str, trainer: Any) -> None:
        if self._last_applied_phase == phase:
            return
        agents = getattr(trainer, "agents", []) or []
        if phase == "qmix":
            for agent in agents:
                for net_name in (
                    "actor",
                    "q_net",
                    "target_actor",
                    "target_q_net",
                    "jammer_predictor",
                    "target_jammer_predictor",
                ):
                    net = getattr(agent, net_name, None)
                    if net is not None:
                        net.train()
            for name in ("mixer", "target_mixer"):
                net = getattr(trainer, name, None)
                if net is not None:
                    net.train()
            self.world_model.eval()
        elif phase == "wm":
            for agent in agents:
                for net_name in (
                    "actor",
                    "q_net",
                    "target_actor",
                    "target_q_net",
                    "jammer_predictor",
                    "target_jammer_predictor",
                ):
                    net = getattr(agent, net_name, None)
                    if net is not None:
                        net.eval()
            for name in ("mixer", "target_mixer"):
                net = getattr(trainer, name, None)
                if net is not None:
                    net.eval()
            self.world_model.train()
        self._last_applied_phase = phase

    # ------------------------------------------------------------------
    # callback hooks
    # ------------------------------------------------------------------
    def should_skip_q_update(self, context: TrainHookContext) -> bool:
        episode = int(context.episode)
        self.current_episode = episode
        phase = self._phase_for_episode(episode)
        self._apply_phase_modes(phase, context.trainer)
        if not self.phase_history or self.phase_history[-1][1] != phase:
            self.phase_history.append((episode, phase))
        return phase == "wm"

    def after_train_step(self, context: TrainHookContext, result: Optional[dict[str, float]]) -> None:
        episode = int(context.episode)
        self.current_episode = episode
        phase = self._phase_for_episode(episode)
        if phase != "wm":
            return
        # WM phase：result 一定是 None（被 should_skip_q_update 短路）。跑 WM update。
        self._run_wm_updates(episode=episode)

    # ------------------------------------------------------------------
    # persistence: extend parent state_dict with phase metadata
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "qmix_block_episodes": int(self.qmix_block_episodes),
                "wm_block_episodes": int(self.wm_block_episodes),
                "current_episode": self.current_episode,
                "phase_history": list(self.phase_history),
                "last_applied_phase": self._last_applied_phase,
                "last_wm_result": list(self.last_wm_result),
            }
        )
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        # Stage 7：父类只验 4 个 WM 相关 key；这里允许额外的 phase 字段。
        wm_keys = {"wm_state_dict", "opt_state_dict", "wm_cfg", "td_cfg"}
        extra_keys = {
            "qmix_block_episodes",
            "wm_block_episodes",
            "current_episode",
            "phase_history",
            "last_applied_phase",
            "last_wm_result",
        }
        if strict:
            unexpected = set(state) - (wm_keys | extra_keys)
            if unexpected:
                raise ValueError(
                    f"{self.name}: unexpected callback state keys {sorted(unexpected)}"
                )
        super().load_state_dict({k: state[k] for k in wm_keys if k in state}, strict=strict)
        if "qmix_block_episodes" in state:
            self.qmix_block_episodes = int(state["qmix_block_episodes"])
        if "wm_block_episodes" in state:
            self.wm_block_episodes = int(state["wm_block_episodes"])
        if "current_episode" in state:
            current = state["current_episode"]
            self.current_episode = None if current is None else int(current)
        if "phase_history" in state:
            self.phase_history = [tuple(item) for item in state["phase_history"]]
        if "last_wm_result" in state:
            self.last_wm_result = [dict(item) for item in state["last_wm_result"]]
        # Do not restore this gate directly: a freshly attached callback must
        # apply train/eval modes on its first phase check after reload.
        if "last_applied_phase" in state:
            self._last_applied_phase = None


__all__ = ["WMBlockAlternatingCallback"]
