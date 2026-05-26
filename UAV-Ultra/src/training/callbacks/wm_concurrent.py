"""WM concurrent (per-step) update callback.

每个 Q step 后跑 ``wm_updates_per_learn`` 次 WM grad step。对齐 baseline
``train_qmix_value_expansion.py`` 范式：QMIX 与 WM 同时更新，Q 的 TD target
通过 ``ValueExpansionCallback`` 注入 G_λ。

Stage 7 起 ``WMConcurrentCallback`` 是这个范式的官方实现；旧名
``WMAlternatingCallback`` 仍可 import 但触发 ``FutureWarning``，作为
``WMConcurrentCallback`` 的 alias。L_VC 由本 callback 装配进 WM 训练目标。
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Callable, Optional, Tuple

import torch

from src.algorithms.world_model.action_encoding import encode_joint_action_exec, exec_action_dim
from src.algorithms.world_model.losses import compute_wm_losses
from src.algorithms.world_model.model import JointWorldModel, JointWorldModelConfig
from src.algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer
from src.algorithms.world_model.value_expansion import TDlambdaConfig, rollout_td_lambda_return
from src.config import specs
from src.training.callbacks.base import TrainHookContext, TrainingCallback


_MISSING = object()


def _diff_cfg(saved: dict, current: dict, *, atol: float = 1e-12) -> list[str]:
    diffs: list[str] = []
    for key in sorted(set(saved) | set(current)):
        a = saved.get(key, _MISSING)
        b = current.get(key, _MISSING)
        if a is _MISSING:
            diffs.append(f"{key}: <missing in saved> vs {b!r}")
        elif b is _MISSING:
            diffs.append(f"{key}: {a!r} vs <missing in current>")
        elif isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, abs_tol=atol):
                diffs.append(f"{key}: {a} vs {b}")
        elif a != b:
            diffs.append(f"{key}: {a!r} vs {b!r}")
    return diffs


def _vc_eta(episode: int, *, warmup_ep: int, ramp_end_ep: int, v_max: float) -> float:
    """Linear ramp eta(ep) from 0 to ``v_max`` over ``[warmup_ep, ramp_end_ep]``."""
    ep = int(episode)
    w = int(warmup_ep)
    e = int(ramp_end_ep)
    vmax = float(v_max)
    if ep < w:
        return 0.0
    if ep >= e:
        return vmax
    span = max(1, e - w)
    return vmax * float(ep - w) / float(span)


class WMConcurrentCallback(TrainingCallback):
    """Per-step concurrent WM update; supports L_VC + linear ramp eta schedule."""

    name = "wm_concurrent"

    def __init__(
        self,
        *,
        env_cfg: Any,
        algo_cfg: Any,
        shared: dict[str, Any] | None = None,
        seq_len: int = 4,
        lr: float = 1e-3,
        curriculum_active: bool = False,
    ):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.shared = shared if shared is not None else {}
        self.seq_len = int(getattr(algo_cfg, "value_expansion_seq_len", seq_len))
        self.lr = float(getattr(algo_cfg, "wm_lr", lr))
        self.wm_max_grad_norm = float(getattr(algo_cfg, "wm_max_grad_norm", 0.0))
        self.curriculum_active = bool(curriculum_active)
        self.base_param_dim = int(specs.total_param_dim(env_cfg))
        # Stage 7：从 algo_cfg 取 WM batch + ramp 参数；callback 单测时 algo_cfg 可能不带这些
        # 字段，提供 baseline 默认值兜底。
        self.wm_batch_size = int(getattr(algo_cfg, "wm_batch_size", 512))
        self.wm_updates_per_learn = int(getattr(algo_cfg, "wm_updates_per_learn", 2))
        self.vc_eta_max = float(getattr(algo_cfg, "wm_vc_eta_max", 0.0))
        self.vc_warmup_ep = int(getattr(algo_cfg, "wm_vc_warmup_ep", 0))
        self.vc_ramp_end_ep = int(getattr(algo_cfg, "wm_vc_ramp_end_ep", 0))
        self.value_expansion_model_warmup_ep = int(
            getattr(algo_cfg, "value_expansion_model_warmup_ep", 200)
        )
        self.value_expansion_ramp_start_ep = int(
            getattr(algo_cfg, "value_expansion_ramp_start_ep", 300)
        )
        self.value_expansion_ramp_end_ep = int(
            getattr(algo_cfg, "value_expansion_ramp_end_ep", 500)
        )
        self.last_wm_result: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # attach + shared
    # ------------------------------------------------------------------
    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        super().attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=n_envs)
        device = getattr(trainer, "device", torch.device("cpu"))
        device = torch.device(device)
        n_agents = int(env_cfg.n_ch)
        global_state_dim = n_agents * int(specs.state_dim(env_cfg))
        action_dim = exec_action_dim(
            n_agents=n_agents,
            n_des=int(specs.n_des(env_cfg)),
            n_channel=int(env_cfg.n_channel),
            param_dim=int(specs.param_dim_per_action(env_cfg)),
        )

        self.wm_cfg = self.shared.get("wm_cfg") or JointWorldModelConfig(
            state_dim=global_state_dim,
            action_dim=action_dim,
            hidden_dim=int(getattr(algo_cfg, "wm_hidden_dim", 256)),
            n_layers=int(getattr(algo_cfg, "wm_n_layers", 1)),
            stochastic_dim=int(getattr(algo_cfg, "wm_stochastic_dim", 32)),
            kl_beta=float(getattr(algo_cfg, "wm_kl_beta", 0.1)),
            free_nats=float(getattr(algo_cfg, "wm_free_nats", 1.0)),
        )
        self.world_model = self.shared.get("world_model") or JointWorldModel(self.wm_cfg).to(device)
        self.td_cfg = self.shared.get("td_cfg") or TDlambdaConfig(
            gamma=float(algo_cfg.gamma),
            lam=float(getattr(algo_cfg, "value_expansion_td_lambda", 0.8)),
            rollout_k=int(getattr(algo_cfg, "value_expansion_rollout_k", 4)),
        )
        self.wm_replay = self.shared.get("wm_replay") or WorldModelSequenceReplayBuffer(
            n_envs=max(1, int(n_envs)),
            capacity=int(
                getattr(
                    algo_cfg,
                    "wm_buffer_capacity",
                    getattr(algo_cfg, "buffer_capacity", 100_000),
                )
            ),
        )
        self.optimizer = self.shared.get("wm_optimizer") or torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.lr,
        )
        # value_teacher 由 ValueExpansionCallback.attach 填充到 shared；可能为 None
        # （单测/未启用 value_expansion 路径）。
        self.value_teacher = self.shared.get("value_teacher")
        self.shared.update(
            {
                "world_model": self.world_model,
                "wm_replay": self.wm_replay,
                "wm_cfg": self.wm_cfg,
                "td_cfg": self.td_cfg,
                "wm_optimizer": self.optimizer,
            }
        )

    # ------------------------------------------------------------------
    # sequence sampling
    # ------------------------------------------------------------------
    def _sample_tensors(self, *, batch_size: int) -> Optional[dict[str, torch.Tensor]]:
        if self.wm_replay.count_ready_envs(seq_len=self.seq_len) <= 0:
            return None
        try:
            sample = self.wm_replay.sample_sequences(batch_size=batch_size, seq_len=self.seq_len)
        except (RuntimeError, ValueError):
            return None

        device = next(self.world_model.parameters()).device
        state_seq = torch.from_numpy(sample["state_seq"]).to(device)
        next_state_seq = torch.from_numpy(sample["next_state_seq"]).to(device)
        reward_seq = torch.from_numpy(sample["reward_seq"]).to(device)
        ad_seq = torch.from_numpy(sample["action_discrete_seq"]).long().to(device)
        ap_seq = torch.from_numpy(sample["action_params_seq"]).to(device)
        bsz, seq_len = int(ad_seq.shape[0]), int(ad_seq.shape[1])
        action_enc = encode_joint_action_exec(
            ad_seq.reshape(bsz * seq_len, int(self.env_cfg.n_ch)),
            ap_seq.reshape(bsz * seq_len, int(self.env_cfg.n_ch), self.base_param_dim),
            n_agents=int(self.env_cfg.n_ch),
            n_channel=int(self.env_cfg.n_channel),
            n_des=int(specs.n_des(self.env_cfg)),
            n_actions=int(specs.action_dim(self.env_cfg)),
            param_dim=int(specs.param_dim_per_action(self.env_cfg)),
            power_min_dbm=float(self.env_cfg.uav_power_min),
            power_max_dbm=float(self.env_cfg.uav_power_max),
        ).reshape(bsz, seq_len, -1)
        return {
            "state_seq": state_seq,
            "next_state_seq": next_state_seq,
            "reward_seq": reward_seq,
            "action_enc_seq": action_enc,
            "action_discrete_seq": ad_seq,
            "action_params_seq": ap_seq,
        }

    # ------------------------------------------------------------------
    # L_VC assembly
    # ------------------------------------------------------------------
    def _build_policy_fn(self) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        env_cfg = self.env_cfg
        n_agents = int(env_cfg.n_ch)
        n_channel = int(env_cfg.n_channel)
        n_des = int(specs.n_des(env_cfg))
        n_actions = int(specs.action_dim(env_cfg))
        param_dim = int(specs.param_dim_per_action(env_cfg))
        power_min = float(env_cfg.uav_power_min)
        power_max = float(env_cfg.uav_power_max)
        value_teacher = self.value_teacher

        def _policy_fn(s_flat: torch.Tensor):
            with torch.no_grad():
                a_star, p_star = value_teacher.greedy_action(s_flat)
                u_star = encode_joint_action_exec(
                    a_star,
                    p_star,
                    n_agents=n_agents,
                    n_channel=n_channel,
                    n_des=n_des,
                    n_actions=n_actions,
                    param_dim=param_dim,
                    power_min_dbm=power_min,
                    power_max_dbm=power_max,
                )
            return u_star, a_star, p_star

        return _policy_fn

    def _maybe_vc_targets(
        self,
        *,
        batch: dict[str, torch.Tensor],
        eta: float,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if eta <= 0.0 or self.value_teacher is None:
            return None, None
        state_seq = batch["state_seq"]
        ad_seq = batch["action_discrete_seq"]
        ap_seq = batch["action_params_seq"]
        s_t = state_seq[:, -1, :]
        ad_t = ad_seq[:, -1, :]
        ap_t = ap_seq[:, -1, :, :]
        # q_teacher：**不**包 no_grad（target net 的 detach 保护已经够）
        q_teacher = self.value_teacher.q_tot_target(s_t, ad_t, ap_t)
        policy_fn = self._build_policy_fn()
        # rollout：**不**包 no_grad，梯度回传到 WM
        g_lambda, _ = rollout_td_lambda_return(
            wm=self.world_model,
            state_seq=state_seq,
            action_seq=batch["action_enc_seq"],
            policy_fn=policy_fn,
            q_tot_target_fn=self.value_teacher.q_tot_target,
            cfg=self.td_cfg,
        )
        # 两者都过 trainer 的 value-target clip（与 Q-learning 同源）
        clip_fn = getattr(self.trainer, "_clip_value_target", None)
        if callable(clip_fn):
            q_teacher = clip_fn(q_teacher)
            g_lambda = clip_fn(g_lambda)
        return q_teacher, g_lambda

    # ------------------------------------------------------------------
    # WM update
    # ------------------------------------------------------------------
    def _eta_for_episode(self, episode: int) -> float:
        if self.curriculum_active:
            return _vc_eta(
                episode,
                warmup_ep=self.value_expansion_ramp_start_ep,
                ramp_end_ep=self.value_expansion_ramp_end_ep,
                v_max=self.vc_eta_max,
            )
        return _vc_eta(
            episode,
            warmup_ep=self.vc_warmup_ep,
            ramp_end_ep=self.vc_ramp_end_ep,
            v_max=self.vc_eta_max,
        )

    def update_world_model_once(
        self,
        *,
        batch_size: int,
        episode: int,
    ) -> Optional[dict[str, float]]:
        batch = self._sample_tensors(batch_size=batch_size)
        if batch is None:
            return None
        self.world_model.train()
        out = self.world_model.observe(
            state_seq=batch["state_seq"],
            action_seq=batch["action_enc_seq"],
            sample=True,
        )
        eta = self._eta_for_episode(episode)
        q_teacher_t, g_lambda_t = self._maybe_vc_targets(batch=batch, eta=eta)
        details, total = compute_wm_losses(
            out,
            state_seq=batch["state_seq"],
            next_state_seq=batch["next_state_seq"],
            reward_seq=batch["reward_seq"],
            free_nats=float(self.wm_cfg.free_nats),
            kl_beta=float(self.wm_cfg.kl_beta),
            q_teacher=q_teacher_t,
            g_lambda=g_lambda_t,
            eta=float(eta),
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.wm_max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.wm_max_grad_norm)
        self.optimizer.step()
        result: dict[str, float] = {
            "wm_loss": float(total.item()),
            "wm_L_S": float(details["L_S"].item()),
            "wm_L_R": float(details["L_R"].item()),
            "wm_L_KL": float(details["L_KL"].item()),
            "wm_eta": float(eta),
        }
        if "L_VC" in details:
            result["wm_L_VC"] = float(details["L_VC"].item())
        return result

    def _run_wm_updates(self, *, episode: int) -> Optional[dict[str, float]]:
        if self.curriculum_active and int(episode) < int(self.value_expansion_model_warmup_ep):
            return None
        last: Optional[dict[str, float]] = None
        for _ in range(int(max(1, self.wm_updates_per_learn))):
            res = self.update_world_model_once(batch_size=self.wm_batch_size, episode=episode)
            if res is not None:
                last = res
                self.last_wm_result.append(res)
        return last

    def after_train_step(self, context: TrainHookContext, result: Optional[dict[str, float]]) -> None:
        if result is None:
            ep = int(context.episode)
            in_wm_only = (
                self.curriculum_active
                and int(self.value_expansion_model_warmup_ep)
                <= ep
                < int(self.value_expansion_ramp_start_ep)
            )
            if not in_wm_only:
                return
            self._run_wm_updates(episode=ep)
            return
        self._run_wm_updates(episode=int(context.episode))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "wm_state_dict": self.world_model.state_dict(),
            "opt_state_dict": self.optimizer.state_dict(),
            "wm_cfg": asdict(self.wm_cfg),
            "td_cfg": asdict(self.td_cfg),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        expected = {"wm_state_dict", "opt_state_dict", "wm_cfg", "td_cfg"}
        if strict and set(state) != expected:
            raise ValueError(
                f"{self.name}: expected state keys {sorted(expected)}, got {sorted(state)}"
            )
        if strict:
            if "wm_cfg" in state:
                diffs = _diff_cfg(dict(state["wm_cfg"]), asdict(self.wm_cfg))
                if diffs:
                    raise ValueError(f"{self.name}: wm_cfg mismatch: {diffs}")
            if "td_cfg" in state:
                diffs = _diff_cfg(dict(state["td_cfg"]), asdict(self.td_cfg))
                if diffs:
                    raise ValueError(f"{self.name}: td_cfg mismatch: {diffs}")
        if "wm_state_dict" in state:
            self.world_model.load_state_dict(state["wm_state_dict"], strict=strict)
        if "opt_state_dict" in state:
            self.optimizer.load_state_dict(state["opt_state_dict"])


__all__ = ["WMConcurrentCallback", "_diff_cfg", "_vc_eta"]
