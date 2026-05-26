"""QMIX value-expansion callback."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from src.algorithms.common.value_decomp import TDTargetContext
from src.algorithms.qmix.trainer import QMIXTrainer
from src.algorithms.qmix.world_model_adapter import MPDQNQMIXDims, QMIXValueTeacher
from src.algorithms.world_model.action_encoding import encode_joint_action_exec, exec_action_dim
from src.algorithms.world_model.model import JointWorldModel, JointWorldModelConfig
from src.algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer
from src.algorithms.world_model.value_expansion import TDlambdaConfig
from src.config import specs
from src.training.callbacks.base import TrainHookContext, TrainingCallback


def tensor_batch_from_numpy(batch: dict[str, np.ndarray], *, device: torch.device) -> dict[str, torch.Tensor]:
    reward = np.asarray(batch["reward"], dtype=np.float32)
    done = np.asarray(batch["done"], dtype=np.float32)
    if reward.ndim != 1:
        raise ValueError(f"value_expansion requires global reward shape (B,), got {reward.shape}")
    if done.ndim != 1:
        raise ValueError(f"value_expansion requires done shape (B,), got {done.shape}")
    out = {
        "state": torch.from_numpy(batch["state"]).to(device),
        "action_discrete": torch.from_numpy(batch["action_discrete"]).long().to(device),
        "action_params": torch.from_numpy(batch["action_params"]).to(device),
        "reward": torch.from_numpy(reward).to(device).view(-1, 1),
        "next_state": torch.from_numpy(batch["next_state"]).to(device),
        "done": torch.from_numpy(done).to(device).view(-1, 1),
    }
    # Stage 8：透传可选 JP 字段，让 [value_expansion, wm_*, jammer_prediction] 组合下
    # JP on_aux_loss 能从 train_step_from_batch 路径拿到 sensing_history。
    for key in ("sensing_history", "next_sensing_history", "jammer_target"):
        if key in batch:
            out[key] = torch.from_numpy(np.asarray(batch[key], dtype=np.float32)).to(device)
    return out


class ValueExpansionCallback(TrainingCallback):
    """QMIX value-expansion target hook.

    This callback depends on ``wm_alternating`` for world-model updates and
    persistence; its own state only covers scalar target-mixing parameters.
    """

    name = "value_expansion"

    def __init__(
        self,
        *,
        env_cfg: Any,
        algo_cfg: Any,
        shared: dict[str, Any] | None = None,
        alpha_model: float = 0.5,
        seq_len: int = 4,
        curriculum_active: bool = False,
    ):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.shared = shared if shared is not None else {}
        self.alpha_model = float(getattr(algo_cfg, "value_expansion_alpha_model", alpha_model))
        self.seq_len = int(getattr(algo_cfg, "value_expansion_seq_len", seq_len))
        self.curriculum_active = bool(curriculum_active)
        self.value_expansion_model_warmup_ep = int(
            getattr(algo_cfg, "value_expansion_model_warmup_ep", 200)
        )
        self.value_expansion_ramp_start_ep = int(
            getattr(algo_cfg, "value_expansion_ramp_start_ep", 300)
        )
        self.value_expansion_ramp_end_ep = int(
            getattr(algo_cfg, "value_expansion_ramp_end_ep", 500)
        )
        self.value_expansion_alpha_model_max = float(
            getattr(algo_cfg, "value_expansion_alpha_model_max", self.alpha_model)
        )
        self.base_param_dim = int(specs.total_param_dim(env_cfg))

    def attach(self, *, trainer: Any, env_cfg: Any, algo_cfg: Any, n_envs: int) -> None:
        if not isinstance(trainer, QMIXTrainer):
            raise TypeError("value_expansion callback requires QMIXTrainer")
        super().attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=n_envs)
        device = getattr(trainer, "device", torch.device("cpu"))
        device = torch.device(device)
        n_agents = int(env_cfg.n_ch)
        agent_state_dim = int(specs.state_dim(env_cfg))
        n_actions = int(specs.action_dim(env_cfg))
        param_dim = int(specs.param_dim_per_action(env_cfg))
        global_state_dim = n_agents * agent_state_dim
        action_dim = exec_action_dim(
            n_agents=n_agents,
            n_des=int(specs.n_des(env_cfg)),
            n_channel=int(env_cfg.n_channel),
            param_dim=param_dim,
        )

        wm_cfg = self.shared.get("wm_cfg")
        if wm_cfg is None:
            wm_cfg = JointWorldModelConfig(
                state_dim=global_state_dim,
                action_dim=action_dim,
                hidden_dim=int(getattr(algo_cfg, "wm_hidden_dim", 256)),
                n_layers=int(getattr(algo_cfg, "wm_n_layers", 1)),
                stochastic_dim=int(getattr(algo_cfg, "wm_stochastic_dim", 32)),
                kl_beta=float(getattr(algo_cfg, "wm_kl_beta", 0.1)),
                free_nats=float(getattr(algo_cfg, "wm_free_nats", 1.0)),
            )
        world_model = self.shared.get("world_model")
        if world_model is None:
            world_model = JointWorldModel(wm_cfg).to(device)

        value_teacher = self.shared.get("value_teacher")
        if value_teacher is None:
            value_teacher = QMIXValueTeacher(
                trainer,
                MPDQNQMIXDims(
                    n_agents=n_agents,
                    agent_state_dim=agent_state_dim,
                    n_actions=n_actions,
                    param_dim=param_dim,
                ),
            )
            value_teacher.env_cfg = env_cfg

        replay = self.shared.get("wm_replay")
        if replay is None:
            replay = WorldModelSequenceReplayBuffer(
                n_envs=max(1, int(n_envs)),
                capacity=int(
                    getattr(
                        algo_cfg,
                        "wm_buffer_capacity",
                        getattr(algo_cfg, "buffer_capacity", 100_000),
                    )
                ),
            )

        td_cfg = self.shared.get("td_cfg")
        if td_cfg is None:
            td_cfg = TDlambdaConfig(
                gamma=float(algo_cfg.gamma),
                lam=float(getattr(algo_cfg, "value_expansion_td_lambda", 0.8)),
                rollout_k=int(getattr(algo_cfg, "value_expansion_rollout_k", 4)),
            )

        self.world_model = world_model
        self.value_teacher = value_teacher
        self.wm_replay = replay
        self.wm_cfg = wm_cfg
        self.td_cfg = td_cfg
        self.shared.update(
            {
                "world_model": world_model,
                "value_teacher": value_teacher,
                "wm_replay": replay,
                "wm_cfg": wm_cfg,
                "td_cfg": td_cfg,
            }
        )

    def on_transition_batch(
        self,
        *,
        states: np.ndarray,
        actions: Any,
        action_discrete: np.ndarray,
        action_params: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        infos: list[dict],
    ) -> None:
        del actions, infos
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        action_discrete = np.asarray(action_discrete, dtype=np.int64)
        action_params = np.asarray(action_params, dtype=np.float32)[..., : self.base_param_dim]
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        for env_id in range(int(states.shape[0])):
            reward_team = float(np.mean(rewards[env_id]))
            self.wm_replay.add(
                env_id=env_id,
                state=states[env_id].reshape(-1),
                action_discrete=action_discrete[env_id],
                action_params=action_params[env_id],
                reward_team=reward_team,
                next_state=next_states[env_id].reshape(-1),
                done=bool(dones[env_id]),
            )

    def _curriculum_fraction(self, episode: int) -> float:
        ep = int(episode)
        start = int(self.value_expansion_ramp_start_ep)
        end = int(self.value_expansion_ramp_end_ep)
        if ep < start:
            return 0.0
        if ep >= end:
            return 1.0
        return float(ep - start) / float(max(1, end - start))

    def _effective_alpha(self, episode: int) -> float:
        if not self.curriculum_active:
            return float(self.alpha_model)
        return float(self.value_expansion_alpha_model_max) * self._curriculum_fraction(episode)

    def should_skip_q_update(self, context: TrainHookContext) -> bool:
        if not self.curriculum_active:
            return False
        ep = int(context.episode)
        return (
            int(self.value_expansion_model_warmup_ep)
            <= ep
            < int(self.value_expansion_ramp_start_ep)
        )

    def _make_target_context(self, *, batch_size: int, episode: int) -> Optional[TDTargetContext]:
        alpha = self._effective_alpha(int(episode))
        if alpha <= 0.0:
            return None
        if self.wm_replay.count_ready_envs(seq_len=self.seq_len) <= 0:
            return None
        try:
            sample = self.wm_replay.sample_sequences(batch_size=batch_size, seq_len=self.seq_len)
        except (RuntimeError, ValueError):
            return None

        device = next(self.world_model.parameters()).device
        state_seq = torch.from_numpy(sample["state_seq"]).to(device)
        ad_seq = torch.from_numpy(sample["action_discrete_seq"]).long().to(device)
        ap_seq = torch.from_numpy(sample["action_params_seq"]).to(device)
        bsz, seq_len = int(ad_seq.shape[0]), int(ad_seq.shape[1])
        ad_flat = ad_seq.reshape(bsz * seq_len, int(self.env_cfg.n_ch))
        ap_flat = ap_seq.reshape(bsz * seq_len, int(self.env_cfg.n_ch), self.base_param_dim)
        action_enc = encode_joint_action_exec(
            ad_flat,
            ap_flat,
            n_agents=int(self.env_cfg.n_ch),
            n_channel=int(self.env_cfg.n_channel),
            n_des=int(specs.n_des(self.env_cfg)),
            n_actions=int(specs.action_dim(self.env_cfg)),
            param_dim=int(specs.param_dim_per_action(self.env_cfg)),
            power_min_dbm=float(self.env_cfg.uav_power_min),
            power_max_dbm=float(self.env_cfg.uav_power_max),
        ).reshape(bsz, seq_len, -1)
        return TDTargetContext(
            state_seq=state_seq,
            action_enc_seq=action_enc,
            world_model=self.world_model,
            value_teacher=self.value_teacher,
            td_cfg=self.td_cfg,
            alpha_model=float(alpha),
        )

    def on_train_step(self, context: TrainHookContext) -> Optional[dict[str, float]]:
        trainer = context.trainer
        if not hasattr(trainer, "train_step_from_batch"):
            return None
        if len(trainer.buffer) < trainer.batch_size:
            return None
        target_context = self._make_target_context(
            batch_size=int(trainer.batch_size),
            episode=int(context.episode),
        )
        if target_context is None:
            return None
        batch_np = trainer.buffer.sample(int(trainer.batch_size))
        batch = tensor_batch_from_numpy(batch_np, device=trainer.device)
        return trainer.train_step_from_batch(
            batch,
            target_context=target_context,
            hook_context=context,  # ★ Stage 8：透传 hook_context，让 trainer 调 _collect_aux_losses
        )

    def state_dict(self) -> dict:
        return {
            "alpha_model": float(self.alpha_model),
            "seq_len": int(self.seq_len),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        allowed = {"alpha_model", "seq_len"}
        if strict and (set(state) - allowed):
            raise ValueError(f"{self.name}: unexpected state keys {sorted(set(state) - allowed)}")
        if "alpha_model" in state:
            self.alpha_model = float(state["alpha_model"])
        if "seq_len" in state:
            self.seq_len = int(state["seq_len"])


__all__ = ["ValueExpansionCallback", "tensor_batch_from_numpy"]
