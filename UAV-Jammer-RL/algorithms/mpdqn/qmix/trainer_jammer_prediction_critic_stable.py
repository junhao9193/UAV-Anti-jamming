"""Critic-stabilized variant of trainer_jammer_prediction.

Adds two mechanisms on top of A:
  1. Polyak (soft) target updates applied every learn step, replacing the hard
     copy every `target_update_interval` steps. The hard-copy pattern was the
     dominant source of late-stage critic drift in A's seed-0 run (qloss 2.5 -> 4.2
     by ep 2000), since each target sync injects a discontinuous jump that the
     online Q-net then chases.
  2. set_lr_scale(scale): uniform multiplicative scale on every optimizer's lr,
     relative to the lr the optimizer was constructed with. Training entry drives
     this with a per-episode linear ramp (default 1.0 from ep 0..1500, decay to
     0.1 by ep 3000), which freezes the Q/actor params toward the end of training
     and lets the replay distribution catch up.

All other A semantics (jammer prediction head, BCE loss, feature_scale warmup,
VE/VC/BA framework, JointReplayBuffer schema) are inherited unchanged from
:class:`JammerAwareMPDQNQMIXTrainer`. The accompanying value teacher and dims
classes also reuse A's directly — the MLP target_q_net has no cudnn restriction
so the parent's behavior is correct here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from algorithms.mpdqn.qmix.trainer_jammer_prediction import (
    JammerAwareJointReplayBuffer,
    JammerAwareMPDQNAgent,
    JammerAwareMPDQNQMIXDims,
    JammerAwareMPDQNQMIXTrainer,
    JammerAwareMPDQNQMIXValueTeacher,
    JammerAwareSequenceReplayBuffer,
    JammerPredictionHead,
)


class CriticStableJammerAwareMPDQNQMIXTrainer(JammerAwareMPDQNQMIXTrainer):
    """JammerAware QMIX trainer with Polyak soft target updates + uniform LR scale.

    Constructor accepts one new kwarg `target_tau` (default 0.005); all other
    arguments forwarded unchanged to the parent.
    """

    def __init__(self, *args, target_tau: float = 0.005, **kwargs):
        super().__init__(*args, **kwargs)
        tau = float(target_tau)
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"target_tau must be in (0, 1], got {tau}")
        self.target_tau = tau

        # Snapshot the initial learning rate of every optimizer so set_lr_scale
        # can multiply against the original value rather than the current value
        # (which would compound across calls).
        self._lr_init: dict[str, float] = {}
        for i, agent in enumerate(self.agents):
            self._lr_init[f"actor_{i}"] = float(agent.actor_opt.param_groups[0]["lr"])
            self._lr_init[f"q_{i}"] = float(agent.q_opt.param_groups[0]["lr"])
            self._lr_init[f"jammer_{i}"] = float(agent.jammer_predictor_opt.param_groups[0]["lr"])
        self._lr_init["mixer"] = float(self.mixer_opt.param_groups[0]["lr"])

        self._lr_scale = 1.0

    def _target_update_if_needed(self) -> None:
        """Polyak soft update applied every learn step.

        Ignores `target_update_interval` entirely; the field remains on the parent
        for checkpoint-config compatibility but no longer gates updates.
        """
        self.learn_steps += 1
        tau = float(self.target_tau)
        one_minus_tau = 1.0 - tau
        with torch.no_grad():
            for agent in self.agents:
                for p, pt in zip(agent.actor.parameters(), agent.target_actor.parameters()):
                    pt.data.mul_(one_minus_tau).add_(p.data, alpha=tau)
                for p, pt in zip(agent.q_net.parameters(), agent.target_q_net.parameters()):
                    pt.data.mul_(one_minus_tau).add_(p.data, alpha=tau)
                for p, pt in zip(
                    agent.jammer_predictor.parameters(),
                    agent.target_jammer_predictor.parameters(),
                ):
                    pt.data.mul_(one_minus_tau).add_(p.data, alpha=tau)
            for p, pt in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                pt.data.mul_(one_minus_tau).add_(p.data, alpha=tau)

    def set_lr_scale(self, scale: float) -> None:
        """Uniformly scale all QMIX-side optimizer LRs relative to their init values.

        Does not touch the world-model trainer's optimizer (that one is owned by
        ``ValueConsistentWorldModelTrainer`` and managed separately).
        """
        s = float(np.clip(float(scale), 0.0, 1.0))
        self._lr_scale = s
        for i, agent in enumerate(self.agents):
            for pg in agent.actor_opt.param_groups:
                pg["lr"] = self._lr_init[f"actor_{i}"] * s
            for pg in agent.q_opt.param_groups:
                pg["lr"] = self._lr_init[f"q_{i}"] * s
            for pg in agent.jammer_predictor_opt.param_groups:
                pg["lr"] = self._lr_init[f"jammer_{i}"] * s
        for pg in self.mixer_opt.param_groups:
            pg["lr"] = self._lr_init["mixer"] * s


__all__ = [
    "CriticStableJammerAwareMPDQNQMIXTrainer",
    # Re-export A's helpers so the training entry has a single import line.
    "JammerAwareJointReplayBuffer",
    "JammerAwareSequenceReplayBuffer",
    "JammerAwareMPDQNAgent",
    "JammerAwareMPDQNQMIXDims",
    "JammerAwareMPDQNQMIXValueTeacher",
    "JammerPredictionHead",
]
