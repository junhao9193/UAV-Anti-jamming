"""Critic-stabilized variant of train_qmix_wm_alternating_jammer_prediction.

Identical to A's entry except:
  - Uses :class:`CriticStableJammerAwareMPDQNQMIXTrainer` (Polyak soft target
    updates + per-step uniform LR scaling).
  - Adds a per-episode LR decay schedule. Default: full LR for ep < 1500, then
    linearly anneal to ``--lr-decay-min`` over ``[1500, 3000)``. Disable via
    ``--no-lr-decay``.
  - Algorithm name suffix is ``_critic_stable`` so output directories don't clash.

Helpers (`_init_sensing_histories`, `_append_sensing_history`,
`_jammer_target_from_info`, `_jammer_targets_from_infos`) are imported from A's
entry so they cannot silently drift between the two variants.

Run from `UAV-Jammer-RL/`:
  python -m Main.train.train_qmix_wm_alternating_jammer_prediction_critic_stable
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
from tqdm.auto import trange

from envs import Environ
from Main.common import (
    SubprocVecEnv,
    env_run_config,
    make_fixed_p_trans,
    resolve_episode_steps,
    save_training_data,
)
from Main.train.train_qmix_wm_alternating import _linear_ramp
from Main.train.train_qmix_wm_alternating_jammer_prediction import (
    _append_sensing_history,
    _init_sensing_histories,
    _jammer_targets_from_infos,
)


def _lr_decay_scale(
    ep: int,
    *,
    start_ep: int,
    end_ep: int,
    floor: float,
    enabled: bool,
) -> float:
    """Per-episode multiplier on the optimizer LR.

    Returns 1.0 before ``start_ep``, then linearly anneals to ``floor`` over
    ``[start_ep, end_ep]``. After ``end_ep`` stays at ``floor``.
    Disabled (``enabled=False``) keeps a constant 1.0.
    """
    if not enabled:
        return 1.0
    if end_ep <= start_ep:
        return float(floor) if int(ep) >= int(start_ep) else 1.0
    if int(ep) < int(start_ep):
        return 1.0
    if int(ep) >= int(end_ep):
        return float(floor)
    frac = (float(ep) - float(start_ep)) / (float(end_ep) - float(start_ep))
    return float(1.0 + frac * (float(floor) - 1.0))


def train_qmix_wm_alternating_jammer_prediction_critic_stable(
    *,
    total_episodes: int = 500,
    qmix_block_episodes: int = 50,
    wm_block_episodes: int = 50,
    n_steps: int | None = None,
    num_envs: int = 32,
    batch_size: int = 256,
    buffer_capacity: int = 200_000,
    learn_every: int = 4,
    updates_per_learn: int = 1,
    lr_actor: float = 1e-3,
    lr_q: float = 1e-3,
    lr_jammer: float | None = None,
    lr_mixer: float | None = None,
    max_grad_norm: float = 10.0,
    use_amp: bool = True,
    device: str | None = None,
    start_method: str = "spawn",
    alpha_model: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.8,
    rollout_k: int = 4,
    seq_len: int = 8,
    wm_buffer_capacity: int = 500_000,
    wm_hidden_dim: int = 256,
    wm_n_layers: int = 1,
    wm_stochastic_dim: int = 32,
    wm_kl_beta: float = 0.1,
    wm_free_nats: float = 1.0,
    wm_max_grad_norm: float = 10.0,
    wm_lr: float = 3e-4,
    wm_batch_size: int = 512,
    wm_updates_per_learn: int = 2,
    wm_alpha_r: float = 1.0,
    wm_eta_vc: float = 0.2,
    wm_vc_warmup_ep: int = 300,
    wm_vc_ramp_end_ep: int = 800,
    jammer_history_len: int = 4,
    jammer_pred_hidden_dim: int = 64,
    jammer_aux_weight: float = 0.1,
    use_jammer_feature: bool = True,
    jammer_warmup_episodes: int = 200,
    target_tau: float = 0.005,
    lr_decay_enabled: bool = True,
    lr_decay_start_ep: int = 1500,
    lr_decay_end_ep: int = 3000,
    lr_decay_min: float = 0.1,
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    save_data: bool = True,
    seed: int = 0,
) -> None:
    import json
    import torch

    from algorithms.mpdqn.qmix.trainer_jammer_prediction_critic_stable import (
        CriticStableJammerAwareMPDQNQMIXTrainer,
        JammerAwareMPDQNQMIXDims,
        JammerAwareMPDQNQMIXValueTeacher,
        JammerAwareSequenceReplayBuffer,
    )
    from algorithms.world_model import JointWorldModelConfig, TDlambdaConfig
    from algorithms.world_model.action_encoding import exec_action_dim
    from algorithms.world_model.trainer import ValueConsistentWorldModelTrainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(use_amp) and str(device).startswith("cuda")
    if not (0.0 <= float(lr_decay_min) <= 1.0):
        raise ValueError(f"lr_decay_min must be in [0, 1], got {lr_decay_min}")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env0 = Environ()
    n_steps = resolve_episode_steps(env0, n_steps)
    p_trans_fixed = make_fixed_p_trans(env0)
    vecenv = SubprocVecEnv(
        int(num_envs),
        p_trans=p_trans_fixed,
        start_method=str(start_method),
        seed=int(seed),
    )

    qmix = CriticStableJammerAwareMPDQNQMIXTrainer(
        n_agents=int(env0.n_ch),
        state_dim=int(env0.state_dim),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        global_state_dim=int(env0.state_dim * env0.n_ch),
        n_channel=int(env0.n_channel),
        jammer_history_len=int(jammer_history_len),
        jammer_pred_hidden_dim=int(jammer_pred_hidden_dim),
        jammer_aux_weight=float(jammer_aux_weight),
        use_jammer_feature=bool(use_jammer_feature),
        buffer_capacity=int(buffer_capacity),
        batch_size=int(batch_size),
        gamma=float(gamma),
        lr_actor=float(lr_actor),
        lr_q=float(lr_q),
        lr_jammer=(None if lr_jammer is None else float(lr_jammer)),
        lr_mixer=(float(lr_mixer) if lr_mixer is not None else None),
        use_amp=use_amp,
        max_grad_norm=float(max_grad_norm),
        target_tau=float(target_tau),
        device=str(device),
    )
    for agent in qmix.agents:
        agent.actor.train()
        agent.q_net.train()
        agent.jammer_predictor.train()
    qmix.mixer.train()

    n_agents = int(env0.n_ch)
    agent_state_dim = int(env0.state_dim)
    global_state_dim = int(n_agents * agent_state_dim)
    action_exec_dim = exec_action_dim(
        n_agents=int(n_agents),
        n_des=int(env0.n_des),
        n_channel=int(env0.n_channel),
        param_dim=int(env0.param_dim_per_action),
    )

    wm_cfg = JointWorldModelConfig(
        state_dim=int(global_state_dim),
        action_dim=int(action_exec_dim),
        hidden_dim=int(wm_hidden_dim),
        n_layers=int(wm_n_layers),
        stochastic_dim=int(wm_stochastic_dim),
        kl_beta=float(wm_kl_beta),
        free_nats=float(wm_free_nats),
    )

    td_cfg = TDlambdaConfig(gamma=float(gamma), lam=float(lam), rollout_k=int(rollout_k))
    wm_trainer = ValueConsistentWorldModelTrainer(
        wm_cfg=wm_cfg,
        n_agents=int(n_agents),
        n_channel=int(env0.n_channel),
        n_des=int(env0.n_des),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        alpha=float(wm_alpha_r),
        eta=0.0,
        td_cfg=td_cfg,
        lr=float(wm_lr),
        max_grad_norm=float(wm_max_grad_norm),
        power_min_dbm=float(env0.uav_power_min),
        power_max_dbm=float(env0.uav_power_max),
        device=str(device),
    )

    value_teacher = JammerAwareMPDQNQMIXValueTeacher(
        qmix,
        JammerAwareMPDQNQMIXDims(
            n_agents=int(n_agents),
            agent_state_dim=int(agent_state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
            n_channel=int(env0.n_channel),
            jammer_history_len=int(jammer_history_len),
        ),
    )

    seq_buffer = JammerAwareSequenceReplayBuffer(n_envs=int(num_envs), capacity=int(wm_buffer_capacity))
    epsilon = float(epsilon_start)

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []
    phase_history = []
    wm_loss_total_history = []
    wm_loss_vc_history = []
    qmix_loss_q_history = []
    qmix_loss_jammer_history = []

    block_len = int(qmix_block_episodes) + int(wm_block_episodes)
    if block_len <= 0:
        raise ValueError("qmix_block_episodes + wm_block_episodes must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm = "mpdqn_qmix_wm_alternating_ve_jammer_pred_critic_stable"

    pbar = trange(int(total_episodes), desc="Training(QMIX<->WM-VE+JP+CS)", unit="ep", ascii=True)
    try:
        for ep in pbar:
            pos_in_block = int(ep) % int(block_len)
            train_qmix = pos_in_block < int(qmix_block_episodes)
            train_wm = not bool(train_qmix)
            phase = "qmix" if bool(train_qmix) else "wm"
            phase_history.append(phase)
            wm_eta_now = _linear_ramp(
                int(ep),
                t0=int(wm_vc_warmup_ep),
                t1=int(wm_vc_ramp_end_ep),
                v_max=float(wm_eta_vc),
            )
            wm_trainer.eta = float(wm_eta_now)

            feature_scale_now = _linear_ramp(
                int(ep),
                t0=0,
                t1=int(jammer_warmup_episodes),
                v_max=1.0,
            )
            qmix.set_feature_scale(float(feature_scale_now))

            lr_scale_now = _lr_decay_scale(
                int(ep),
                start_ep=int(lr_decay_start_ep),
                end_ep=int(lr_decay_end_ep),
                floor=float(lr_decay_min),
                enabled=bool(lr_decay_enabled),
            )
            qmix.set_lr_scale(float(lr_scale_now))

            if bool(train_qmix):
                wm_trainer.wm.eval()
                for agent in qmix.agents:
                    agent.actor.train()
                    agent.q_net.train()
                    agent.jammer_predictor.train()
                qmix.mixer.train()
            else:
                for agent in qmix.agents:
                    agent.actor.eval()
                    agent.q_net.eval()
                    agent.jammer_predictor.eval()
                qmix.mixer.eval()
                wm_trainer.wm.train()

            states = vecenv.reset()
            sensing_histories = _init_sensing_histories(
                states,
                history_len=int(jammer_history_len),
                n_channel=int(env0.n_channel),
            )
            global_states = states.reshape(int(num_envs), -1).astype(np.float32)

            episode_reward = 0.0
            steps_done = 0
            q_loss_q_sum = 0.0
            q_loss_jammer_sum = 0.0
            q_loss_count = 0
            wm_loss_sum = 0.0
            wm_vc_sum = 0.0
            wm_loss_count = 0

            for step in range(int(n_steps)):
                action_discrete_all = np.zeros((int(num_envs), int(n_agents)), dtype=np.int32)
                action_params_all = np.zeros((int(num_envs), int(n_agents), int(env0.total_param_dim)), dtype=np.float32)

                for i in range(int(n_agents)):
                    ad, ap = qmix.agents[i].select_action_batch(
                        states[:, i, :],
                        float(epsilon),
                        sensing_histories[:, i, :, :],
                    )
                    action_discrete_all[:, i] = ad
                    action_params_all[:, i, :] = ap

                actions = [
                    [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(int(n_agents))]
                    for e in range(int(num_envs))
                ]

                vecenv.step_async(actions)

                if (step + 1) % int(max(1, learn_every)) == 0:
                    if bool(train_qmix):
                        for _ in range(int(max(1, updates_per_learn))):
                            loss_info = qmix.train_step_value_expansion(
                                seq_buffer=seq_buffer,
                                seq_len=int(seq_len),
                                world_model=wm_trainer.wm,
                                value_teacher=value_teacher,
                                td_cfg=td_cfg,
                                alpha_model=float(alpha_model),
                                n_channel=int(env0.n_channel),
                                n_des=int(env0.n_des),
                                power_min_dbm=float(env0.uav_power_min),
                                power_max_dbm=float(env0.uav_power_max),
                            )
                            if loss_info is None:
                                loss_info = qmix.train_step()
                            if loss_info is not None:
                                q_loss_q_sum += float(loss_info.get("loss_q", 0.0))
                                q_loss_jammer_sum += float(loss_info.get("loss_jammer", 0.0))
                                q_loss_count += 1

                    if bool(train_wm):
                        for _ in range(int(max(1, wm_updates_per_learn))):
                            try:
                                batch = seq_buffer.sample_sequences(batch_size=int(wm_batch_size), seq_len=int(seq_len))
                            except Exception:
                                break
                            losses, _ = wm_trainer.train_step(
                                state_seq=torch.from_numpy(batch["state_seq"]),
                                action_discrete_seq=torch.from_numpy(batch["action_discrete_seq"]),
                                action_params_seq=torch.from_numpy(batch["action_params_seq"]),
                                reward_seq=torch.from_numpy(batch["reward_seq"]),
                                next_state_seq=torch.from_numpy(batch["next_state_seq"]),
                                value_teacher=(value_teacher if float(wm_eta_now) > 0.0 else None),
                            )
                            wm_loss_sum += float(losses.loss_total)
                            wm_vc_sum += float(losses.loss_vc)
                            wm_loss_count += 1

                next_states, rewards, dones, infos = vecenv.step_wait()
                next_global_states = next_states.reshape(int(num_envs), -1).astype(np.float32)
                reward_team = np.mean(np.asarray(rewards, dtype=np.float32), axis=1)
                jammer_targets = _jammer_targets_from_infos(infos, n_channel=int(env0.n_channel))
                next_sensing_histories = _append_sensing_history(
                    sensing_histories,
                    next_states,
                    n_channel=int(env0.n_channel),
                )

                is_last_step = step == int(n_steps) - 1
                for e in range(int(num_envs)):
                    done_e = bool(dones[e]) or bool(is_last_step)
                    qmix.store_transition(
                        states=states[e],
                        actions=actions[e],
                        rewards=np.asarray(rewards[e], dtype=np.float32),
                        next_states=next_states[e],
                        done=bool(done_e),
                        jammer_target=jammer_targets[e],
                        sensing_histories=sensing_histories[e],
                        next_sensing_histories=next_sensing_histories[e],
                    )

                    seq_buffer.add(
                        env_id=int(e),
                        state=global_states[e],
                        action_discrete=action_discrete_all[e],
                        action_params=action_params_all[e],
                        reward_team=float(reward_team[e]),
                        next_state=next_global_states[e],
                        done=bool(done_e),
                        jammer_target=jammer_targets[e],
                    )

                states = next_states
                sensing_histories = next_sensing_histories
                global_states = next_global_states
                episode_reward += float(np.mean(rewards))
                steps_done += 1
                if bool(np.any(dones)):
                    break

            steps_done = max(1, int(steps_done))
            total_links = float(steps_done * int(num_envs) * int(n_agents) * int(env0.n_des))
            energy_arr, jump_arr, suc_arr = vecenv.get_metrics()
            avg_energy = float(np.sum(energy_arr) / (total_links + 1e-12))
            avg_jump = float(np.sum(jump_arr) / (total_links + 1e-12))
            avg_suc_per_link = float(np.sum(suc_arr) / (total_links + 1e-12))
            success_rate = float(np.clip((avg_suc_per_link + 3.0) / 4.0, 0.0, 1.0))

            if epsilon > float(epsilon_min):
                epsilon *= float(epsilon_decay)

            reward_history.append(float(episode_reward))
            success_rate_history.append(float(success_rate))
            energy_history.append(float(avg_energy))
            jump_history.append(float(avg_jump))
            qmix_loss_q_history.append(float(q_loss_q_sum / max(1, q_loss_count)))
            qmix_loss_jammer_history.append(float(q_loss_jammer_sum / max(1, q_loss_count)))
            wm_loss_total_history.append(float(wm_loss_sum / max(1, wm_loss_count)))
            wm_loss_vc_history.append(float(wm_vc_sum / max(1, wm_loss_count)))

            pbar.set_postfix(
                phase=phase,
                avg_r=float(np.mean(reward_history[-10:])),
                sr=float(success_rate_history[-1]),
                eps=float(epsilon),
                alpha=float(alpha_model),
                eta=float(wm_eta_now),
                fs=float(feature_scale_now),
                lr=float(lr_scale_now),
                qloss=float(qmix_loss_q_history[-1]),
                jloss=float(qmix_loss_jammer_history[-1]),
                wm=float(wm_loss_total_history[-1]),
                lvc=float(wm_loss_vc_history[-1]),
                buf=int(len(seq_buffer)),
            )
    finally:
        vecenv.close()

    if not bool(save_data):
        return

    _, _, out_dir = save_training_data(
        algorithm=algorithm,
        reward_history=reward_history,
        success_rate_history=success_rate_history,
        energy_history=energy_history,
        jump_history=jump_history,
        n_episode=int(total_episodes),
        n_steps=int(n_steps),
        trainer=qmix,
        run_config={
            "algorithm": algorithm,
            "seed": int(seed),
            "num_envs": int(num_envs),
            "batch_size": int(batch_size),
            "buffer_capacity": int(buffer_capacity),
            "learn_every": int(learn_every),
            "updates_per_learn": int(updates_per_learn),
            "lr_actor": float(lr_actor),
            "lr_q": float(lr_q),
            "lr_jammer": None if lr_jammer is None else float(lr_jammer),
            "lr_mixer": None if lr_mixer is None else float(lr_mixer),
            "max_grad_norm": float(max_grad_norm),
            "use_amp": bool(use_amp),
            "device": str(device),
            "start_method": str(start_method),
            "qmix_block_episodes": int(qmix_block_episodes),
            "wm_block_episodes": int(wm_block_episodes),
            "alpha_model": float(alpha_model),
            "gamma": float(gamma),
            "lam": float(lam),
            "rollout_k": int(rollout_k),
            "seq_len": int(seq_len),
            "wm_buffer_capacity": int(wm_buffer_capacity),
            "wm_hidden_dim": int(wm_hidden_dim),
            "wm_n_layers": int(wm_n_layers),
            "wm_stochastic_dim": int(wm_stochastic_dim),
            "wm_kl_beta": float(wm_kl_beta),
            "wm_free_nats": float(wm_free_nats),
            "wm_lr": float(wm_lr),
            "wm_max_grad_norm": float(wm_max_grad_norm),
            "wm_batch_size": int(wm_batch_size),
            "wm_updates_per_learn": int(wm_updates_per_learn),
            "wm_alpha_r": float(wm_alpha_r),
            "wm_eta_vc": float(wm_eta_vc),
            "wm_vc_warmup_ep": int(wm_vc_warmup_ep),
            "wm_vc_ramp_end_ep": int(wm_vc_ramp_end_ep),
            "jammer_history_len": int(jammer_history_len),
            "jammer_pred_hidden_dim": int(jammer_pred_hidden_dim),
            "jammer_aux_weight": float(jammer_aux_weight),
            "use_jammer_feature": bool(use_jammer_feature),
            "jammer_warmup_episodes": int(jammer_warmup_episodes),
            "target_tau": float(target_tau),
            "lr_decay_enabled": bool(lr_decay_enabled),
            "lr_decay_start_ep": int(lr_decay_start_ep),
            "lr_decay_end_ep": int(lr_decay_end_ep),
            "lr_decay_min": float(lr_decay_min),
            "epsilon_start": float(epsilon_start),
            "epsilon_min": float(epsilon_min),
            "epsilon_decay": float(epsilon_decay),
            **env_run_config(env0),
        },
    )

    torch.save(
        {
            "wm_cfg": {
                "state_dim": int(wm_cfg.state_dim),
                "action_dim": int(wm_cfg.action_dim),
                "hidden_dim": int(wm_cfg.hidden_dim),
                "n_layers": int(wm_cfg.n_layers),
                "stochastic_dim": int(wm_cfg.stochastic_dim),
                "kl_beta": float(wm_cfg.kl_beta),
                "free_nats": float(wm_cfg.free_nats),
            },
            "td_cfg": {"gamma": float(gamma), "lam": float(lam), "rollout_k": int(rollout_k)},
            "wm_state_dict": wm_trainer.wm.state_dict(),
            "opt_state_dict": wm_trainer.opt.state_dict(),
            "init_ckpt_meta": {"initialized_from_scratch": True},
        },
        str(out_dir / "world_model_weights.pth"),
    )

    (out_dir / "alternating_config.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "qmix_initialized_from_scratch": True,
                "world_model_initialized_from_scratch": True,
                "total_episodes": int(total_episodes),
                "qmix_block_episodes": int(qmix_block_episodes),
                "wm_block_episodes": int(wm_block_episodes),
                "alpha_model": float(alpha_model),
                "wm_eta_vc": float(wm_eta_vc),
                "wm_vc_warmup_ep": int(wm_vc_warmup_ep),
                "wm_vc_ramp_end_ep": int(wm_vc_ramp_end_ep),
                "wm_max_grad_norm": float(wm_max_grad_norm),
                "td_cfg": {"gamma": float(gamma), "lam": float(lam), "rollout_k": int(rollout_k)},
                "seq_len": int(seq_len),
                "num_envs": int(num_envs),
                "jammer_history_len": int(jammer_history_len),
                "jammer_pred_hidden_dim": int(jammer_pred_hidden_dim),
                "jammer_aux_weight": float(jammer_aux_weight),
                "use_jammer_feature": bool(use_jammer_feature),
                "jammer_warmup_episodes": int(jammer_warmup_episodes),
                "target_tau": float(target_tau),
                "lr_decay_enabled": bool(lr_decay_enabled),
                "lr_decay_start_ep": int(lr_decay_start_ep),
                "lr_decay_end_ep": int(lr_decay_end_ep),
                "lr_decay_min": float(lr_decay_min),
                "wm_cfg": {
                    "state_dim": int(wm_cfg.state_dim),
                    "action_dim": int(wm_cfg.action_dim),
                    "hidden_dim": int(wm_cfg.hidden_dim),
                    "n_layers": int(wm_cfg.n_layers),
                    "stochastic_dim": int(wm_cfg.stochastic_dim),
                    "kl_beta": float(wm_cfg.kl_beta),
                    "free_nats": float(wm_cfg.free_nats),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "alternating_metrics.json").write_text(
        json.dumps(
            {
                "phase": phase_history,
                "qmix_loss_q": qmix_loss_q_history,
                "qmix_loss_jammer": qmix_loss_jammer_history,
                "wm_loss_total": wm_loss_total_history,
                "wm_loss_vc": wm_loss_vc_history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QMIX+WM with jammer prediction + critic stabilization (Polyak + LR decay)"
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--qmix-block", type=int, default=50)
    parser.add_argument("--wm-block", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--updates-per-learn", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-q", type=float, default=1e-3)
    parser.add_argument("--lr-jammer", type=float, default=None)
    parser.add_argument("--lr-mixer", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-method", type=str, default="spawn")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--alpha-model", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--rollout-k", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--wm-buffer-capacity", type=int, default=500_000)
    parser.add_argument("--wm-hidden-dim", type=int, default=256)
    parser.add_argument("--wm-n-layers", type=int, default=1)
    parser.add_argument("--wm-stochastic-dim", type=int, default=32)
    parser.add_argument("--wm-kl-beta", type=float, default=0.1)
    parser.add_argument("--wm-free-nats", type=float, default=1.0)
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--wm-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--wm-batch-size", type=int, default=512)
    parser.add_argument("--wm-updates-per-learn", type=int, default=2)
    parser.add_argument("--wm-alpha-r", type=float, default=1.0)
    parser.add_argument("--wm-eta-vc", type=float, default=0.2)
    parser.add_argument("--wm-vc-warmup-ep", type=int, default=300)
    parser.add_argument("--wm-vc-ramp-end-ep", type=int, default=800)
    parser.add_argument("--jammer-history-len", type=int, default=4)
    parser.add_argument("--jammer-pred-hidden-dim", type=int, default=64)
    parser.add_argument("--jammer-aux-weight", type=float, default=0.1)
    parser.add_argument("--no-jammer-feature", action="store_true")
    parser.add_argument("--jammer-warmup-episodes", type=int, default=200)
    parser.add_argument("--target-tau", type=float, default=0.005,
                        help="Polyak soft-update coefficient applied every learn step (default 0.005)")
    parser.add_argument("--lr-decay-start-ep", type=int, default=1500,
                        help="Episode at which LR begins linearly annealing toward --lr-decay-min")
    parser.add_argument("--lr-decay-end-ep", type=int, default=3000,
                        help="Episode at which LR reaches --lr-decay-min and stays there")
    parser.add_argument("--lr-decay-min", type=float, default=0.1,
                        help="Final LR multiplier relative to initial (default 0.1 = 10x decay)")
    parser.add_argument("--no-lr-decay", action="store_true", help="Disable LR decay entirely")
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    train_qmix_wm_alternating_jammer_prediction_critic_stable(
        total_episodes=int(args.episodes),
        qmix_block_episodes=int(args.qmix_block),
        wm_block_episodes=int(args.wm_block),
        n_steps=args.steps,
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        learn_every=int(args.learn_every),
        updates_per_learn=int(args.updates_per_learn),
        lr_actor=float(args.lr_actor),
        lr_q=float(args.lr_q),
        lr_jammer=args.lr_jammer,
        lr_mixer=args.lr_mixer,
        max_grad_norm=float(args.max_grad_norm),
        use_amp=not bool(args.no_amp),
        device=args.device,
        start_method=str(args.start_method),
        alpha_model=float(args.alpha_model),
        gamma=float(args.gamma),
        lam=float(args.lam),
        rollout_k=int(args.rollout_k),
        seq_len=int(args.seq_len),
        wm_buffer_capacity=int(args.wm_buffer_capacity),
        wm_hidden_dim=int(args.wm_hidden_dim),
        wm_n_layers=int(args.wm_n_layers),
        wm_stochastic_dim=int(args.wm_stochastic_dim),
        wm_kl_beta=float(args.wm_kl_beta),
        wm_free_nats=float(args.wm_free_nats),
        wm_max_grad_norm=float(args.wm_max_grad_norm),
        wm_lr=float(args.wm_lr),
        wm_batch_size=int(args.wm_batch_size),
        wm_updates_per_learn=int(args.wm_updates_per_learn),
        wm_alpha_r=float(args.wm_alpha_r),
        wm_eta_vc=float(args.wm_eta_vc),
        wm_vc_warmup_ep=int(args.wm_vc_warmup_ep),
        wm_vc_ramp_end_ep=int(args.wm_vc_ramp_end_ep),
        jammer_history_len=int(args.jammer_history_len),
        jammer_pred_hidden_dim=int(args.jammer_pred_hidden_dim),
        jammer_aux_weight=float(args.jammer_aux_weight),
        use_jammer_feature=not bool(args.no_jammer_feature),
        jammer_warmup_episodes=int(args.jammer_warmup_episodes),
        target_tau=float(args.target_tau),
        lr_decay_enabled=not bool(args.no_lr_decay),
        lr_decay_start_ep=int(args.lr_decay_start_ep),
        lr_decay_end_ep=int(args.lr_decay_end_ep),
        lr_decay_min=float(args.lr_decay_min),
        epsilon_start=float(args.epsilon_start),
        epsilon_min=float(args.epsilon_min),
        epsilon_decay=float(args.epsilon_decay),
        save_data=not bool(args.no_save),
        seed=int(args.seed),
    )
