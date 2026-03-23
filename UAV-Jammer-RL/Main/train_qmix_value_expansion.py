"""
MP-DQN (QMIX) + Value Expansion with a Value-Consistent RSSM World Model.

Implements the alternating loop described in `doc/价值一致性世界模型.md`:
  Step 0: collect real data with QMIX behavior policy (epsilon-greedy)
  Step 1: update critic with mixed target y = (1-α) y_real + α y_model
  Step 2: periodic target network update (handled inside QMIX trainer)
  Step 3: update world model with L_WM = L_S + alpha_r L_R + η L_VC
  Step 4: curriculum schedule for α(t), η(t)

Run from `UAV-Jammer-RL/`:
  python -m Main.train_qmix_value_expansion
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from envs import Environ
from Main.common import SubprocVecEnv, get_repo_root, make_fixed_p_trans, save_training_data
from tqdm.auto import trange


def _linear_ramp(t: int, *, t0: int, t1: int, v_max: float) -> float:
    if t1 <= t0:
        return float(v_max) if int(t) >= int(t0) else 0.0
    frac = (float(t) - float(t0)) / (float(t1) - float(t0))
    return float(v_max) * float(np.clip(frac, 0.0, 1.0))


def _save_world_model_checkpoint(
    *,
    out_dir: Path,
    wm_trainer,
    wm_cfg: dict,
    td_cfg: dict,
    metrics: dict,
) -> None:
    import json
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "world_model_config.json").write_text(
        json.dumps({"wm_cfg": wm_cfg, "td_cfg": td_cfg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "world_model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    torch.save(
        {
            "wm_cfg": wm_cfg,
            "td_cfg": td_cfg,
            "wm_state_dict": wm_trainer.wm.state_dict(),
            "opt_state_dict": wm_trainer.opt.state_dict(),
        },
        str(out_dir / "world_model_weights.pth"),
    )


def train_qmix_value_expansion(
    *,
    n_episode: int = 3000,
    n_steps: int = 1000,
    num_envs: int = 32,
    batch_size: int = 256,
    buffer_capacity: int = 200_000,
    learn_every: int = 4,
    updates_per_learn: int = 1,
    lr_actor: float = 1e-3,
    lr_q: float = 1e-3,
    lr_mixer: float | None = None,
    max_grad_norm: float = 10.0,
    use_amp: bool = True,
    device: str | None = None,
    start_method: str = "spawn",
    save_data: bool = True,
    # Sequence replay for WM/VE
    seq_len: int = 8,
    wm_buffer_capacity: int = 500_000,
    # World model config
    wm_hidden_dim: int = 256,
    wm_n_layers: int = 1,
    wm_stochastic_dim: int = 32,
    wm_kl_beta: float = 0.1,
    wm_free_nats: float = 1.0,
    wm_lr: float = 1e-3,
    wm_batch_size: int = 512,
    wm_updates_per_learn: int = 2,
    # TD(lambda)
    gamma: float = 0.99,
    lam: float = 0.8,
    rollout_k: int = 4,
    # Curriculum schedule
    critic_warmup_ep: int = 200,
    model_warmup_ep: int = 200,
    ramp_start_ep: int = 300,
    ramp_end_ep: int = 1000,
    alpha_model_max: float = 0.01,
    eta_max: float = 0.2,
    seed: int = 0,
) -> Tuple[object, dict]:
    import torch

    from algorithms.mpdqn.qmix.trainer_greedy_actor import MPDQNQMIXTrainer
    from algorithms.world_model import JointWorldModelConfig, MPDQNQMIXDims, MPDQNQMIXValueTeacher, TDlambdaConfig
    from algorithms.world_model.action_encoding import exec_action_dim
    from algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer
    from algorithms.world_model.trainer import ValueConsistentWorldModelTrainer

    def _configure_torch(dev: str) -> None:
        if dev.startswith("cuda"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")

    env0 = Environ()
    p_trans_fixed = make_fixed_p_trans(env0)
    vecenv = SubprocVecEnv(
        int(num_envs),
        p_trans=p_trans_fixed,
        start_method=str(start_method),
        seed=int(seed),
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_torch(str(device))
    use_amp = bool(use_amp) and str(device).startswith("cuda")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # --- QMIX trainer ---
    qmix = MPDQNQMIXTrainer(
        n_agents=int(env0.n_ch),
        state_dim=int(env0.state_dim),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        global_state_dim=int(env0.state_dim * env0.n_ch),
        buffer_capacity=int(buffer_capacity),
        batch_size=int(batch_size),
        gamma=float(gamma),
        lr_actor=float(lr_actor),
        lr_q=float(lr_q),
        lr_mixer=(float(lr_mixer) if lr_mixer is not None else None),
        use_amp=use_amp,
        max_grad_norm=float(max_grad_norm),
        device=str(device),
    )

    # --- World model trainer + sequence replay ---
    n_envs = int(num_envs)
    n_agents = int(env0.n_ch)
    agent_state_dim = int(env0.state_dim)
    global_state_dim = int(n_agents * agent_state_dim)
    action_dim_exec = exec_action_dim(
        n_agents=int(n_agents),
        n_des=int(env0.n_des),
        n_channel=int(env0.n_channel),
        param_dim=int(env0.param_dim_per_action),
    )

    wm_cfg_obj = JointWorldModelConfig(
        state_dim=int(global_state_dim),
        action_dim=int(action_dim_exec),
        hidden_dim=int(wm_hidden_dim),
        n_layers=int(wm_n_layers),
        stochastic_dim=int(wm_stochastic_dim),
        kl_beta=float(wm_kl_beta),
        free_nats=float(wm_free_nats),
    )
    td_cfg_obj = TDlambdaConfig(gamma=float(gamma), lam=float(lam), rollout_k=int(rollout_k))

    wm_trainer = ValueConsistentWorldModelTrainer(
        wm_cfg=wm_cfg_obj,
        n_agents=int(n_agents),
        n_channel=int(env0.n_channel),
        n_des=int(env0.n_des),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        alpha=1.0,
        eta=0.0,  # scheduled
        td_cfg=td_cfg_obj,
        lr=float(wm_lr),
        power_min_dbm=float(env0.uav_power_min),
        power_max_dbm=float(env0.uav_power_max),
        device=str(device),
    )

    value_teacher = MPDQNQMIXValueTeacher(
        qmix,
        MPDQNQMIXDims(
            n_agents=int(n_agents),
            agent_state_dim=int(agent_state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
        ),
    )

    seq_buffer = WorldModelSequenceReplayBuffer(n_envs=int(n_envs), capacity=int(wm_buffer_capacity))

    # Epsilon schedule for behavior policy.
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []

    wm_loss_total_history = []
    alpha_history = []
    eta_history = []

    pbar = trange(n_episode, desc="Training(QMIX+WMVE-RSSM)", unit="ep", ascii=True)
    try:
        for episode in pbar:
            # Training phases (3-phase curriculum):
            #  - [0, critic_warmup_ep): train QMIX with real TD only (alpha_model=0), do not train WM
            #  - [model_warmup_ep, ramp_start_ep): freeze QMIX params, train WM with supervised loss only (eta=0)
            #  - [ramp_start_ep, ...): alternate updating QMIX and WM; enable alpha/eta ramp
            train_qmix = int(episode) < int(critic_warmup_ep) or int(episode) >= int(ramp_start_ep)
            train_wm = int(episode) >= int(model_warmup_ep)

            alpha_model = (
                0.0
                if int(episode) < int(ramp_start_ep)
                else _linear_ramp(int(episode), t0=int(ramp_start_ep), t1=int(ramp_end_ep), v_max=float(alpha_model_max))
            )
            eta = (
                0.0
                if int(episode) < int(ramp_start_ep)
                else _linear_ramp(int(episode), t0=int(ramp_start_ep), t1=int(ramp_end_ep), v_max=float(eta_max))
            )
            wm_trainer.eta = float(eta)
            alpha_history.append(float(alpha_model))
            eta_history.append(float(eta))

            states = vecenv.reset()  # (E,N,S)
            global_states = states.reshape(n_envs, -1).astype(np.float32)

            episode_reward = 0.0
            loss_q_sum = 0.0
            loss_actor_sum = 0.0
            loss_count = 0
            wm_loss_sum = 0.0
            wm_loss_count = 0

            for step in range(int(n_steps)):
                action_discrete_all = np.zeros((n_envs, n_agents), dtype=np.int32)
                action_params_all = np.zeros((n_envs, n_agents, int(env0.total_param_dim)), dtype=np.float32)

                for i in range(n_agents):
                    ad, ap = qmix.agents[i].select_action_batch(states[:, i, :], float(epsilon))
                    action_discrete_all[:, i] = ad
                    action_params_all[:, i, :] = ap

                actions = [
                    [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(n_agents)]
                    for e in range(n_envs)
                ]

                vecenv.step_async(actions)

                # --- Updates while env workers simulate ---
                if (step + 1) % int(max(1, learn_every)) == 0:
                    if bool(train_qmix):
                        for _ in range(int(max(1, updates_per_learn))):
                            if float(alpha_model) > 0.0:
                                loss_info = qmix.train_step_value_expansion(
                                    seq_buffer=seq_buffer,
                                    seq_len=int(seq_len),
                                    world_model=wm_trainer.wm,
                                    value_teacher=value_teacher,
                                    td_cfg=td_cfg_obj,
                                    alpha_model=float(alpha_model),
                                    n_channel=int(env0.n_channel),
                                    n_des=int(env0.n_des),
                                    power_min_dbm=float(env0.uav_power_min),
                                    power_max_dbm=float(env0.uav_power_max),
                                )
                                if loss_info is None:
                                    loss_info = qmix.train_step()
                            else:
                                loss_info = qmix.train_step()

                            if loss_info is not None:
                                loss_q_sum += float(loss_info["loss_q"])
                                loss_actor_sum += float(loss_info["loss_actor"])
                                loss_count += 1

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
                                value_teacher=(value_teacher if float(eta) > 0.0 else None),
                            )
                            wm_loss_sum += float(losses.loss_total)
                            wm_loss_count += 1

                next_states, rewards, dones, infos = vecenv.step_wait()
                next_global_states = next_states.reshape(n_envs, -1).astype(np.float32)
                reward_team = np.mean(np.asarray(rewards, dtype=np.float32), axis=1)

                is_last_step = step == int(n_steps) - 1
                for e in range(n_envs):
                    done_e = bool(dones[e]) or bool(is_last_step)

                    # QMIX replay (for warmup / baseline).
                    qmix.store_transition(
                        states=states[e],
                        actions=actions[e],
                        rewards=np.asarray(rewards[e], dtype=np.float32),
                        next_states=next_states[e],
                        done=bool(done_e),
                    )

                    # Sequence replay (s,u,r,s',done,env_id) for WM + VE.
                    seq_buffer.add(
                        env_id=int(e),
                        state=global_states[e],
                        action_discrete=action_discrete_all[e],
                        action_params=action_params_all[e],
                        reward_team=float(reward_team[e]),
                        next_state=next_global_states[e],
                        done=bool(done_e),
                    )

                states = next_states
                global_states = next_global_states
                episode_reward += float(np.mean(rewards))

            # Episode metrics
            total_links = float(int(n_steps) * n_envs * n_agents * int(env0.n_des))
            energy_arr, jump_arr, suc_arr = vecenv.get_metrics()
            avg_energy = float(np.sum(energy_arr) / (total_links + 1e-12))
            avg_jump = float(np.sum(jump_arr) / (total_links + 1e-12))
            avg_suc_per_link = float(np.sum(suc_arr) / (total_links + 1e-12))
            success_rate = float(np.clip((avg_suc_per_link + 3.0) / 4.0, 0.0, 1.0))

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            reward_history.append(float(episode_reward))
            success_rate_history.append(float(success_rate))
            energy_history.append(float(avg_energy))
            jump_history.append(float(avg_jump))

            wm_loss_count = max(1, int(wm_loss_count))
            wm_loss_total_history.append(float(wm_loss_sum / wm_loss_count))

            recent_window = min(100, len(reward_history))
            avg_reward = float(np.mean(reward_history[-recent_window:]))
            avg_sr = float(np.mean(success_rate_history[-recent_window:]))
            postfix = {
                "avg_r": f"{avg_reward:.3f}",
                "sr": f"{avg_sr:.3f}",
                "eps": f"{epsilon:.3f}",
                "a": f"{alpha_model:.2f}",
                "eta": f"{eta:.2f}",
                "phase": ("qmix" if bool(train_qmix) else "-") + ("+wm" if bool(train_wm) else ""),
            }
            if loss_count > 0:
                postfix["loss_q"] = f"{(loss_q_sum / loss_count):.3f}"
                postfix["loss_a"] = f"{(loss_actor_sum / loss_count):.3f}"
            postfix["wm"] = f"{(wm_loss_sum / wm_loss_count):.3f}"
            pbar.set_postfix(postfix)
    finally:
        vecenv.close()

    metrics = {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
        "wm_loss_total": wm_loss_total_history,
        "alpha_model": alpha_history,
        "eta": eta_history,
    }

    if save_data:
        _, _, out_dir = save_training_data(
            algorithm="mpdqn_qmix_wmve_rssm",
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=n_episode,
            n_steps=n_steps,
            trainer=qmix,
        )

        # Save world model checkpoint into the same experiment directory.
        _save_world_model_checkpoint(
            out_dir=out_dir,
            wm_trainer=wm_trainer,
            wm_cfg=asdict(wm_cfg_obj),
            td_cfg=asdict(td_cfg_obj),
            metrics={
                "wm_loss_total": [float(x) for x in wm_loss_total_history],
                "alpha_model": [float(x) for x in alpha_history],
                "eta": [float(x) for x in eta_history],
            },
        )

    return qmix, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QMIX + RSSM World Model Value Expansion")

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--updates-per-learn", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-q", type=float, default=1e-3)
    parser.add_argument("--lr-mixer", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics/weights")

    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--wm-buffer-capacity", type=int, default=500_000)
    parser.add_argument("--wm-hidden-dim", type=int, default=256)
    parser.add_argument("--wm-n-layers", type=int, default=1)
    parser.add_argument("--wm-stochastic-dim", type=int, default=32)
    parser.add_argument("--wm-kl-beta", type=float, default=0.1)
    parser.add_argument("--wm-free-nats", type=float, default=1.0)
    parser.add_argument("--wm-lr", type=float, default=1e-3)
    parser.add_argument("--wm-batch-size", type=int, default=512)
    parser.add_argument("--wm-updates-per-learn", type=int, default=2)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--rollout-k", type=int, default=4)

    parser.add_argument("--critic-warmup-ep", type=int, default=200)
    parser.add_argument("--model-warmup-ep", type=int, default=200)
    parser.add_argument("--ramp-start-ep", type=int, default=300)
    parser.add_argument("--ramp-end-ep", type=int, default=500)
    parser.add_argument("--alpha-model-max", type=float, default=0.01)
    parser.add_argument("--eta-max", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print("=" * 60)
    print("Starting QMIX + World Model Value Expansion Training (RSSM)")
    print("=" * 60)

    train_qmix_value_expansion(
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        learn_every=int(args.learn_every),
        updates_per_learn=int(args.updates_per_learn),
        lr_actor=float(args.lr_actor),
        lr_q=float(args.lr_q),
        lr_mixer=args.lr_mixer,
        max_grad_norm=float(args.max_grad_norm),
        use_amp=not bool(args.no_amp),
        device=args.device,
        start_method=str(args.start_method),
        save_data=not bool(args.no_save),
        seq_len=int(args.seq_len),
        wm_buffer_capacity=int(args.wm_buffer_capacity),
        wm_hidden_dim=int(args.wm_hidden_dim),
        wm_n_layers=int(args.wm_n_layers),
        wm_stochastic_dim=int(args.wm_stochastic_dim),
        wm_kl_beta=float(args.wm_kl_beta),
        wm_free_nats=float(args.wm_free_nats),
        wm_lr=float(args.wm_lr),
        wm_batch_size=int(args.wm_batch_size),
        wm_updates_per_learn=int(args.wm_updates_per_learn),
        gamma=float(args.gamma),
        lam=float(args.lam),
        rollout_k=int(args.rollout_k),
        critic_warmup_ep=int(args.critic_warmup_ep),
        model_warmup_ep=int(args.model_warmup_ep),
        ramp_start_ep=int(args.ramp_start_ep),
        ramp_end_ep=int(args.ramp_end_ep),
        alpha_model_max=float(args.alpha_model_max),
        eta_max=float(args.eta_max),
        seed=int(args.seed),
    )
    print("Training completed!")


