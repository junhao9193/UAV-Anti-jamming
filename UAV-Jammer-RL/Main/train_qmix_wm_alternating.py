"""
Alternating training loop: QMIX (with Value Expansion) <-> Value-Consistent RSSM World Model.

Supports two modes:
  - From scratch (default): initialize QMIX + RSSM randomly, then alternate training blocks
  - Resume/fine-tune: load QMIX and/or RSSM checkpoints if explicit paths are provided

Default schedule:
  - Freeze world model, update QMIX with mixed TD target for 50 episodes
  - Freeze QMIX, use it as behavior policy + value teacher, update world model for 50 episodes
  - Repeat, total episodes = 500

This is a "block coordinate descent" style alternating optimization, different from the
alpha/eta ramp schedule used in `train_qmix_value_expansion.py`.

Run from `UAV-Jammer-RL/`:
  python -m Main.train_qmix_wm_alternating
  python -m Main.train_qmix_wm_alternating --qmix-weights <...> --world-model-weights <...>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import trange

from envs import Environ
from Main.common import SubprocVecEnv, get_repo_root, make_fixed_p_trans, save_training_data


def _find_latest_weights(*, repo_root: Path, prefix: str, filename: str) -> Path:
    base_dir = repo_root / "Draw" / "experiment-data"
    if not base_dir.exists():
        raise FileNotFoundError(f"experiment-data dir not found: {base_dir}")

    candidates = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in candidates:
        w = d / filename
        if w.exists():
            return w
    raise FileNotFoundError(f"No {filename} found under: {base_dir} (prefix={prefix})")


def _load_qmix_weights(trainer, weights_path: Path, *, device: str) -> None:
    import torch

    ckpt = torch.load(str(weights_path), map_location=device)
    agents_sd = ckpt.get("agents", None)
    if not isinstance(agents_sd, list) or len(agents_sd) == 0:
        raise ValueError(f"Invalid QMIX checkpoint (missing 'agents'): {weights_path}")
    if len(agents_sd) != int(trainer.n_agents):
        raise ValueError(f"QMIX checkpoint agents={len(agents_sd)} != n_agents={trainer.n_agents}")

    for i, sd in enumerate(agents_sd):
        trainer.agents[i].actor.load_state_dict(sd["actor"])
        trainer.agents[i].q_net.load_state_dict(sd["q_net"])
        trainer.agents[i].target_actor.load_state_dict(sd.get("target_actor", sd["actor"]))
        trainer.agents[i].target_q_net.load_state_dict(sd.get("target_q_net", sd["q_net"]))

    if "mixer" in ckpt:
        trainer.mixer.load_state_dict(ckpt["mixer"])
    if "target_mixer" in ckpt:
        trainer.target_mixer.load_state_dict(ckpt["target_mixer"])


def _load_world_model_weights(wm_trainer, weights_path: Path, *, device: str, ckpt: Optional[dict] = None) -> dict:
    import torch

    if ckpt is None:
        ckpt = torch.load(str(weights_path), map_location=device)
    wm_sd = ckpt.get("wm_state_dict", None)
    if wm_sd is None:
        raise ValueError(f"Invalid world model checkpoint (missing 'wm_state_dict'): {weights_path}")
    wm_trainer.wm.load_state_dict(wm_sd, strict=True)

    opt_sd = ckpt.get("opt_state_dict", None)
    if opt_sd is not None:
        try:
            wm_trainer.opt.load_state_dict(opt_sd)
        except Exception:
            pass
    return ckpt


def train_qmix_wm_alternating(
    *,
    qmix_weights: str | None = None,
    world_model_weights: str | None = None,
    total_episodes: int = 500,
    qmix_block_episodes: int = 50,
    wm_block_episodes: int = 50,
    n_steps: int = 1000,
    num_envs: int = 32,
    # QMIX hyper-params
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
    # Value Expansion
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
    # World model training
    wm_lr: float = 1e-3,
    wm_batch_size: int = 512,
    wm_updates_per_learn: int = 2,
    wm_alpha_r: float = 1.0,
    wm_eta_vc: float = 0.2,
    # Behavior policy
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    save_data: bool = True,
    seed: int = 0,
) -> None:
    import json
    import torch

    from algorithms.mpdqn.qmix.trainer_greedy_actor import MPDQNQMIXTrainer
    from algorithms.world_model import JointWorldModelConfig, MPDQNQMIXDims, MPDQNQMIXValueTeacher, TDlambdaConfig
    from algorithms.world_model.action_encoding import exec_action_dim
    from algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer
    from algorithms.world_model.trainer import ValueConsistentWorldModelTrainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(use_amp) and str(device).startswith("cuda")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env0 = Environ()
    p_trans_fixed = make_fixed_p_trans(env0)
    vecenv = SubprocVecEnv(int(num_envs), p_trans=p_trans_fixed, start_method=str(start_method), seed=int(seed))

    # --- QMIX ---
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
    if qmix_weights is not None:
        _load_qmix_weights(qmix, Path(qmix_weights), device=str(device))
    for agent in qmix.agents:
        agent.actor.train()
        agent.q_net.train()
    qmix.mixer.train()

    # --- World model (trainer) ---
    n_agents = int(env0.n_ch)
    agent_state_dim = int(env0.state_dim)
    global_state_dim = int(n_agents * agent_state_dim)
    action_exec_dim = exec_action_dim(
        n_agents=int(n_agents),
        n_des=int(env0.n_des),
        n_channel=int(env0.n_channel),
        param_dim=int(env0.param_dim_per_action),
    )

    wm_ckpt_raw: dict = {}
    if world_model_weights is not None:
        wm_ckpt_raw = torch.load(str(world_model_weights), map_location=str(device))

        # Support both checkpoint formats:
        # - train_qmix_value_expansion.py: {"wm_cfg": ..., "wm_state_dict": ...}
        # - train_world_model.py: {"config": {"wm_cfg": ...}, "wm_state_dict": ...}
        wm_cfg_raw = None
        if isinstance(wm_ckpt_raw.get("wm_cfg", None), dict):
            wm_cfg_raw = wm_ckpt_raw["wm_cfg"]
        elif isinstance(wm_ckpt_raw.get("config", None), dict) and isinstance(wm_ckpt_raw["config"].get("wm_cfg", None), dict):
            wm_cfg_raw = wm_ckpt_raw["config"]["wm_cfg"]
        if not isinstance(wm_cfg_raw, dict):
            raise ValueError(
                "Invalid world model checkpoint (missing 'wm_cfg' or 'config.wm_cfg'): "
                f"{world_model_weights}"
            )
        if int(wm_cfg_raw.get("state_dim", -1)) != int(global_state_dim):
            raise ValueError(
                f"WM state_dim mismatch: ckpt={wm_cfg_raw.get('state_dim')} vs env={global_state_dim}"
            )
        if int(wm_cfg_raw.get("action_dim", -1)) != int(action_exec_dim):
            raise ValueError(
                f"WM action_dim mismatch: ckpt={wm_cfg_raw.get('action_dim')} vs env={action_exec_dim}"
            )
        wm_cfg = JointWorldModelConfig(
            state_dim=int(wm_cfg_raw["state_dim"]),
            action_dim=int(wm_cfg_raw["action_dim"]),
            hidden_dim=int(wm_cfg_raw.get("hidden_dim", 256)),
            n_layers=int(wm_cfg_raw.get("n_layers", 1)),
            stochastic_dim=int(wm_cfg_raw.get("stochastic_dim", 32)),
            kl_beta=float(wm_cfg_raw.get("kl_beta", 0.1)),
            free_nats=float(wm_cfg_raw.get("free_nats", 1.0)),
        )
    else:
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
        eta=float(wm_eta_vc),
        td_cfg=td_cfg,
        lr=float(wm_lr),
        power_min_dbm=float(env0.uav_power_min),
        power_max_dbm=float(env0.uav_power_max),
        device=str(device),
    )
    if world_model_weights is not None:
        wm_ckpt = _load_world_model_weights(
            wm_trainer,
            Path(world_model_weights),
            device=str(device),
            ckpt=wm_ckpt_raw,
        )
    else:
        wm_ckpt = {}

    # Teacher uses QMIX target networks (frozen params).
    value_teacher = MPDQNQMIXValueTeacher(
        qmix,
        MPDQNQMIXDims(
            n_agents=int(n_agents),
            agent_state_dim=int(agent_state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
        ),
    )

    # Sequence replay for both WM training and VE targets.
    seq_buffer = WorldModelSequenceReplayBuffer(n_envs=int(num_envs), capacity=int(wm_buffer_capacity))

    epsilon = float(epsilon_start)

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []
    phase_history = []
    wm_loss_total_history = []
    wm_loss_vc_history = []
    qmix_loss_q_history = []

    block_len = int(qmix_block_episodes) + int(wm_block_episodes)
    if block_len <= 0:
        raise ValueError("qmix_block_episodes + wm_block_episodes must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm = "mpdqn_qmix_wm_alternating"

    pbar = trange(int(total_episodes), desc="Training(QMIX<->WM)", unit="ep", ascii=True)
    try:
        for ep in pbar:
            pos_in_block = int(ep) % int(block_len)
            train_qmix = pos_in_block < int(qmix_block_episodes)
            train_wm = not bool(train_qmix)
            phase = "qmix" if bool(train_qmix) else "wm"
            phase_history.append(phase)

            # Purely for clarity/debug: set modes (no dropout/bn here, but keeps intent clear).
            if bool(train_qmix):
                wm_trainer.wm.eval()
                for agent in qmix.agents:
                    agent.actor.train()
                    agent.q_net.train()
                qmix.mixer.train()
            else:
                for agent in qmix.agents:
                    agent.actor.eval()
                    agent.q_net.eval()
                qmix.mixer.eval()
                wm_trainer.wm.train()

            states = vecenv.reset()  # (E,N,S)
            global_states = states.reshape(int(num_envs), -1).astype(np.float32)

            episode_reward = 0.0
            q_loss_q_sum = 0.0
            q_loss_count = 0
            wm_loss_sum = 0.0
            wm_vc_sum = 0.0
            wm_loss_count = 0

            for step in range(int(n_steps)):
                action_discrete_all = np.zeros((int(num_envs), int(n_agents)), dtype=np.int32)
                action_params_all = np.zeros((int(num_envs), int(n_agents), int(env0.total_param_dim)), dtype=np.float32)

                for i in range(int(n_agents)):
                    ad, ap = qmix.agents[i].select_action_batch(states[:, i, :], float(epsilon))
                    action_discrete_all[:, i] = ad
                    action_params_all[:, i, :] = ap

                actions = [
                    [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(int(n_agents))]
                    for e in range(int(num_envs))
                ]

                vecenv.step_async(actions)

                # --- Updates while env workers simulate ---
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
                                value_teacher=value_teacher,
                            )
                            wm_loss_sum += float(losses.loss_total)
                            wm_vc_sum += float(losses.loss_vc)
                            wm_loss_count += 1

                next_states, rewards, dones, _infos = vecenv.step_wait()
                next_global_states = next_states.reshape(int(num_envs), -1).astype(np.float32)
                reward_team = np.mean(np.asarray(rewards, dtype=np.float32), axis=1)

                is_last_step = step == int(n_steps) - 1
                for e in range(int(num_envs)):
                    done_e = bool(dones[e]) or bool(is_last_step)

                    # Keep QMIX replay populated (used by qmix.train_step fallback).
                    qmix.store_transition(
                        states=states[e],
                        actions=actions[e],
                        rewards=np.asarray(rewards[e], dtype=np.float32),
                        next_states=next_states[e],
                        done=bool(done_e),
                    )

                    # Sequence replay for WM + VE.
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
            total_links = float(int(n_steps) * int(num_envs) * int(n_agents) * int(env0.n_des))
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
            wm_loss_total_history.append(float(wm_loss_sum / max(1, wm_loss_count)))
            wm_loss_vc_history.append(float(wm_vc_sum / max(1, wm_loss_count)))

            pbar.set_postfix(
                phase=phase,
                avg_r=float(np.mean(reward_history[-10:])),
                sr=float(success_rate_history[-1]),
                eps=float(epsilon),
                alpha=float(alpha_model),
                eta=float(wm_eta_vc),
                qloss=float(qmix_loss_q_history[-1]),
                wm=float(wm_loss_total_history[-1]),
                lvc=float(wm_loss_vc_history[-1]),
                buf=int(len(seq_buffer)),
            )
    finally:
        vecenv.close()

    if not bool(save_data):
        return

    # Save standard RL metrics + QMIX weights.
    _, _, out_dir = save_training_data(
        algorithm=algorithm,
        reward_history=reward_history,
        success_rate_history=success_rate_history,
        energy_history=energy_history,
        jump_history=jump_history,
        n_episode=int(total_episodes),
        n_steps=int(n_steps),
        trainer=qmix,
    )

    # Save extra artifacts (WM weights + combined metrics) in the same experiment dir.

    # WM weights
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
            "init_ckpt_meta": {
                "loaded_from": (str(world_model_weights) if world_model_weights is not None else None),
                "orig_td_cfg": wm_ckpt.get("td_cfg", None)
                if wm_ckpt.get("td_cfg", None) is not None
                else (wm_ckpt.get("config", {}) or {}).get("td_cfg", None),
                "initialized_from_scratch": bool(world_model_weights is None),
            },
        },
        str(out_dir / "world_model_weights.pth"),
    )

    # Extra metrics
    (out_dir / "alternating_config.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "qmix_weights": (str(qmix_weights) if qmix_weights is not None else None),
                "world_model_weights": (str(world_model_weights) if world_model_weights is not None else None),
                "qmix_initialized_from_scratch": bool(qmix_weights is None),
                "world_model_initialized_from_scratch": bool(world_model_weights is None),
                "total_episodes": int(total_episodes),
                "qmix_block_episodes": int(qmix_block_episodes),
                "wm_block_episodes": int(wm_block_episodes),
                "alpha_model": float(alpha_model),
                "wm_eta_vc": float(wm_eta_vc),
                "td_cfg": {"gamma": float(gamma), "lam": float(lam), "rollout_k": int(rollout_k)},
                "seq_len": int(seq_len),
                "num_envs": int(num_envs),
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
                "wm_loss_total": wm_loss_total_history,
                "wm_loss_vc": wm_loss_vc_history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternating QMIX<->WorldModel training (from scratch or resume)")
    parser.add_argument("--qmix-weights", type=str, default=None, help="Optional QMIX checkpoint to resume from")
    parser.add_argument(
        "--world-model-weights",
        type=str,
        default=None,
        help="Optional RSSM world model checkpoint to resume from",
    )

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--qmix-block", type=int, default=50)
    parser.add_argument("--wm-block", type=int, default=50)
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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
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

    parser.add_argument("--wm-lr", type=float, default=1e-3)
    parser.add_argument("--wm-batch-size", type=int, default=512)
    parser.add_argument("--wm-updates-per-learn", type=int, default=2)
    parser.add_argument("--wm-alpha-r", type=float, default=1.0)
    parser.add_argument("--wm-eta-vc", type=float, default=0.2)

    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_qmix_wm_alternating(
        qmix_weights=args.qmix_weights,
        world_model_weights=args.world_model_weights,
        total_episodes=int(args.episodes),
        qmix_block_episodes=int(args.qmix_block),
        wm_block_episodes=int(args.wm_block),
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
        wm_lr=float(args.wm_lr),
        wm_batch_size=int(args.wm_batch_size),
        wm_updates_per_learn=int(args.wm_updates_per_learn),
        wm_alpha_r=float(args.wm_alpha_r),
        wm_eta_vc=float(args.wm_eta_vc),
        epsilon_start=float(args.epsilon_start),
        epsilon_min=float(args.epsilon_min),
        epsilon_decay=float(args.epsilon_decay),
        save_data=not bool(args.no_save),
        seed=int(args.seed),
    )

