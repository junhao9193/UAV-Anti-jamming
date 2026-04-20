"""
Train MP-DQN (QMIX) with Value Expansion using a *frozen* pretrained RSSM world model.

User request:
  - load existing world model weights
  - do NOT update the world model
  - use fixed alpha_model=0.3 (y_model weight)
  - update QMIX only
  - run 1500 episodes

Run from `UAV-Jammer-RL/`:
  python -m Main.train.train_qmix_value_expansion_fixed_wm --world-model-weights ./world_model_weights.pth
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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


def _load_world_model(weights_path: Path, *, device: str):
    """
    Load a pretrained RSSM world model.

    Supports both checkpoint formats:
      - train_qmix_value_expansion.py: {"wm_cfg": ..., "td_cfg": ..., "wm_state_dict": ...}
      - train_world_model.py: {"config": {"wm_cfg": ..., "td_cfg": ...}, "wm_state_dict": ...}
    """
    import torch

    from algorithms.world_model.model import JointWorldModel, JointWorldModelConfig

    ckpt = torch.load(str(weights_path), map_location=device)
    wm_state_dict = ckpt.get("wm_state_dict", None)
    if wm_state_dict is None:
        raise ValueError(f"Invalid world model checkpoint (missing 'wm_state_dict'): {weights_path}")

    wm_cfg_raw = ckpt.get("wm_cfg", None)
    if wm_cfg_raw is None and isinstance(ckpt.get("config", None), dict):
        wm_cfg_raw = ckpt["config"].get("wm_cfg", None)
    if not isinstance(wm_cfg_raw, dict):
        raise ValueError(
            "Invalid world model checkpoint (missing 'wm_cfg' or 'config.wm_cfg'): "
            f"{weights_path}"
        )

    td_cfg_raw = ckpt.get("td_cfg", None)
    if td_cfg_raw is None and isinstance(ckpt.get("config", None), dict):
        td_cfg_raw = ckpt["config"].get("td_cfg", None)

    wm_cfg = JointWorldModelConfig(
        state_dim=int(wm_cfg_raw["state_dim"]),
        action_dim=int(wm_cfg_raw["action_dim"]),
        hidden_dim=int(wm_cfg_raw.get("hidden_dim", 256)),
        n_layers=int(wm_cfg_raw.get("n_layers", 1)),
        stochastic_dim=int(wm_cfg_raw.get("stochastic_dim", 32)),
        kl_beta=float(wm_cfg_raw.get("kl_beta", 0.1)),
        free_nats=float(wm_cfg_raw.get("free_nats", 1.0)),
    )

    wm = JointWorldModel(wm_cfg).to(device)
    wm.load_state_dict(wm_state_dict, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    return wm, wm_cfg, td_cfg_raw


def train_qmix_value_expansion_fixed_wm(
    *,
    n_episode: int = 1500,
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
    qmix_weights: str | None = None,
    world_model_weights: str | None = None,
    # Value Expansion
    alpha_model: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.8,
    rollout_k: int = 4,
    seq_len: int = 8,
    wm_buffer_capacity: int = 500_000,
    # Behavior policy
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 0,
) -> Tuple[object, dict]:
    # Delay torch import so env workers don't import torch/CUDA.
    import torch

    from algorithms.mpdqn.qmix.trainer_greedy_actor import MPDQNQMIXTrainer
    from algorithms.world_model import MPDQNQMIXDims, MPDQNQMIXValueTeacher, TDlambdaConfig
    from algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer

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

    # --- Frozen world model ---
    if world_model_weights is None:
        raise ValueError("world_model_weights is required")
    wm, wm_cfg, _td_cfg_raw = _load_world_model(Path(world_model_weights), device=str(device))

    # Sanity check: ensure env dims match world model dims.
    expected_state_dim = int(env0.n_ch) * int(env0.state_dim)
    if int(wm_cfg.state_dim) != int(expected_state_dim):
        raise ValueError(f"WM state_dim mismatch: wm={wm_cfg.state_dim} vs env={expected_state_dim}")

    # Value teacher uses QMIX target networks (frozen params).
    value_teacher = MPDQNQMIXValueTeacher(
        qmix,
        MPDQNQMIXDims(
            n_agents=int(env0.n_ch),
            agent_state_dim=int(env0.state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
        ),
    )
    td_cfg_obj = TDlambdaConfig(gamma=float(gamma), lam=float(lam), rollout_k=int(rollout_k))

    # Sequence replay for VE targets.
    seq_buffer = WorldModelSequenceReplayBuffer(n_envs=int(num_envs), capacity=int(wm_buffer_capacity))

    epsilon = float(epsilon_start)

    reward_history = []
    success_rate_history = []
    energy_history = []
    jump_history = []
    loss_q_history = []
    loss_actor_history = []

    n_envs = int(num_envs)
    n_agents = int(env0.n_ch)

    pbar = trange(n_episode, desc="Training(QMIX+VE-frozenWM)", unit="ep", ascii=True)
    try:
        for ep in pbar:
            states = vecenv.reset()  # (E,N,S)
            global_states = states.reshape(n_envs, -1).astype(np.float32)

            episode_reward = 0.0
            loss_q_sum = 0.0
            loss_actor_sum = 0.0
            loss_count = 0

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

                # --- Update QMIX while env workers simulate ---
                if (step + 1) % int(max(1, learn_every)) == 0:
                    for _ in range(int(max(1, updates_per_learn))):
                        loss_info = qmix.train_step_value_expansion(
                            seq_buffer=seq_buffer,
                            seq_len=int(seq_len),
                            world_model=wm,
                            value_teacher=value_teacher,
                            td_cfg=td_cfg_obj,
                            alpha_model=float(alpha_model),
                            n_channel=int(env0.n_channel),
                            n_des=int(env0.n_des),
                            power_min_dbm=float(env0.uav_power_min),
                            power_max_dbm=float(env0.uav_power_max),
                        )
                        if loss_info is None:
                            break
                        loss_q_sum += float(loss_info["loss_q"])
                        loss_actor_sum += float(loss_info["loss_actor"])
                        loss_count += 1

                next_states, rewards, dones, _infos = vecenv.step_wait()
                next_global_states = next_states.reshape(n_envs, -1).astype(np.float32)
                reward_team = np.mean(np.asarray(rewards, dtype=np.float32), axis=1)

                is_last_step = step == int(n_steps) - 1
                for e in range(n_envs):
                    done_e = bool(dones[e]) or bool(is_last_step)

                    # Keep QMIX replay populated (for completeness / debugging).
                    qmix.store_transition(
                        states=states[e],
                        actions=actions[e],
                        rewards=np.asarray(rewards[e], dtype=np.float32),
                        next_states=next_states[e],
                        done=bool(done_e),
                    )

                    # Sequence replay for VE.
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

            # Metrics
            total_links = float(int(n_steps) * n_envs * n_agents * int(env0.n_des))
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
            if loss_count > 0:
                loss_q_history.append(float(loss_q_sum / loss_count))
                loss_actor_history.append(float(loss_actor_sum / loss_count))
            else:
                loss_q_history.append(float("nan"))
                loss_actor_history.append(float("nan"))

            pbar.set_postfix(
                avg_r=float(np.mean(reward_history[-50:])),
                sr=float(success_rate_history[-1]),
                eps=float(epsilon),
                alpha=float(alpha_model),
                loss_q=float(loss_q_history[-1]),
                loss_a=float(loss_actor_history[-1]),
                buf=int(len(seq_buffer)),
            )
    finally:
        vecenv.close()

    metrics = {
        "reward": reward_history,
        "success_rate": success_rate_history,
        "energy": energy_history,
        "jump": jump_history,
        "loss_q": loss_q_history,
        "loss_actor": loss_actor_history,
        "alpha_model": float(alpha_model),
        "world_model_weights": str(world_model_weights),
        "qmix_weights": str(qmix_weights) if qmix_weights is not None else None,
        "td_cfg": {"gamma": float(gamma), "lam": float(lam), "rollout_k": int(rollout_k)},
    }

    if bool(save_data):
        algorithm = "mpdqn_qmix_ve_fixed_wm"
        _, _, out_dir = save_training_data(
            algorithm=algorithm,
            reward_history=reward_history,
            success_rate_history=success_rate_history,
            energy_history=energy_history,
            jump_history=jump_history,
            n_episode=int(n_episode),
            n_steps=int(n_steps),
            trainer=qmix,
        )

        # Save a sidecar config for reproducibility.
        try:
            import json

            (out_dir / "ve_fixed_wm_config.json").write_text(
                    json.dumps(
                        {
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "qmix_weights": metrics["qmix_weights"],
                            "world_model_weights": metrics["world_model_weights"],
                            "alpha_model": float(alpha_model),
                            "td_cfg": metrics["td_cfg"],
                            "seq_len": int(seq_len),
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
        except Exception:
            pass

    return qmix, metrics


if __name__ == "__main__":
    repo_root = get_repo_root()

    # Defaults: if weights are placed under `UAV-Jammer-RL/`, use them; otherwise fall back to latest in Draw/.
    local_qmix = Path("mpdqn_qmix_weights.pth")
    local_wm = Path("world_model_weights.pth")
    default_qmix = str(local_qmix) if local_qmix.exists() else str(
        _find_latest_weights(repo_root=repo_root, prefix="mpdqn_qmix_", filename="mpdqn_qmix_weights.pth")
    )
    default_wm = str(local_wm) if local_wm.exists() else str(
        _find_latest_weights(repo_root=repo_root, prefix="world_model_rssm_", filename="world_model_weights.pth")
    )

    parser = argparse.ArgumentParser(description="Train QMIX with Value Expansion using a frozen pretrained world model")

    parser.add_argument("--qmix-weights", type=str, default=default_qmix)
    parser.add_argument("--world-model-weights", type=str, default=default_wm)

    parser.add_argument("--episodes", type=int, default=1500)
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

    # Fixed per request (still exposed for convenience).
    parser.add_argument("--alpha-model", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--rollout-k", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--wm-buffer-capacity", type=int, default=500_000)

    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_qmix_value_expansion_fixed_wm(
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
        qmix_weights=str(args.qmix_weights) if args.qmix_weights else None,
        world_model_weights=str(args.world_model_weights),
        alpha_model=float(args.alpha_model),
        gamma=float(args.gamma),
        lam=float(args.lam),
        rollout_k=int(args.rollout_k),
        seq_len=int(args.seq_len),
        wm_buffer_capacity=int(args.wm_buffer_capacity),
        epsilon_start=float(args.epsilon_start),
        epsilon_min=float(args.epsilon_min),
        epsilon_decay=float(args.epsilon_decay),
        seed=int(args.seed),
    )


