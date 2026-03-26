"""
Train the joint recurrent world model (RSSM) from step transitions.

Replay stores only (s, u, r, s', done, env_id) per step, then samples contiguous
sequences by `env_id` for RNN training and value-consistency regularization.

Run from `UAV-Jammer-RL/`:
  python -m Main.train_world_model
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from envs import Environ
from Main.common import SubprocVecEnv, get_repo_root, make_fixed_p_trans, make_unique_output_dir
from tqdm.auto import trange


def _linear_ramp(t: int, *, t0: int, t1: int, v_max: float) -> float:
    if t1 <= t0:
        return float(v_max) if int(t) >= int(t0) else 0.0
    frac = (float(t) - float(t0)) / (float(t1) - float(t0))
    return float(v_max) * float(np.clip(frac, 0.0, 1.0))


def _find_latest_qmix_weights(repo_root: Path) -> Path:
    base_dir = repo_root / "Draw" / "experiment-data"
    if not base_dir.exists():
        raise FileNotFoundError(f"experiment-data dir not found: {base_dir}")

    candidates = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("mpdqn_qmix_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in candidates:
        w = d / "mpdqn_qmix_weights.pth"
        if w.exists():
            return w
    raise FileNotFoundError(f"No mpdqn_qmix_weights.pth found under: {base_dir}")


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

    for agent in trainer.agents:
        agent.actor.eval()
        agent.q_net.eval()
        agent.target_actor.eval()
        agent.target_q_net.eval()
    trainer.mixer.eval()
    trainer.target_mixer.eval()


def _save_world_model_artifacts(
    *,
    algorithm: str,
    repo_root: Path,
    losses: dict,
    config: dict,
    trainer,
) -> Path:
    import json
    import torch

    base_dir = repo_root / "Draw" / "experiment-data"
    out_dir = make_unique_output_dir(base_dir, algorithm)
    timestamp = out_dir.name.removeprefix(f"{algorithm}_")

    (out_dir / "world_model_config.json").write_text(
        json.dumps({"config": config, "losses": losses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ckpt = {
        "timestamp": timestamp,
        "algorithm": algorithm,
        "config": config,
        "wm_state_dict": trainer.wm.state_dict(),
        "opt_state_dict": trainer.opt.state_dict(),
    }
    torch.save(ckpt, str(out_dir / "world_model_weights.pth"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(losses["loss_total"]), dtype=np.int32)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(x, losses["loss_total"], label="loss_total")
        ax.plot(x, losses["loss_state"], label="loss_state")
        ax.plot(x, losses["loss_reward"], label="loss_reward")
        if "loss_kl" in losses:
            ax.plot(x, losses["loss_kl"], label="loss_kl")
        if any(v != 0.0 for v in losses["loss_vc"]):
            ax.plot(x, losses["loss_vc"], label="loss_vc")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("World Model Training Losses (RSSM)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out_dir / "world_model_losses.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")

    return out_dir


def train_world_model(
    *,
    n_episode: int = 1500,
    n_steps: int = 1000,
    num_envs: int = 32,
    batch_size: int = 512,
    buffer_capacity: int = 500_000,
    learn_every: int = 4,
    updates_per_learn: int = 2,
    seq_len: int = 8,
    hidden_dim: int = 256,
    n_layers: int = 1,
    stochastic_dim: int = 32,
    kl_beta: float = 0.1,
    free_nats: float = 1.0,
    lr: float = 3e-4,
    max_grad_norm: float = 10.0,
    alpha: float = 1.0,
    eta: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.8,
    rollout_k: int = 4,
    vc_warmup_ep: int = 300,
    vc_ramp_end_ep: int = 800,
    seed: int = 0,
    qmix_weights: Optional[str] = None,
    epsilon: float = 0.05,
    device: Optional[str] = None,
    start_method: str = "spawn",
    save: bool = True,
) -> Tuple[object, dict]:
    import torch

    from algorithms.mpdqn.qmix.trainer_greedy_actor import MPDQNQMIXTrainer
    from algorithms.world_model import JointWorldModelConfig, TDlambdaConfig, ValueConsistentWorldModelTrainer
    from algorithms.world_model.action_encoding import exec_action_dim
    from algorithms.world_model.qmix_adapters import MPDQNQMIXDims, MPDQNQMIXValueTeacher
    from algorithms.world_model.replay_buffer import WorldModelSequenceReplayBuffer

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

    # Basic seeding (sampling buffer still uses its own RNG).
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

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

    wm_cfg = JointWorldModelConfig(
        state_dim=int(global_state_dim),
        action_dim=int(action_dim_exec),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        stochastic_dim=int(stochastic_dim),
        kl_beta=float(kl_beta),
        free_nats=float(free_nats),
    )
    td_cfg = TDlambdaConfig(gamma=float(gamma), lam=float(lam), rollout_k=int(rollout_k))

    wm_trainer = ValueConsistentWorldModelTrainer(
        wm_cfg=wm_cfg,
        n_agents=int(n_agents),
        n_channel=int(env0.n_channel),
        n_des=int(env0.n_des),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        alpha=float(alpha),
        eta=float(eta),
        td_cfg=td_cfg,
        lr=float(lr),
        max_grad_norm=float(max_grad_norm),
        power_min_dbm=float(env0.uav_power_min),
        power_max_dbm=float(env0.uav_power_max),
        device=str(device),
    )

    # QMIX policy for data collection.
    qmix_trainer = MPDQNQMIXTrainer(
        n_agents=int(env0.n_ch),
        state_dim=int(env0.state_dim),
        n_actions=int(env0.action_dim),
        param_dim=int(env0.param_dim_per_action),
        global_state_dim=int(env0.state_dim * env0.n_ch),
        buffer_capacity=1,  # unused
        batch_size=1,
        lr_actor=1e-3,
        lr_q=1e-3,
        lr_mixer=1e-3,
        use_amp=False,
        max_grad_norm=0.0,
        device=str(device),
    )

    repo_root = get_repo_root()
    if qmix_weights is None:
        qmix_weights_path = _find_latest_qmix_weights(repo_root)
    else:
        qmix_weights_path = Path(qmix_weights).expanduser()
        if not qmix_weights_path.is_absolute():
            qmix_weights_path = (repo_root / qmix_weights_path).absolute()
    if not qmix_weights_path.exists():
        raise FileNotFoundError(f"QMIX weights not found: {qmix_weights_path}")
    _load_qmix_weights(qmix_trainer, qmix_weights_path, device=str(device))

    value_teacher = None
    if float(eta) > 0.0:
        dims = MPDQNQMIXDims(
            n_agents=int(n_agents),
            agent_state_dim=int(agent_state_dim),
            n_actions=int(env0.action_dim),
            param_dim=int(env0.param_dim_per_action),
        )
        value_teacher = MPDQNQMIXValueTeacher(qmix_trainer, dims)

    buffer = WorldModelSequenceReplayBuffer(n_envs=int(n_envs), capacity=int(buffer_capacity))
    loss_hist = {"loss_state": [], "loss_reward": [], "loss_kl": [], "loss_vc": [], "loss_total": []}

    try:
        pbar = trange(n_episode, desc="Training(World-Model-RSSM)", unit="ep", ascii=True)
        for episode in pbar:
            eta_now = _linear_ramp(int(episode), t0=int(vc_warmup_ep), t1=int(vc_ramp_end_ep), v_max=float(eta))
            wm_trainer.eta = float(eta_now)
            states = vecenv.reset()  # (E,N,S)
            global_states = states.reshape(n_envs, -1).astype(np.float32)  # (E,Ds)

            loss_state_sum = 0.0
            loss_reward_sum = 0.0
            loss_kl_sum = 0.0
            loss_vc_sum = 0.0
            loss_total_sum = 0.0
            loss_count = 0

            for step in range(int(n_steps)):
                action_discrete_all = np.zeros((n_envs, n_agents), dtype=np.int32)
                action_params_all = np.zeros((n_envs, n_agents, int(env0.total_param_dim)), dtype=np.float32)
                for i in range(n_agents):
                    ad_i, ap_i = qmix_trainer.agents[i].select_action_batch(states[:, i, :], float(epsilon))
                    action_discrete_all[:, i] = ad_i
                    action_params_all[:, i, :] = ap_i

                actions = [
                    [(int(action_discrete_all[e, i]), action_params_all[e, i, :]) for i in range(n_agents)]
                    for e in range(n_envs)
                ]

                vecenv.step_async(actions)

                if (step + 1) % int(max(1, learn_every)) == 0:
                    for _ in range(int(max(1, updates_per_learn))):
                        try:
                            batch = buffer.sample_sequences(batch_size=int(batch_size), seq_len=int(seq_len))
                        except Exception:
                            break
                        losses, _ = wm_trainer.train_step(
                            state_seq=torch.from_numpy(batch["state_seq"]),
                            action_discrete_seq=torch.from_numpy(batch["action_discrete_seq"]),
                            action_params_seq=torch.from_numpy(batch["action_params_seq"]),
                            reward_seq=torch.from_numpy(batch["reward_seq"]),
                            next_state_seq=torch.from_numpy(batch["next_state_seq"]),
                            value_teacher=(value_teacher if float(eta_now) > 0.0 else None),
                        )
                        loss_state_sum += float(losses.loss_state)
                        loss_reward_sum += float(losses.loss_reward)
                        loss_kl_sum += float(losses.loss_kl)
                        loss_vc_sum += float(losses.loss_vc)
                        loss_total_sum += float(losses.loss_total)
                        loss_count += 1

                next_states, rewards, dones, infos = vecenv.step_wait()
                next_global_states = next_states.reshape(n_envs, -1).astype(np.float32)
                reward_team = np.mean(np.asarray(rewards, dtype=np.float32), axis=1)

                is_last_step = step == int(n_steps) - 1
                for e in range(n_envs):
                    done_e = bool(dones[e]) or bool(is_last_step)
                    buffer.add(
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

            loss_count = max(1, int(loss_count))
            loss_hist["loss_state"].append(loss_state_sum / loss_count)
            loss_hist["loss_reward"].append(loss_reward_sum / loss_count)
            loss_hist["loss_kl"].append(loss_kl_sum / loss_count)
            loss_hist["loss_vc"].append(loss_vc_sum / loss_count)
            loss_hist["loss_total"].append(loss_total_sum / loss_count)

            pbar.set_postfix(
                {
                    "buf": str(len(buffer)),
                    "L": f"{(loss_total_sum / loss_count):.4f}",
                    "L_s": f"{(loss_state_sum / loss_count):.4f}",
                    "L_r": f"{(loss_reward_sum / loss_count):.4f}",
                    "L_kl": f"{(loss_kl_sum / loss_count):.4f}",
                    "L_vc": f"{(loss_vc_sum / loss_count):.4f}",
                    "eta": f"{eta_now:.3f}",
                    "envs": str(n_envs),
                }
            )
    finally:
        vecenv.close()

    config = {
        "n_episode": int(n_episode),
        "n_steps": int(n_steps),
        "num_envs": int(num_envs),
        "batch_size": int(batch_size),
        "buffer_capacity": int(buffer_capacity),
        "learn_every": int(learn_every),
        "updates_per_learn": int(updates_per_learn),
        "seq_len": int(seq_len),
        "seed": int(seed),
        "qmix_weights": str(qmix_weights_path),
        "stochastic_dim": int(stochastic_dim),
        "kl_beta": float(kl_beta),
        "free_nats": float(free_nats),
        "epsilon": float(epsilon),
        "max_grad_norm": float(max_grad_norm),
        "vc_warmup_ep": int(vc_warmup_ep),
        "vc_ramp_end_ep": int(vc_ramp_end_ep),
        "start_method": str(start_method),
        "wm_cfg": asdict(wm_cfg),
        "td_cfg": asdict(td_cfg),
        "alpha": float(alpha),
        "eta": float(eta),
        "lr": float(lr),
        "device": str(device),
        "env": {
            "n_agents": int(n_agents),
            "agent_state_dim": int(agent_state_dim),
            "global_state_dim": int(global_state_dim),
            "n_channel": int(env0.n_channel),
            "n_des": int(env0.n_des),
            "n_actions": int(env0.action_dim),
            "param_dim": int(env0.param_dim_per_action),
            "total_param_dim": int(env0.total_param_dim),
            "action_dim_exec": int(action_dim_exec),
        },
    }

    if save:
        out_dir = _save_world_model_artifacts(
            algorithm="world_model_rssm",
            repo_root=repo_root,
            losses=loss_hist,
            config=config,
            trainer=wm_trainer,
        )
        print(f"World model artifacts saved to: {out_dir}")

    return wm_trainer, {"losses": loss_hist, "config": config}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Joint World Model (RSSM)")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-capacity", type=int, default=500_000)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--updates-per-learn", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length sampled for recurrent world model training")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument('--stochastic-dim', type=int, default=32)
    parser.add_argument('--kl-beta', type=float, default=0.1)
    parser.add_argument('--free-nats', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--rollout-k", type=int, default=4)
    parser.add_argument("--vc-warmup-ep", type=int, default=300)
    parser.add_argument("--vc-ramp-end-ep", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--qmix-weights",
        type=str,
        default=None,
        help="Path to mpdqn_qmix_weights.pth. If omitted, uses the latest under Draw/experiment-data.",
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon-greedy for QMIX sampling")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train_world_model(
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        learn_every=int(args.learn_every),
        updates_per_learn=int(args.updates_per_learn),
        seq_len=int(args.seq_len),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        stochastic_dim=int(args.stochastic_dim),
        kl_beta=float(args.kl_beta),
        free_nats=float(args.free_nats),
        lr=float(args.lr),
        max_grad_norm=float(args.max_grad_norm),
        alpha=float(args.alpha),
        eta=float(args.eta),
        gamma=float(args.gamma),
        lam=float(args.lam),
        rollout_k=int(args.rollout_k),
        vc_warmup_ep=int(args.vc_warmup_ep),
        vc_ramp_end_ep=int(args.vc_ramp_end_ep),
        seed=int(args.seed),
        qmix_weights=(str(args.qmix_weights) if args.qmix_weights is not None else None),
        epsilon=float(args.epsilon),
        device=(str(args.device) if args.device is not None else None),
        start_method=str(args.start_method),
        save=(not bool(args.no_save)),
    )




