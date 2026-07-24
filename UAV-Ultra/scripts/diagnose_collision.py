"""第一步诊断闸门：量化 UAV 自干扰（8 link / 6 信道碰撞）是否是 SR 的主瓶颈。

只读诊断（不改环境源码、不训练）。加载一个 JP-on checkpoint（默认 exp10），走**完整
eval 语义**（JP sensing-history 状态机 + greedy/epsilon=0），单进程驱动单个 ``Environ``
以便逐步读取 ``env.uav_channels`` / ``env.last_link_metrics``。

输出三组核心证据：
  1. 信道利用：平均 distinct 信道数、碰撞 link 占比。
  2. **碰撞失败率分层 jammer**（4 格）：fail_rate_{collided,clean}_{jammed,unjammed}，
     核心看 **unjammed 的 collided-vs-clean fail gap**（排除 jammer 致败的纯自干扰证据）。
  3. 两个上界（同一 post-step CSI 重算，delta 内部一致）：
     - ``SR_delete_interference``（乐观天花板）：每 link 去掉同信道干扰者 → 自干扰凭空消失。
     - ``SR_feasible_reassign``（可实现近似）：贪心把碰撞 link 重分配到最低占用信道，
       在**完整候选矩阵**下重算全部 8 link 的 suc。

判据：unjammed collided-clean fail gap 明显 **且** feasible ΔSR ≥ 0.02–0.03 → 做二三步；
否则停（自干扰非主瓶颈）。

⚠️ CSI 近似：上界用 step 之后的信道损耗（mobility/fading 已滚动一拍）重算；real-recomp 与
candidate 用同一 CSI，故 **ΔSR 内部一致**，仅绝对值与 env 真实 delivery 略有偏差（脚本会同时
打印 real-recomp 与 env 真实 SR 供核验）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.envs.reward as reward_module  # noqa: E402
from src.algorithms import build_evaluator, build_trainer  # noqa: E402
from src.config import specs  # noqa: E402
from src.config.loader import (  # noqa: E402
    _deep_merge,
    load_algo_config,
    load_env_config,
    load_experiment_preset,
)
from src.envs import Environ  # noqa: E402
from src.envs.jammer_model import JammerEvent  # noqa: E402
from src.envs.link_budget import compute_link  # noqa: E402
from src.evaluation.runner import _set_eval_mode  # noqa: E402
from src.training.callbacks import build_callbacks  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    load_callback_states,
    load_trainer_state_dict,
)

_EVAL_JP_FULL_SCALE_EPISODE = 10**9


def _install_jammer_event_capture() -> None:
    """Monkeypatch ``compute_step_reward`` 抓取本步实际使用的 jammer_events 快照。

    reward 在用完后清空 ``env.jammer_events``（[reward.py:101]），step 内 jammer 又会转移，
    故 step 返回后无法还原本步 jammer 状态。这里在 reward 入口抓快照（含 reward 自身的
    lazy-build：jammer_events 为空时按 jammer_channels 造静态 [0,t_Rx] 事件，与
    reward.py:42-47 一致），存进 ``env._diag_events`` 供上界重算使用。
    """
    orig = reward_module.compute_step_reward

    def _capturing(env):
        events = env.jammer_events
        if not events:
            events = [
                JammerEvent(jammer_idx=k, channel=int(env.jammer_channels[k]),
                            t_start=0.0, t_end=float(env.t_Rx))
                for k in range(env.n_jammer)
            ]
        env._diag_events = list(events)
        return orig(env)

    reward_module.compute_step_reward = _capturing


def _build_eval(preset: str, checkpoint: str, seed: int, device: str):
    """完整 eval 加载链：env_cfg / algo_cfg / trainer(+ckpt) / callbacks(+states) / evaluator。"""
    ep = load_experiment_preset(preset)
    if ep.algorithm != "qmix":
        raise ValueError(f"diagnose_collision 仅支持 qmix preset，得到 {ep.algorithm!r}")
    # env_seed 固定才可复现（preset 默认 null 是随机熵）；p_trans_seed 走 preset/默认（同一 jammer 基底）
    env_overrides = _deep_merge(ep.env, {"env_seed": int(seed)})
    env_cfg = load_env_config(overrides=env_overrides)
    algo_cfg = load_algo_config("qmix", overrides=ep.algo, env_cfg=env_cfg)

    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device=device)
    ckpt = load_trainer_state_dict(
        trainer, checkpoint, "qmix", device=device, strict=True, load_optimizers=False
    )
    _set_eval_mode(trainer)

    callbacks = build_callbacks(getattr(algo_cfg, "callbacks", []), env_cfg=env_cfg, algo_cfg=algo_cfg)
    callbacks.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    load_callback_states(callbacks.callbacks, ckpt.get("callbacks", {}), strict=True)

    evaluator = build_evaluator("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)
    return env_cfg, env_overrides, callbacks, evaluator


def _recompute_sucs(env, channel_matrix: np.ndarray, events, *, drop_self_interference: bool) -> np.ndarray:
    """在给定 channel_matrix + 本步 jammer events 下重算全部 link 的 suc（不污染 env）。

    ``drop_self_interference=True``：每 link 的 other_channel_list 剔除同信道干扰者
    （uav_uav_interference→0），即 delete-interference 乐观上界。
    """
    n_ch, n_des = int(env.n_ch), int(env.n_des)
    saved_ch = env.uav_channels
    saved_ev = env.jammer_events
    out = np.zeros((n_ch, n_des), dtype=np.int64)
    try:
        env.uav_channels = channel_matrix
        env.jammer_events = events
        for tra in range(n_ch):
            for rec in range(n_des):
                target = int(channel_matrix[tra][rec])
                other_channel_list, pairs = [], []
                for i in range(n_ch):
                    for j in range(n_des):
                        if i == tra and j == rec:
                            continue
                        ch = int(channel_matrix[i][j])
                        if drop_self_interference and ch == target:
                            continue
                        other_channel_list.append(ch)
                        pairs.append([i, j])
                _t, suc = compute_link(env, tra, rec, other_channel_list, pairs)
                out[tra, rec] = int(suc)
    finally:
        env.uav_channels = saved_ch
        env.jammer_events = saved_ev
    return out


def _greedy_reassign(uav_ch: np.ndarray, n_channel: int) -> np.ndarray:
    """碰撞 link 逐个重分配到当前最低占用信道（tie-breaker 固定 (occupancy, channel_id) 取最小）。"""
    shape = uav_ch.shape
    flat = uav_ch.reshape(-1).astype(np.int64).copy()
    counts = np.bincount(flat, minlength=n_channel).astype(np.int64)
    for idx in range(flat.shape[0]):
        ch = int(flat[idx])
        if counts[ch] > 1:
            counts[ch] -= 1
            target = min(range(n_channel), key=lambda c: (int(counts[c]), c))
            flat[idx] = target
            counts[target] += 1
    return flat.reshape(shape).astype(uav_ch.dtype)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="第一步诊断闸门：UAV 自干扰是否主瓶颈")
    parser.add_argument("--preset", type=str, default="qmix_wm_block_jp_baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=3000, help="总 transition 数（跨 episode reset 累计）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gap-threshold", type=float, default=0.05)
    parser.add_argument("--feasible-threshold", type=float, default=0.02)
    args = parser.parse_args(argv)

    _install_jammer_event_capture()
    env_cfg, env_overrides, callbacks, evaluator = _build_eval(
        args.preset, args.checkpoint, int(args.seed), str(args.device)
    )

    env = Environ(config=env_overrides)
    p_trans = np.asarray(env.generate_p_trans(rng=np.random.default_rng(int(env.p_trans_seed))), dtype=np.float32)
    n_channel = int(env_cfg.n_channel)

    # 4 格累计：[collided][jammed] -> {fail, total}
    fail = np.zeros((2, 2), dtype=np.int64)
    total = np.zeros((2, 2), dtype=np.int64)
    distinct_sum = 0.0
    collided_link_sum = 0.0
    n_links_per_step = int(env.n_ch) * int(env.n_des)

    sr_real_env = 0.0          # env 真实 delivery
    sr_real_recomp = 0.0       # 同 CSI 重算（核验重算保真度）
    sr_delete = 0.0            # 乐观上界
    sr_feasible = 0.0          # 可实现近似
    n_steps_done = 0
    n_resets = 0

    states = np.asarray(env.reset(p_trans), dtype=np.float32)   # (n_ch, state_dim)
    callbacks.reset_jp_state(states[None, ...], episode=_EVAL_JP_FULL_SCALE_EPISODE)
    n_resets += 1

    for _ in range(int(args.steps)):
        sh = callbacks.get_current_sensing_histories()          # (1, n_ch, H, C) or None
        hist = None if sh is None else sh[0]                    # (n_ch, H, C)
        vec_actions = [evaluator.select_actions(list(states), sensing_histories=hist)]
        vec_actions = callbacks.on_action_selected(vec_actions)
        actions = vec_actions[0]

        next_states, _reward, done, _info = env.step(actions)
        next_states = np.asarray(next_states, dtype=np.float32)
        callbacks.advance_jp_history(states=states[None, ...], next_states=next_states[None, ...])
        callbacks.commit_jp_history_swap()

        # ---- 采集诊断（env 此刻为 post-step）----
        uav_ch = env.uav_channels.copy()                        # (n_ch, n_des) int
        m = env.last_link_metrics
        delivery = np.asarray(m["delivery"])                    # (n_ch, n_des) 0/1
        jammed = np.asarray(m["jammer_exposure"], dtype=bool)   # (n_ch, n_des) bool
        events = env._diag_events

        counts = np.bincount(uav_ch.reshape(-1), minlength=n_channel)
        collided = counts[uav_ch] > 1                           # (n_ch, n_des) bool
        link_fail = delivery < 0.5

        for c in (0, 1):
            for jm in (0, 1):
                mask = (collided == bool(c)) & (jammed == bool(jm))
                total[c, jm] += int(mask.sum())
                fail[c, jm] += int((mask & link_fail).sum())

        distinct_sum += float(len(np.unique(uav_ch.reshape(-1))))
        collided_link_sum += float(collided.sum()) / n_links_per_step

        # 上界重算（同一 post-step CSI + 本步 jammer events 快照）
        suc_real = _recompute_sucs(env, uav_ch, events, drop_self_interference=False)
        suc_delete = _recompute_sucs(env, uav_ch, events, drop_self_interference=True)
        cand = _greedy_reassign(uav_ch, n_channel)
        suc_feasible = _recompute_sucs(env, cand, events, drop_self_interference=False)

        sr_real_env += float((delivery >= 0.5).mean())
        sr_real_recomp += float((suc_real == 1).mean())
        sr_delete += float((suc_delete == 1).mean())
        sr_feasible += float((suc_feasible == 1).mean())

        n_steps_done += 1
        states = next_states
        if bool(np.any(done)):
            states = np.asarray(env.reset(p_trans), dtype=np.float32)
            callbacks.reset_jp_state(states[None, ...], episode=_EVAL_JP_FULL_SCALE_EPISODE)
            n_resets += 1

    n = max(1, n_steps_done)
    fr = np.where(total > 0, fail / np.maximum(total, 1), np.nan)
    gap_unjammed = float(fr[1, 0] - fr[0, 0])      # collided_unjammed - clean_unjammed
    gap_jammed = float(fr[1, 1] - fr[0, 1])
    sr_real_env /= n
    sr_real_recomp /= n
    sr_delete /= n
    sr_feasible /= n
    feasible_dsr = sr_feasible - sr_real_recomp
    delete_dsr = sr_delete - sr_real_recomp

    def _pct(x: float) -> str:
        return "  nan" if np.isnan(x) else f"{100*x:5.1f}%"

    print("=" * 68)
    print(f" 诊断闸门  preset={args.preset}  ckpt={args.checkpoint}")
    print(f" steps={n_steps_done}  resets={n_resets}  seed={args.seed}  n_channel={n_channel}  links/step={n_links_per_step}")
    print("=" * 68)
    print(" [1] 信道利用")
    print(f"     平均 distinct 信道数 : {distinct_sum / n:.3f} / {n_channel}")
    print(f"     碰撞 link 占比       : {_pct(collided_link_sum / n)}")
    print(" [2] 失败率分层 jammer   (fail / total)")
    print(f"     collided & jammed    : {_pct(fr[1,1])}  ({int(fail[1,1])}/{int(total[1,1])})")
    print(f"     clean    & jammed    : {_pct(fr[0,1])}  ({int(fail[0,1])}/{int(total[0,1])})")
    print(f"     collided & UNJAMMED  : {_pct(fr[1,0])}  ({int(fail[1,0])}/{int(total[1,0])})")
    print(f"     clean    & UNJAMMED  : {_pct(fr[0,0])}  ({int(fail[0,0])}/{int(total[0,0])})")
    print(f"     >>> UNJAMMED collided-clean gap = {_pct(gap_unjammed)}   (jammed gap = {_pct(gap_jammed)})")
    print(" [3] SR 上界  (同 post-step CSI 重算，ΔSR 内部一致)")
    print(f"     SR_real (env delivery)     : {_pct(sr_real_env)}")
    print(f"     SR_real_recomp (核验)      : {_pct(sr_real_recomp)}   <- 应接近 SR_real")
    print(f"     SR_delete_interference     : {_pct(sr_delete)}   (Δ +{_pct(delete_dsr)})  乐观天花板")
    print(f"     SR_feasible_reassign       : {_pct(sr_feasible)}   (Δ +{_pct(feasible_dsr)})  可实现近似")
    print("=" * 68)
    gap_ok = (not np.isnan(gap_unjammed)) and gap_unjammed >= args.gap_threshold
    feasible_ok = feasible_dsr >= args.feasible_threshold
    verdict = "继续做二三步 (reward shaping + observation)" if (gap_ok and feasible_ok) else "停：自干扰非主瓶颈 / 协调收益有限"
    print(f" 判据  unjammed gap≥{args.gap_threshold:.0%}: {gap_ok}   feasible ΔSR≥{args.feasible_threshold:.0%}: {feasible_ok}")
    print(f" 结论  >>> {verdict}")
    print("=" * 68)


if __name__ == "__main__":
    main()
