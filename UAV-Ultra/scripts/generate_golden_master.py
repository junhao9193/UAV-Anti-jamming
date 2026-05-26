"""生成 Stage 0 golden-master 环境轨迹。

这个脚本只读取固定基准提交中的旧 `UAV-Jammer-RL/` 环境，不读取当前工作区里的
脏改动。生成出的 `.npz` 保存数值数组，`.json` 保存元信息和每步 info。

设计原则：
- 不依赖策略网络，只使用固定随机种子生成合法动作序列；
- 固定 env_seed 和 action_seed，确保新旧环境吃到同一串动作；
- 不修改旧环境源码；用运行期包装 `compute_reward` 的方式记录链路投递结果；
- 不把旧 `.pth` 权重或训练产物纳入 golden master。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


BASELINE_COMMIT = "2360ab92ec438528f6e194feda2405f9e943179d"
DEFAULT_ENV_SEEDS = (0, 1, 2)
DEFAULT_ACTION_SEED = 20260522
DEFAULT_STEPS = 32


def _run_git(repo_root: Path, *args: str) -> str:
    """在指定 worktree 中运行 git 命令，并返回 stdout。

    这里不用 shell 拼接，避免路径里有空格时出现歧义。失败时让 subprocess 抛错，
    这样生成器不会在错误基准上悄悄产出 trace。
    """

    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    """计算配置文件哈希，用来证明 trace 对应的环境配置没有漂移。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """把 numpy 标量/数组递归转换为 JSON 可写对象。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _state_to_array(state: Any) -> np.ndarray:
    """把旧环境返回的 state 转成稳定的 float64 数组。

    用 float64 而非 float32：旧环境用 numpy 默认 float64 计算 state，降精度成 float32
    会把标准答案截断到 ~1e-7 相对精度，Stage 3 的严格容差对齐会因参考值本身不准而误判。
    如果未来旧环境返回非规则嵌套结构，这里会直接报错，而不是保存 object array。
    golden-master 的目标是可数值比较，因此不接受隐式 object 序列。
    """

    return np.asarray(state, dtype=np.float64)


def _make_actions(env: Any, *, steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成固定合法动作序列。

    旧 MP-DQN 动作由两部分组成：
    - discrete action：选择每个目的节点使用的信道组合；
    - continuous params：为每个离散动作槽提供归一化功率参数。
    """

    rng = np.random.default_rng(seed)
    discrete = rng.integers(
        low=0,
        high=int(env.action_dim),
        size=(steps, int(env.n_ch)),
        dtype=np.int64,
    )
    params = rng.random(
        size=(steps, int(env.n_ch), int(env.total_param_dim)),
        dtype=np.float32,
    )
    return discrete, params


def _action_for_step(discrete: np.ndarray, params: np.ndarray, step: int) -> list[tuple[int, np.ndarray]]:
    """把保存用数组还原成旧环境 `step()` 接受的动作列表。"""

    return [
        (int(discrete[step, agent_idx]), np.asarray(params[step, agent_idx], dtype=np.float32).copy())
        for agent_idx in range(discrete.shape[1])
    ]


def _attach_delivery_probe(env: Any) -> None:
    """记录每条链路的投递结果，而不修改旧环境源码。

    旧环境的 `get_reward()` 会依次调用 `compute_reward(i, j, ...)`，返回传输时间和
    成功标记（成功为 1，失败为 -3）。这里在实例上包一层函数，把每次调用的结果
    写入 `_golden_delivery` 和 `_golden_transmit_time`，供 step 后保存。
    """

    original_compute_reward = env.compute_reward

    def wrapped_compute_reward(i, j, other_channel_list, pairs):
        transmit_time, success_flag = original_compute_reward(i, j, other_channel_list, pairs)
        env._golden_transmit_time[int(i), int(j)] = float(transmit_time)
        env._golden_success_flag[int(i), int(j)] = float(success_flag)
        env._golden_delivery[int(i), int(j)] = 1.0 if int(success_flag) == 1 else 0.0
        return transmit_time, success_flag

    env.compute_reward = wrapped_compute_reward


def _reset_delivery_probe_arrays(env: Any) -> None:
    """每个 step 前清空探针数组，避免上一步结果残留。"""

    env._golden_transmit_time = np.full((int(env.n_ch), int(env.n_des)), np.nan, dtype=np.float64)
    env._golden_success_flag = np.full((int(env.n_ch), int(env.n_des)), np.nan, dtype=np.float64)
    env._golden_delivery = np.full((int(env.n_ch), int(env.n_des)), np.nan, dtype=np.float64)


def _generate_trace_for_seed(environ_cls: type, *, env_seed: int, steps: int, action_seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """生成单个 env_seed 的 trace。"""

    env = environ_cls(config={"env_seed": int(env_seed)})
    # 基准提交里 `__init__` 的默认 p_trans 生成没有显式传入 p_trans_seed，会导致跨进程漂移。
    # Stage 0 必须是可复现标准答案，因此这里用配置里的 p_trans_seed 生成矩阵，并通过
    # reset(p_trans=...) 明确钉住 jammer Markov 转移。
    p_trans_rng = np.random.default_rng(int(env.p_trans_seed))
    p_trans = np.asarray(env.generate_p_trans(rng=p_trans_rng), dtype=np.float32)
    initial_state = _state_to_array(env.reset(p_trans=p_trans))
    _attach_delivery_probe(env)

    # reward_details() 恒返回固定长度元组（能量 / 跳频 / 成功率）；从实例推断宽度，
    # 不写死魔法数字，奖励分量数量变化时这里会自动跟随。
    n_reward_details = len(env.reward_details())

    discrete, params = _make_actions(env, steps=steps, seed=action_seed + int(env_seed))

    # 数值型输出统一用 float64 存：旧环境用 float64 计算，降精度成 float32 会把标准答案
    # 截断到 ~1e-7，Stage 3 的严格容差对齐会因参考值本身不准而误判。actions_params 是
    # 喂回环境的输入、保持 float32 即可；信道/干扰机索引是整数，用 int32。
    states = np.empty((steps + 1, *initial_state.shape), dtype=np.float64)
    rewards = np.empty((steps, int(env.n_ch)), dtype=np.float64)
    dones = np.empty((steps,), dtype=np.bool_)
    deliveries = np.empty((steps, int(env.n_ch), int(env.n_des)), dtype=np.float64)
    success_flags = np.empty_like(deliveries)
    transmit_times = np.empty_like(deliveries)
    uav_channels = np.empty((steps, int(env.n_ch), int(env.n_des)), dtype=np.int32)
    uav_powers = np.empty((steps, int(env.n_ch), int(env.n_des)), dtype=np.float64)
    jammer_current = np.empty((steps, int(env.n_jammer)), dtype=np.int32)
    jammer_next = np.empty((steps, int(env.n_jammer)), dtype=np.int32)
    reward_details = np.empty((steps, n_reward_details), dtype=np.float64)

    states[0] = initial_state
    step_infos: list[dict[str, Any]] = []

    for step in range(steps):
        _reset_delivery_probe_arrays(env)
        action = _action_for_step(discrete, params, step)
        next_state, reward, done, info = env.step(action)

        states[step + 1] = _state_to_array(next_state)
        # 数值型输出全部走 float64，避免 dtype=np.float32 截断（与 Stage 2 修复一致；
        # 否则即便存储数组是 float64，赋值前的显式 cast 仍会把 baseline 浮点结果截到 ~1e-7）。
        rewards[step] = np.asarray(reward, dtype=np.float64)
        dones[step] = bool(done)
        deliveries[step] = env._golden_delivery
        success_flags[step] = env._golden_success_flag
        transmit_times[step] = env._golden_transmit_time
        uav_channels[step] = np.asarray(env.uav_channels, dtype=np.int32)
        uav_powers[step] = np.asarray(env.uav_powers, dtype=np.float64)
        jammer_current[step] = np.asarray(info["jammer_channels_current"], dtype=np.int32)
        jammer_next[step] = np.asarray(info["jammer_channels_next"], dtype=np.int32)
        reward_details[step] = np.asarray(env.reward_details(), dtype=np.float64)
        step_infos.append(_jsonable(info))

        if done:
            # 默认 32 步远小于旧配置 1000 步；如果配置后续改变导致提前结束，直接停止。
            states = states[: step + 2]
            rewards = rewards[: step + 1]
            dones = dones[: step + 1]
            deliveries = deliveries[: step + 1]
            success_flags = success_flags[: step + 1]
            transmit_times = transmit_times[: step + 1]
            uav_channels = uav_channels[: step + 1]
            uav_powers = uav_powers[: step + 1]
            jammer_current = jammer_current[: step + 1]
            jammer_next = jammer_next[: step + 1]
            reward_details = reward_details[: step + 1]
            discrete = discrete[: step + 1]
            params = params[: step + 1]
            break

    prefix = f"seed_{env_seed}"
    arrays = {
        f"{prefix}_states": states,
        f"{prefix}_rewards": rewards,
        f"{prefix}_dones": dones,
        f"{prefix}_actions_discrete": discrete,
        f"{prefix}_actions_params": params,
        f"{prefix}_deliveries": deliveries,
        f"{prefix}_success_flags": success_flags,
        f"{prefix}_transmit_times": transmit_times,
        f"{prefix}_uav_channels": uav_channels,
        f"{prefix}_uav_powers": uav_powers,
        f"{prefix}_jammer_channels_current": jammer_current,
        f"{prefix}_jammer_channels_next": jammer_next,
        f"{prefix}_reward_details": reward_details,
        f"{prefix}_p_trans": p_trans,
    }
    metadata = {
        "env_seed": int(env_seed),
        "action_seed": int(action_seed + int(env_seed)),
        "steps_requested": int(steps),
        "steps_recorded": int(rewards.shape[0]),
        "n_ch": int(env.n_ch),
        "n_des": int(env.n_des),
        "n_jammer": int(env.n_jammer),
        "n_channel": int(env.n_channel),
        "state_shape": list(states.shape[1:]),
        "action_dim": int(env.action_dim),
        "param_dim_per_action": int(env.param_dim_per_action),
        "total_param_dim": int(env.total_param_dim),
        "p_trans_shape": list(p_trans.shape),
        "step_infos": step_infos,
    }
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="固定提交的独立 worktree 根目录，例如 /tmp/uav-ultra-baseline-2360。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/golden_master"),
        help="golden-master 输出目录，默认写入 UAV-Ultra/tests/fixtures/golden_master。",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="每个 seed 记录的环境步数。")
    parser.add_argument("--action-seed", type=int, default=DEFAULT_ACTION_SEED, help="动作序列根随机种子。")
    parser.add_argument(
        "--env-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENV_SEEDS),
        help="需要生成 trace 的 env_seed 列表。",
    )
    parser.add_argument(
        "--allow-dirty-baseline",
        action="store_true",
        help="允许 baseline worktree 有未提交改动。默认禁止，防止 golden master 漂移。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    baseline_project = baseline_root / "UAV-Jammer-RL"
    config_path = baseline_project / "configs" / "env.yaml"

    if not baseline_project.exists():
        raise FileNotFoundError(f"baseline project not found: {baseline_project}")

    head = _run_git(baseline_root, "rev-parse", "HEAD")
    if head != BASELINE_COMMIT:
        raise RuntimeError(f"baseline HEAD mismatch: got {head}, expected {BASELINE_COMMIT}")

    dirty = _run_git(baseline_root, "status", "--porcelain")
    if dirty and not args.allow_dirty_baseline:
        raise RuntimeError("baseline worktree is dirty; refuse to generate drifting golden master")

    sys.path.insert(0, str(baseline_project))
    old_envs = importlib.import_module("envs")
    environ_cls = old_envs.Environ

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    traces: list[dict[str, Any]] = []
    for env_seed in args.env_seeds:
        trace_arrays, trace_metadata = _generate_trace_for_seed(
            environ_cls,
            env_seed=int(env_seed),
            steps=int(args.steps),
            action_seed=int(args.action_seed),
        )
        arrays.update(trace_arrays)
        traces.append(trace_metadata)

    stem = f"env_trace_{BASELINE_COMMIT[:8]}"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "schema_version": 1,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_project": "UAV-Jammer-RL",
        "config_path": "UAV-Jammer-RL/configs/env.yaml",
        "config_sha256": _sha256_file(config_path),
        "generator": "UAV-Ultra/scripts/generate_golden_master.py",
        "steps": int(args.steps),
        "action_seed": int(args.action_seed),
        "env_seeds": [int(seed) for seed in args.env_seeds],
        "npz_file": npz_path.name,
        "traces": traces,
        "notes": [
            "旧环境不直接暴露 SINR；本 fixture 记录 state/reward/action、jammer channels、UAV channel/power、reward_details、p_trans 和每链路 delivery/transmit_time。",
            "delivery/transmit_time 通过运行期包装 compute_reward 记录，不修改旧环境源码。",
            "基准提交默认 p_trans 生成存在跨进程漂移；生成器显式使用 env.p_trans_seed 并通过 reset(p_trans=...) 固定 Markov 矩阵。",
        ],
    }
    json_path.write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
