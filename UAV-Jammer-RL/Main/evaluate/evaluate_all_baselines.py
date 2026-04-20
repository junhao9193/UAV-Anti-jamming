"""Batch evaluation for all saved comparison baselines under Draw/experiment-data.

It scans each first-level experiment folder and dispatches to the appropriate
single-model evaluator:
- MP-DQN-family checkpoints -> Main.evaluate.evaluate_mpdqn
- MAPPO checkpoints         -> Main.evaluate.evaluate_mappo

Output naming uses the folder name instead of the checkpoint/json algorithm name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from Main.common import get_repo_root, resolve_env_config_path, validate_positive_run_args
from Main.evaluate.evaluate_mappo import evaluate_mappo
from Main.evaluate.evaluate_mpdqn import evaluate_policy


def _default_experiments_root() -> str:
    return str((get_repo_root() / "Draw" / "experiment-data").absolute())


def _normalize_folder_selection(folders: list[str] | None) -> set[str] | None:
    if not folders:
        return None
    return {str(x).strip() for x in folders if str(x).strip()}


def _find_checkpoint(exp_dir: Path) -> tuple[str | None, Path | None]:
    mappo_ckpt = exp_dir / "mappo_weights.pth"
    if mappo_ckpt.exists():
        return "mappo", mappo_ckpt

    mpdqn_ckpts = sorted(
        [p for p in exp_dir.glob("mpdqn*_weights.pth") if p.name != "world_model_weights.pth"]
    )
    if mpdqn_ckpts:
        return "mpdqn", mpdqn_ckpts[0]

    return None, None


def evaluate_all_baselines(
    *,
    experiments_root: str | None = None,
    folders: list[str] | None = None,
    include_heuristic: bool = True,
    heuristic_policies: list[str] | None = None,
    n_episode: int = 100,
    n_steps: int = 1000,
    num_envs: int = 32,
    device: str | None = None,
    save_data: bool = True,
    start_method: str = "spawn",
    seed: int = 0,
    config_path: str | None = None,
) -> dict:
    validate_positive_run_args(n_episode=n_episode, n_steps=n_steps, num_envs=num_envs)
    config_path = resolve_env_config_path(config_path)
    experiments_root = str(Path(experiments_root or _default_experiments_root()).expanduser().absolute())

    root = Path(experiments_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"experiments_root does not exist or is not a directory: {root}")

    selected = _normalize_folder_selection(folders)
    results = {"evaluated": [], "skipped": []}

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        folder_name = exp_dir.name
        if folder_name == "__pycache__":
            continue
        if selected is not None and folder_name not in selected:
            continue

        kind, ckpt = _find_checkpoint(exp_dir)
        if kind is None or ckpt is None:
            results["skipped"].append({
                "folder": folder_name,
                "reason": "no_supported_checkpoint",
            })
            continue

        algorithm_name = f"eval_{folder_name}_seed{int(seed)}"
        print(f"[EVAL] {folder_name} -> {kind} using {ckpt.name}")

        if kind == "mpdqn":
            evaluate_policy(
                mode="mpdqn",
                weights=str(ckpt),
                algorithm_name=algorithm_name,
                n_episode=int(n_episode),
                n_steps=int(n_steps),
                num_envs=int(num_envs),
                device=device,
                save_data=bool(save_data),
                start_method=str(start_method),
                seed=int(seed),
                config_path=config_path,
            )
        elif kind == "mappo":
            evaluate_mappo(
                weights=str(ckpt),
                algorithm_name=algorithm_name,
                n_episode=int(n_episode),
                n_steps=int(n_steps),
                num_envs=int(num_envs),
                device=device,
                save_data=bool(save_data),
                start_method=str(start_method),
                seed=int(seed),
                config_path=config_path,
            )

        results["evaluated"].append({
            "folder": folder_name,
            "kind": kind,
            "checkpoint": str(ckpt),
            "algorithm_name": algorithm_name,
        })

    if include_heuristic:
        default_policy_modes = {
            "random": "fixed_mid",
            "greedy_sensing": "quality_adaptive",
            "max_csi": "quality_adaptive",
            "min_interference": "fixed_mid",
        }
        chosen = heuristic_policies or list(default_policy_modes.keys())
        for policy in chosen:
            if policy not in default_policy_modes:
                raise ValueError(
                    f"Unknown heuristic policy {policy!r}; expected one of {sorted(default_policy_modes)}"
                )
            power_mode = default_policy_modes[policy]
            algorithm_name = f"eval_{policy}_seed{int(seed)}"
            print(f"[EVAL] heuristic {policy} ({power_mode})")
            evaluate_policy(
                mode="heuristic",
                heuristic_policy=str(policy),
                power_mode=str(power_mode),
                algorithm_name=algorithm_name,
                n_episode=int(n_episode),
                n_steps=int(n_steps),
                num_envs=int(num_envs),
                device=device,
                save_data=bool(save_data),
                start_method=str(start_method),
                seed=int(seed),
                config_path=config_path,
            )
            results["evaluated"].append({
                "folder": policy,
                "kind": "heuristic",
                "checkpoint": None,
                "algorithm_name": algorithm_name,
            })

    print("[SUMMARY] evaluated:", len(results["evaluated"]), "skipped:", len(results["skipped"]))
    for item in results["skipped"]:
        print(f"  [SKIP] {item['folder']}: {item['reason']}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch evaluate all comparison baselines under Draw/experiment-data")
    parser.add_argument("--experiments-root", type=str, default=None, help="Root dir containing experiment folders (default: Draw/experiment-data)")
    parser.add_argument("--folders", nargs="*", default=None, help="Optional subset of folder names to evaluate")
    parser.add_argument("--skip-heuristic", action="store_true", help="Skip heuristic evaluations")
    parser.add_argument("--heuristic-policies", nargs="*", default=None, choices=["random", "greedy_sensing", "max_csi", "min_interference"], help="Optional subset of heuristic policies")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--start-method", type=str, default="spawn", help="spawn|fork|forkserver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-path", type=str, default=None, help="Env YAML config path (default: configs/env.yaml)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving metrics")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_all_baselines(
        experiments_root=args.experiments_root,
        folders=args.folders,
        include_heuristic=not bool(args.skip_heuristic),
        heuristic_policies=args.heuristic_policies,
        n_episode=int(args.episodes),
        n_steps=int(args.steps),
        num_envs=int(args.num_envs),
        device=args.device,
        save_data=not bool(args.no_save),
        start_method=str(args.start_method),
        seed=int(args.seed),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
