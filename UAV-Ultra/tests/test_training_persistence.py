from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.training.checkpoint import save_checkpoint
from src.training.logging import save_training_data
from src.training.runner import run_training


def test_training_data_schema_matches_baseline(tmp_path):
    result = run_training(
        "qmix",
        algo_overrides={
            "n_episode": 1,
            "n_steps": 2,
            "num_envs": 1,
            "batch_size": 2,
            "buffer_capacity": 8,
            "learn_every": 1,
            "updates_per_learn": 1,
            "seed": 7,
            "device": "cpu",
            "start_method": "fork",
        },
        no_save=False,
        output_root=tmp_path,
    )
    assert result.output_dir is not None
    json_path = result.output_dir / "training_data.json"
    npz_path = result.output_dir / "training_data.npz"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(data.keys()) == {"algorithm", "timestamp", "config", "metrics"}
    assert data["algorithm"] == "qmix"
    assert {"n_episode", "n_steps", "artifact_kind", "env_summary"}.issubset(data["config"])
    assert data["config"]["preset"] is None
    assert set(data["metrics"].keys()) == {"reward", "success_rate", "energy", "jump"}
    for key in ("reward", "success_rate", "energy", "jump"):
        assert isinstance(data["metrics"][key], list)
        assert len(data["metrics"][key]) == 1
        assert isinstance(data["metrics"][key][0], float)

    npz = np.load(npz_path)
    assert set(npz.files) == {"reward", "success_rate", "energy", "jump"}
    assert npz["reward"].dtype == np.float32


def test_training_data_records_baseline_preset_metadata(tmp_path):
    result = run_training(
        "qmix",
        preset="qmix_plain_baseline",
        algo_overrides={
            "n_episode": 1,
            "n_steps": 2,
            "num_envs": 1,
            "batch_size": 2,
            "buffer_capacity": 8,
            "learn_every": 1,
            "updates_per_learn": 1,
            "seed": 13,
            "device": "cpu",
            "start_method": "fork",
        },
        no_save=False,
        output_root=tmp_path,
    )
    assert result.output_dir is not None
    data = json.loads((result.output_dir / "training_data.json").read_text(encoding="utf-8"))
    preset = data["config"]["preset"]
    assert preset["name"] == "qmix_plain_baseline"
    assert preset["path"].endswith("qmix_plain_baseline.yaml")
    assert len(preset["sha256"]) == 64
    assert preset["source"] == "Main/train/train_qmix.py@2360ab92"


def test_save_training_data_eval_kind_switches_filename_and_rejects_unknown_kind(tmp_path):
    json_path, npz_path, data_dir = save_training_data(
        algorithm="qmix",
        reward_history=[1.0],
        success_rate_history=[0.5],
        energy_history=[0.1],
        jump_history=[0.2],
        n_episode=1,
        n_steps=2,
        output_root=tmp_path,
        artifact_kind="eval",
    )
    assert json_path == data_dir / "evaluation_data.json"
    assert npz_path == data_dir / "evaluation_data.npz"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["config"]["artifact_kind"] == "eval"

    with pytest.raises(ValueError, match="artifact_kind"):
        save_training_data(
            algorithm="bad",
            reward_history=[],
            success_rate_history=[],
            energy_history=[],
            jump_history=[],
            n_episode=1,
            n_steps=1,
            output_root=tmp_path,
            artifact_kind="report",
        )


def test_checkpoint_contains_dqn_mappo_and_callback_keys(tmp_path):
    qmix_result = run_training(
        "qmix",
        algo_overrides={
            "n_episode": 1,
            "n_steps": 2,
            "num_envs": 1,
            "batch_size": 2,
            "buffer_capacity": 8,
            "learn_every": 1,
            "updates_per_learn": 1,
            "seed": 8,
            "device": "cpu",
            "start_method": "fork",
        },
        no_save=True,
    )

    class _WorldModelCallback:
        name = "wm_alternating"

        def state_dict(self):
            return {
                "wm_state_dict": {},
                "opt_state_dict": {},
                "wm_cfg": {},
                "td_cfg": {},
            }

    ckpt_path = save_checkpoint(
        path=tmp_path / "qmix_weights.pth",
        algorithm="qmix",
        trainer=qmix_result.trainer,
        callbacks=[_WorldModelCallback()],
    )
    qmix_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert {"agents", "mixer", "target_mixer", "callbacks"}.issubset(qmix_ckpt)
    assert set(qmix_ckpt["callbacks"]["wm_alternating"]) == {
        "wm_state_dict",
        "opt_state_dict",
        "wm_cfg",
        "td_cfg",
    }

    mappo_result = run_training(
        "mappo",
        algo_overrides={
            "n_episode": 1,
            "n_steps": 2,
            "seed": 9,
            "device": "cpu",
            "minibatch_size": 8,
        },
        no_save=True,
    )
    mappo_path = save_checkpoint(
        path=tmp_path / "mappo_weights.pth",
        algorithm="mappo",
        trainer=mappo_result.trainer,
        callbacks=[],
    )
    mappo_ckpt = torch.load(mappo_path, map_location="cpu", weights_only=True)
    assert {"actor", "critic", "actor_opt", "critic_opt", "callbacks"}.issubset(mappo_ckpt)
    assert mappo_ckpt["callbacks"] == {}
