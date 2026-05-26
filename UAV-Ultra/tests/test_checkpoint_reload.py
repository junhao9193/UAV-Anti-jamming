from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.algorithms import build_trainer
from src.config.loader import load_algo_config, load_env_config
from src.training.callbacks import build_callbacks
from src.training.checkpoint import (
    load_callback_states,
    load_trainer_state_dict,
    save_checkpoint,
    trainer_state_dict,
)


def _algo_overrides(name: str) -> dict:
    if name == "mappo":
        return {"seed": 3, "device": "cpu", "minibatch_size": 8}
    return {
        "n_episode": 1,
        "n_steps": 2,
        "num_envs": 1,
        "batch_size": 2,
        "buffer_capacity": 8,
        "seed": 3,
        "device": "cpu",
        "start_method": "fork",
    }


def _build(name: str):
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(name, overrides=_algo_overrides(name), env_cfg=env_cfg)
    return env_cfg, algo_cfg, build_trainer(name, env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")


@pytest.mark.parametrize("algorithm", ["iql", "vdn", "qmix", "qplex"])
def test_load_trainer_state_dict_restores_dqn_family(tmp_path, algorithm):
    env_cfg, algo_cfg, trainer = _build(algorithm)
    path = save_checkpoint(
        path=tmp_path / f"{algorithm}_weights.pth",
        algorithm=algorithm,
        trainer=trainer,
        callbacks=[],
    )
    _, _, fresh = _build(algorithm)
    checkpoint = load_trainer_state_dict(fresh, path, algorithm, device="cpu")

    for src_agent, dst_agent in zip(trainer.agents, fresh.agents):
        for name in ("actor", "q_net", "target_actor", "target_q_net"):
            src = getattr(src_agent, name).state_dict()
            dst = getattr(dst_agent, name).state_dict()
            for key in src:
                torch.testing.assert_close(dst[key], src[key])
    if hasattr(trainer, "mixer"):
        for key, value in trainer.mixer.state_dict().items():
            torch.testing.assert_close(fresh.mixer.state_dict()[key], value)
    assert checkpoint["algorithm"] == algorithm


def test_load_trainer_state_dict_restores_mappo_and_optimizers(tmp_path):
    _, _, trainer = _build("mappo")
    path = save_checkpoint(
        path=tmp_path / "mappo_weights.pth",
        algorithm="mappo",
        trainer=trainer,
        callbacks=[],
    )
    _, _, fresh = _build("mappo")
    load_trainer_state_dict(fresh, path, "mappo", device="cpu", load_optimizers=True)

    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(fresh.actor.state_dict()[key], value)
    for key, value in trainer.critic.state_dict().items():
        torch.testing.assert_close(fresh.critic.state_dict()[key], value)
    assert fresh.actor_opt.state_dict()["param_groups"][0]["lr"] == pytest.approx(
        trainer.actor_opt.state_dict()["param_groups"][0]["lr"]
    )


def test_load_trainer_state_dict_strict_rejects_stage4_or_mismatched_checkpoint():
    _, _, trainer = _build("qmix")
    checkpoint = trainer_state_dict(trainer, algorithm="qmix")
    with pytest.raises(ValueError, match="callbacks"):
        load_trainer_state_dict(trainer, checkpoint, "qmix", device="cpu")

    checkpoint["callbacks"] = {}
    bad_algorithm = deepcopy(checkpoint)
    bad_algorithm["algorithm"] = "vdn"
    with pytest.raises(ValueError, match="algorithm"):
        load_trainer_state_dict(trainer, bad_algorithm, "qmix", device="cpu")

    bad_agent_cfg = deepcopy(checkpoint)
    bad_agent_cfg["agent_config"]["state_dim"] += 1
    with pytest.raises(ValueError, match="agent_config.state_dim"):
        load_trainer_state_dict(trainer, bad_agent_cfg, "qmix", device="cpu")

    bad_mixer_cfg = deepcopy(checkpoint)
    bad_mixer_cfg["mixer_config"]["global_state_dim"] += 1
    with pytest.raises(ValueError, match="mixer_config.global_state_dim"):
        load_trainer_state_dict(trainer, bad_mixer_cfg, "qmix", device="cpu")


def test_world_model_callback_load_validates_wm_and_td_cfg():
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={
            "callbacks": ["value_expansion", "wm_concurrent"],
            "batch_size": 2,
            "buffer_capacity": 8,
            "device": "cpu",
        },
        env_cfg=env_cfg,
    )
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    manager = build_callbacks(algo_cfg.callbacks, env_cfg=env_cfg, algo_cfg=algo_cfg)
    manager.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    state = manager.state_dict()

    load_callback_states(list(manager), state, strict=True)

    bad = deepcopy(state)
    bad["wm_concurrent"]["td_cfg"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="td_cfg mismatch"):
        load_callback_states(list(manager), bad, strict=True)

    bad = deepcopy(state)
    bad["wm_concurrent"]["wm_cfg"].pop("hidden_dim")
    with pytest.raises(ValueError, match="missing in saved"):
        load_callback_states(list(manager), bad, strict=True)


def test_checkpoint_callback_key_alias_normalize_for_legacy_wm_alternating():
    """Stage 5/6 旧 checkpoint 用 'wm_alternating' 作 key；Stage 7 应 strict-load 通过。"""
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={
            "callbacks": ["value_expansion", "wm_concurrent"],
            "batch_size": 2,
            "buffer_capacity": 8,
            "device": "cpu",
        },
        env_cfg=env_cfg,
    )
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    manager = build_callbacks(algo_cfg.callbacks, env_cfg=env_cfg, algo_cfg=algo_cfg)
    manager.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    # 生成新 state 后改 key 模拟旧 checkpoint
    state = manager.state_dict()
    legacy = dict(state)
    legacy["wm_alternating"] = legacy.pop("wm_concurrent")

    with pytest.warns(FutureWarning, match="wm_alternating"):
        load_callback_states(list(manager), legacy, strict=True)


# ------------------------------------------------------------------
# Stage 8 — JP-on/off checkpoint strict 双向校验 + jammer_predictor_opt 持久化
# ------------------------------------------------------------------


def _build_qmix(callbacks: list[str]):
    env_cfg = load_env_config()
    algo_cfg = load_algo_config(
        "qmix",
        overrides={
            "callbacks": callbacks,
            "batch_size": 4,
            "buffer_capacity": 16,
            "device": "cpu",
        },
        env_cfg=env_cfg,
    )
    return env_cfg, algo_cfg, build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")


def test_jp_checkpoint_roundtrip_restores_predictor_and_target_and_opt(tmp_path):
    """JP-on trainer save → 同 cfg 新 trainer reload，predictor + target + optimizer state 字段级一致。"""
    import torch as _t

    _, _, trainer = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    # 故意修改 predictor 参数让 save 不是初始值
    with _t.no_grad():
        for p in trainer.agents[0].jammer_predictor.parameters():
            p.add_(0.1)
    path = save_checkpoint(
        path=tmp_path / "qmix_jp.pth",
        algorithm="qmix",
        trainer=trainer,
        callbacks=[],
    )
    _, _, fresh = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    load_trainer_state_dict(fresh, path, "qmix", device="cpu", load_optimizers=True)
    for src_agent, dst_agent in zip(trainer.agents, fresh.agents):
        for key, val in src_agent.jammer_predictor.state_dict().items():
            torch.testing.assert_close(dst_agent.jammer_predictor.state_dict()[key], val)
        for key, val in src_agent.target_jammer_predictor.state_dict().items():
            torch.testing.assert_close(dst_agent.target_jammer_predictor.state_dict()[key], val)


def test_jp_checkpoint_strict_rejects_jp_off_trainer_loading_jp_on_checkpoint(tmp_path):
    """JP-on checkpoint × JP-off trainer → strict raise（jammer_prediction_enabled mismatch）。"""
    _, _, jp_trainer = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    path = save_checkpoint(
        path=tmp_path / "qmix_jp.pth",
        algorithm="qmix",
        trainer=jp_trainer,
        callbacks=[],
    )
    _, _, plain_trainer = _build_qmix([])
    with pytest.raises(ValueError, match="jammer_prediction_enabled"):
        load_trainer_state_dict(plain_trainer, path, "qmix", device="cpu")


def test_jp_checkpoint_strict_rejects_jp_on_trainer_loading_jp_off_checkpoint(tmp_path):
    """JP-off checkpoint × JP-on trainer → strict raise（反向）。"""
    _, _, plain_trainer = _build_qmix([])
    path = save_checkpoint(
        path=tmp_path / "qmix_plain.pth",
        algorithm="qmix",
        trainer=plain_trainer,
        callbacks=[],
    )
    _, _, jp_trainer = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    with pytest.raises(ValueError, match="jammer_prediction_enabled"):
        load_trainer_state_dict(jp_trainer, path, "qmix", device="cpu")


def test_jp_checkpoint_strict_rejects_history_len_mismatch(tmp_path):
    """JP-on checkpoint × JP-on trainer，history_len 不同 → strict raise。"""
    import copy as _copy
    _, _, trainer = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    path = save_checkpoint(
        path=tmp_path / "qmix_jp.pth",
        algorithm="qmix",
        trainer=trainer,
        callbacks=[],
    )
    # load checkpoint dict 然后改 history_len 模拟 mismatch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    ckpt["agent_config"]["jammer_history_len"] = int(ckpt["agent_config"]["jammer_history_len"]) + 1
    _, _, fresh = _build_qmix(["value_expansion", "wm_block_alternating", "jammer_prediction"])
    with pytest.raises(ValueError, match="agent_config.jammer_history_len"):
        load_trainer_state_dict(fresh, ckpt, "qmix", device="cpu")
