from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.algorithms.world_model.action_encoding import exec_action_dim
from src.algorithms.world_model.model import JointWorldModelConfig
from src.algorithms.world_model.value_expansion import TDlambdaConfig
from src.config import specs
from src.config.loader import load_env_config
from src.training.callbacks import build_callbacks, canonicalize_callback_names
from src.training.callbacks.critic_stable import (
    _DISABLED_HARD_TARGET_SYNC_INTERVAL,
    CriticStableCallback,
    soft_update_module,
)
from src.training.callbacks.jammer_prediction import jammer_target_from_info
from src.training.callbacks.policy_mobility import PolicyMobilityCallback
from src.training.callbacks.value_expansion import ValueExpansionCallback, tensor_batch_from_numpy
from src.training.callbacks.wm_block_alternating import WMBlockAlternatingCallback
from src.training.callbacks.wm_concurrent import WMConcurrentCallback, _vc_eta


def test_callback_names_use_canonical_order_and_dependency():
    assert canonicalize_callback_names(["critic_stable", "policy_mobility"]) == [
        "policy_mobility",
        "critic_stable",
    ]
    with pytest.raises(ValueError, match="requires 'value_expansion'"):
        canonicalize_callback_names(["wm_concurrent"])
    with pytest.raises(ValueError, match="requires 'value_expansion'"):
        canonicalize_callback_names(["wm_block_alternating"])
    with pytest.raises(ValueError, match="value_expansion callback requires exactly one"):
        canonicalize_callback_names(["value_expansion"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        canonicalize_callback_names(["wm_concurrent", "wm_block_alternating", "value_expansion"])


def test_canonicalize_aliases_wm_alternating_with_future_warning():
    with pytest.warns(FutureWarning, match="wm_alternating"):
        out = canonicalize_callback_names(["wm_alternating", "value_expansion"])
    assert out == ["value_expansion", "wm_concurrent"]
    # alias 与新名同时出现 → 视为 duplicate
    with pytest.raises(ValueError, match="duplicates"):
        canonicalize_callback_names(["wm_alternating", "wm_concurrent", "value_expansion"])


def test_wm_alternating_module_shim_emits_future_warning_and_aliases_concurrent():
    import importlib
    import sys
    import warnings as _warnings
    sys.modules.pop("src.training.callbacks.wm_alternating", None)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        mod = importlib.import_module("src.training.callbacks.wm_alternating")
        assert any(issubclass(item.category, FutureWarning) for item in w)
    assert mod.WMAlternatingCallback is WMConcurrentCallback


def test_policy_mobility_default_noop_and_policy_zero_delta():
    default_cfg = load_env_config()
    base_dim = specs.total_param_dim(default_cfg)
    actions = [(0, np.ones((base_dim,), dtype=np.float32))]
    assert PolicyMobilityCallback(env_cfg=default_cfg).on_action_selected(actions) is actions

    policy_cfg = load_env_config(
        overrides={
            "uav_mobility_control": "policy",
            "uav_velocity_delta_max": 1.0,
            "uav_direction_delta_max": 0.1,
            "uav_p_delta_max": 0.05,
        }
    )
    adapted = PolicyMobilityCallback(env_cfg=policy_cfg).on_action_selected(actions)
    assert adapted[0][1].shape == (base_dim + 3,)
    np.testing.assert_allclose(adapted[0][1][-3:], np.zeros((3,), dtype=np.float32))


def test_jammer_prediction_prefers_multi_hot_and_falls_back_to_channels():
    target = jammer_target_from_info(
        {
            "jammer_channels_current_multi_hot": [0.0, 1.0, 0.0],
            "jammer_channels_current": [2],
        },
        n_channel=3,
    )
    np.testing.assert_array_equal(target, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    fallback = jammer_target_from_info({"jammer_channels_current": [0, 2]}, n_channel=3)
    np.testing.assert_array_equal(fallback, np.asarray([1.0, 0.0, 1.0], dtype=np.float32))


def test_critic_stable_soft_update_and_lr_scale():
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(1.0)
        target.weight.zero_()
    soft_update_module(target, source, tau=0.25)
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 0.25))

    class _Agent:
        def __init__(self):
            self.actor = nn.Linear(2, 1)
            self.target_actor = nn.Linear(2, 1)
            self.q_net = nn.Linear(2, 1)
            self.target_q_net = nn.Linear(2, 1)
            self.jammer_predictor = nn.Linear(2, 1)
            self.target_jammer_predictor = nn.Linear(2, 1)
            self.actor_opt = torch.optim.SGD(self.actor.parameters(), lr=0.1)
            self.q_opt = torch.optim.SGD(self.q_net.parameters(), lr=0.2)
            self.jammer_predictor_opt = torch.optim.SGD(self.jammer_predictor.parameters(), lr=0.3)

    trainer = SimpleNamespace(agents=[_Agent()], target_update_interval=7)
    cb = CriticStableCallback(tau=0.1, lr_scale=0.5)
    cb.attach(trainer=trainer, env_cfg=None, algo_cfg=None, n_envs=1)
    assert trainer.target_update_interval == _DISABLED_HARD_TARGET_SYNC_INTERVAL
    assert trainer.agents[0].actor_opt.param_groups[0]["lr"] == pytest.approx(0.05)
    assert trainer.agents[0].q_opt.param_groups[0]["lr"] == pytest.approx(0.1)
    assert trainer.agents[0].jammer_predictor_opt.param_groups[0]["lr"] == pytest.approx(0.15)
    assert cb.state_dict()["original_target_update_interval"] == 7
    cb.restore_target_sync(trainer)
    assert trainer.target_update_interval == 7


def test_critic_stable_lr_decay_and_missing_new_state_keys():
    from src.training.callbacks.base import TrainHookContext

    class _Agent:
        def __init__(self):
            self.actor = nn.Linear(2, 1)
            self.target_actor = nn.Linear(2, 1)
            self.q_net = nn.Linear(2, 1)
            self.target_q_net = nn.Linear(2, 1)
            self.actor_opt = torch.optim.SGD(self.actor.parameters(), lr=0.1)
            self.q_opt = torch.optim.SGD(self.q_net.parameters(), lr=0.2)

    trainer = SimpleNamespace(agents=[_Agent()], target_update_interval=7)
    cb = CriticStableCallback(
        tau=0.1,
        lr_scale=1.0,
        lr_decay_enabled=True,
        lr_decay_start_ep=10,
        lr_decay_end_ep=20,
        lr_decay_min=0.1,
    )
    cb.attach(trainer=trainer, env_cfg=None, algo_cfg=None, n_envs=1)

    assert cb.should_skip_q_update(TrainHookContext(trainer=trainer, episode=0, step=0)) is False
    assert trainer.agents[0].actor_opt.param_groups[0]["lr"] == pytest.approx(0.1)
    assert cb.should_skip_q_update(TrainHookContext(trainer=trainer, episode=15, step=0)) is False
    assert trainer.agents[0].actor_opt.param_groups[0]["lr"] == pytest.approx(0.055)
    assert trainer.agents[0].q_opt.param_groups[0]["lr"] == pytest.approx(0.11)
    assert cb.should_skip_q_update(TrainHookContext(trainer=trainer, episode=20, step=0)) is False
    assert trainer.agents[0].actor_opt.param_groups[0]["lr"] == pytest.approx(0.01)

    old_state = {"tau": 0.2, "lr_scale": 0.5, "original_target_update_interval": 9}
    fresh = CriticStableCallback(lr_decay_enabled=True, lr_decay_start_ep=3)
    fresh.load_state_dict(old_state, strict=True)
    assert fresh.tau == pytest.approx(0.2)
    assert fresh.lr_scale == pytest.approx(0.5)
    assert fresh.lr_decay_enabled is True
    assert fresh.lr_decay_start_ep == 3
    with pytest.raises(ValueError, match="unexpected state keys"):
        fresh.load_state_dict({**old_state, "bogus": 1}, strict=True)


def test_value_expansion_rejects_non_qmix_trainer_and_per_agent_reward_batch():
    env_cfg = load_env_config()
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=8,
        value_expansion_alpha_model=0.5,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=4,
    )
    cb = ValueExpansionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    with pytest.raises(TypeError, match="QMIXTrainer"):
        cb.attach(
            trainer=SimpleNamespace(device=torch.device("cpu")),
            env_cfg=env_cfg,
            algo_cfg=algo_cfg,
            n_envs=1,
        )

    with pytest.raises(ValueError, match="global reward shape"):
        tensor_batch_from_numpy(
            {
                "state": np.zeros((2, 4, 18), dtype=np.float32),
                "action_discrete": np.zeros((2, 4), dtype=np.int64),
                "action_params": np.zeros((2, 4, 72), dtype=np.float32),
                "reward": np.zeros((2, 4), dtype=np.float32),
                "next_state": np.zeros((2, 4, 18), dtype=np.float32),
                "done": np.zeros((2,), dtype=np.float32),
            },
            device=torch.device("cpu"),
        )


def _populate_wm_replay(cb, env_cfg, global_state_dim: int, n_agents: int, n_steps: int = 4) -> None:
    base_param_dim = specs.total_param_dim(env_cfg)
    for t in range(n_steps):
        state = np.full((global_state_dim,), float(t), dtype=np.float32)
        next_state = np.full((global_state_dim,), float(t + 1), dtype=np.float32)
        cb.wm_replay.add(
            env_id=0,
            state=state,
            action_discrete=np.zeros((n_agents,), dtype=np.int64),
            action_params=np.zeros((n_agents, base_param_dim), dtype=np.float32),
            reward_team=1.0,
            next_state=next_state,
            done=False,
        )


def _build_concurrent_cb(env_cfg, *, vc_eta_max=0.0, vc_warmup=0, vc_ramp=1):
    n_agents = int(env_cfg.n_ch)
    global_state_dim = n_agents * specs.state_dim(env_cfg)
    action_dim = exec_action_dim(
        n_agents=n_agents,
        n_des=specs.n_des(env_cfg),
        n_channel=env_cfg.n_channel,
        param_dim=specs.param_dim_per_action(env_cfg),
    )
    shared = {
        "wm_cfg": JointWorldModelConfig(
            state_dim=global_state_dim,
            action_dim=action_dim,
            hidden_dim=8,
            stochastic_dim=4,
        ),
        "td_cfg": TDlambdaConfig(gamma=0.99, lam=0.8, rollout_k=1),
    }
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=16,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=1,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=float(vc_eta_max),
        wm_vc_warmup_ep=int(vc_warmup),
        wm_vc_ramp_end_ep=int(vc_ramp),
    )
    trainer = SimpleNamespace(device=torch.device("cpu"), batch_size=1)
    cb = WMConcurrentCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, shared=shared, seq_len=4, lr=1e-3)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    _populate_wm_replay(cb, env_cfg, global_state_dim, n_agents)
    return cb, trainer


def test_wm_concurrent_update_is_finite_and_state_keys_present():
    env_cfg = load_env_config()
    cb, _ = _build_concurrent_cb(env_cfg)
    result = cb.update_world_model_once(batch_size=1, episode=0)
    assert result is not None
    assert np.isfinite(result["wm_loss"])
    assert "wm_L_VC" not in result   # eta=0 → no L_VC
    assert set(cb.state_dict().keys()) == {"wm_state_dict", "opt_state_dict", "wm_cfg", "td_cfg"}


def test_wm_concurrent_skips_update_when_trainer_step_returns_none():
    env_cfg = load_env_config()
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=16,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=1,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=0.0,
        wm_vc_warmup_ep=0,
        wm_vc_ramp_end_ep=1,
    )
    cb = WMConcurrentCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    calls = []
    cb._run_wm_updates = lambda *, episode: calls.append(episode) or {"wm_loss": 1.0}

    cb.after_train_step(SimpleNamespace(trainer=SimpleNamespace(batch_size=3), episode=2, step=0), result=None)
    assert calls == []

    cb.after_train_step(
        SimpleNamespace(trainer=SimpleNamespace(batch_size=3), episode=2, step=0),
        result={"loss_q": 1.0, "loss_actor": 2.0},
    )
    assert calls == [2]


def test_wm_concurrent_curriculum_wm_only_phase_runs_without_q_result():
    env_cfg = load_env_config()
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=16,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=1,
        value_expansion_model_warmup_ep=2,
        value_expansion_ramp_start_ep=4,
        value_expansion_ramp_end_ep=6,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=0.2,
        wm_vc_warmup_ep=100,
        wm_vc_ramp_end_ep=200,
    )
    cb = WMConcurrentCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, curriculum_active=True)
    calls = []
    cb._run_wm_updates = lambda *, episode: calls.append(episode) or {"wm_loss": 1.0}

    cb.after_train_step(SimpleNamespace(trainer=SimpleNamespace(batch_size=3), episode=1, step=0), result=None)
    assert calls == []
    cb.after_train_step(SimpleNamespace(trainer=SimpleNamespace(batch_size=3), episode=2, step=0), result=None)
    assert calls == [2]
    cb.after_train_step(SimpleNamespace(trainer=SimpleNamespace(batch_size=3), episode=4, step=0), result=None)
    assert calls == [2]
    assert cb._eta_for_episode(3) == pytest.approx(0.0)
    assert cb._eta_for_episode(5) == pytest.approx(0.1)
    assert cb._eta_for_episode(6) == pytest.approx(0.2)


def test_wm_concurrent_records_every_wm_update_result():
    env_cfg = load_env_config()
    cb, _ = _build_concurrent_cb(env_cfg)
    cb.wm_updates_per_learn = 3

    calls = []

    def _fake_update(*, batch_size, episode):
        calls.append((batch_size, episode))
        return {"wm_loss": float(len(calls))}

    cb.update_world_model_once = _fake_update
    result = cb._run_wm_updates(episode=7)

    assert calls == [(1, 7), (1, 7), (1, 7)]
    assert result == {"wm_loss": 3.0}
    assert cb.last_wm_result == [
        {"wm_loss": 1.0},
        {"wm_loss": 2.0},
        {"wm_loss": 3.0},
    ]


def test_vc_eta_ramp_endpoints_and_midpoint():
    assert _vc_eta(0, warmup_ep=300, ramp_end_ep=800, v_max=0.2) == 0.0
    assert _vc_eta(299, warmup_ep=300, ramp_end_ep=800, v_max=0.2) == 0.0
    assert _vc_eta(550, warmup_ep=300, ramp_end_ep=800, v_max=0.2) == pytest.approx(0.5 * 0.2)
    assert _vc_eta(800, warmup_ep=300, ramp_end_ep=800, v_max=0.2) == pytest.approx(0.2)
    assert _vc_eta(1000, warmup_ep=300, ramp_end_ep=800, v_max=0.2) == pytest.approx(0.2)


def _build_constant_value_teacher(env_cfg):
    n_agents = int(env_cfg.n_ch)
    base_param_dim = int(specs.total_param_dim(env_cfg))

    class _Teacher:
        def __init__(self):
            self.env_cfg = env_cfg
        def q_tot_target(self, s, ad, ap):
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        def greedy_action(self, s_flat):
            bsz = int(s_flat.shape[0])
            ad = torch.zeros((bsz, n_agents), dtype=torch.long, device=s_flat.device)
            ap = torch.zeros((bsz, n_agents, base_param_dim), dtype=s_flat.dtype, device=s_flat.device)
            return ad, ap
    return _Teacher()


def test_wm_concurrent_l_vc_path_is_finite_and_emits_l_vc_field():
    env_cfg = load_env_config()
    cb, trainer = _build_concurrent_cb(env_cfg, vc_eta_max=0.5, vc_warmup=0, vc_ramp=1)
    # 装 value_teacher（用常量 teacher）+ 一个 dummy _clip_value_target
    cb.value_teacher = _build_constant_value_teacher(env_cfg)
    trainer._clip_value_target = lambda x: torch.clamp(x, -1000.0, 1000.0)
    result = cb.update_world_model_once(batch_size=1, episode=10)
    assert result is not None
    assert np.isfinite(result["wm_loss"])
    assert "wm_L_VC" in result
    assert result["wm_eta"] == pytest.approx(0.5)


def test_build_callbacks_ignores_user_order():
    env_cfg = load_env_config()
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=8,
        value_expansion_alpha_model=0.5,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=4,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=0.0,
        wm_vc_warmup_ep=0,
        wm_vc_ramp_end_ep=1,
        wm_block_qmix_episodes=2,
        wm_block_wm_episodes=2,
    )
    manager = build_callbacks(
        ["critic_stable", "wm_concurrent", "value_expansion", "policy_mobility"],
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
    )
    assert [cb.name for cb in manager] == [
        "policy_mobility",
        "value_expansion",
        "wm_concurrent",
        "critic_stable",
    ]
    assert manager.callbacks[1].curriculum_active is True
    assert manager.callbacks[2].curriculum_active is True
    # block alternating 也按 canonical 顺序
    manager2 = build_callbacks(
        ["critic_stable", "wm_block_alternating", "value_expansion"],
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
    )
    assert [cb.name for cb in manager2] == [
        "value_expansion",
        "wm_block_alternating",
        "critic_stable",
    ]
    assert manager2.callbacks[0].curriculum_active is False


def test_value_expansion_curriculum_boundaries_and_static_block_mode():
    from src.training.callbacks.base import TrainHookContext

    env_cfg = load_env_config()
    algo_cfg = SimpleNamespace(
        value_expansion_alpha_model=0.5,
        value_expansion_seq_len=4,
        value_expansion_model_warmup_ep=2,
        value_expansion_ramp_start_ep=4,
        value_expansion_ramp_end_ep=6,
        value_expansion_alpha_model_max=0.2,
    )
    cb = ValueExpansionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, curriculum_active=True)
    assert cb._effective_alpha(0) == pytest.approx(0.0)
    assert cb._effective_alpha(4) == pytest.approx(0.0)
    assert cb._effective_alpha(5) == pytest.approx(0.1)
    assert cb._effective_alpha(6) == pytest.approx(0.2)
    assert cb.should_skip_q_update(TrainHookContext(trainer=object(), episode=1, step=0)) is False
    assert cb.should_skip_q_update(TrainHookContext(trainer=object(), episode=2, step=0)) is True
    assert cb.should_skip_q_update(TrainHookContext(trainer=object(), episode=3, step=0)) is True
    assert cb.should_skip_q_update(TrainHookContext(trainer=object(), episode=4, step=0)) is False

    static = ValueExpansionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, curriculum_active=False)
    assert static._effective_alpha(0) == pytest.approx(0.5)
    assert static.should_skip_q_update(TrainHookContext(trainer=object(), episode=3, step=0)) is False


def test_callback_manager_phase_skip_short_circuits_q_update():
    from src.training.callbacks.base import CallbackManager, TrainHookContext, TrainingCallback

    class _SkipperCB(TrainingCallback):
        name = "skipper"
        def should_skip_q_update(self, context):
            return True
        def after_train_step(self, context, result):
            self.last_result = result

    class _EagerCB(TrainingCallback):
        name = "eager"
        def __init__(self):
            self.on_train_step_calls = 0
        def on_train_step(self, context):
            self.on_train_step_calls += 1
            return {"loss_q": 9.9}

    class _ObserverCB(TrainingCallback):
        name = "observer"
        def __init__(self):
            self.should_calls = 0
        def should_skip_q_update(self, context):
            self.should_calls += 1
            return False

    class _CountingTrainer:
        def __init__(self):
            self.train_step_calls = 0
        def train_step(self):
            self.train_step_calls += 1
            return {"loss_q": 1.1}

    skipper = _SkipperCB()
    eager = _EagerCB()
    observer = _ObserverCB()
    trainer = _CountingTrainer()
    mgr = CallbackManager([skipper, observer, eager])
    ctx = TrainHookContext(trainer=trainer, episode=0, step=0)
    out = mgr.train_step(ctx)
    assert out is None
    assert observer.should_calls == 1
    assert eager.on_train_step_calls == 0
    assert trainer.train_step_calls == 0
    assert skipper.last_result is None

    # 没有 skipper 时 eager.on_train_step 接管
    mgr2 = CallbackManager([eager])
    out2 = mgr2.train_step(ctx)
    assert out2 == {"loss_q": 9.9}
    assert eager.on_train_step_calls == 1
    assert trainer.train_step_calls == 0


def test_wm_block_alternating_phase_switch_and_skip():
    env_cfg = load_env_config()
    n_agents = int(env_cfg.n_ch)
    global_state_dim = n_agents * specs.state_dim(env_cfg)
    action_dim = exec_action_dim(
        n_agents=n_agents,
        n_des=specs.n_des(env_cfg),
        n_channel=env_cfg.n_channel,
        param_dim=specs.param_dim_per_action(env_cfg),
    )
    shared = {
        "wm_cfg": JointWorldModelConfig(
            state_dim=global_state_dim,
            action_dim=action_dim,
            hidden_dim=8,
            stochastic_dim=4,
        ),
        "td_cfg": TDlambdaConfig(gamma=0.99, lam=0.8, rollout_k=1),
    }
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=16,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=1,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=0.0,
        wm_vc_warmup_ep=0,
        wm_vc_ramp_end_ep=1,
        wm_block_qmix_episodes=2,
        wm_block_wm_episodes=2,
    )

    class _MiniAgent:
        def __init__(self):
            self.actor = nn.Linear(2, 2)
            self.q_net = nn.Linear(2, 2)
            self.target_actor = nn.Linear(2, 2)
            self.target_q_net = nn.Linear(2, 2)
            self.jammer_predictor = nn.Linear(2, 2)
            self.target_jammer_predictor = nn.Linear(2, 2)

    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        agents=[_MiniAgent()],
        mixer=nn.Linear(2, 1),
        target_mixer=nn.Linear(2, 1),
    )
    cb = WMBlockAlternatingCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, shared=shared, seq_len=4, lr=1e-3)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    _populate_wm_replay(cb, env_cfg, global_state_dim, n_agents)

    from src.training.callbacks.base import TrainHookContext

    # ep 0,1：qmix phase → 不 skip；WM eval；agents train
    for ep in (0, 1):
        ctx = TrainHookContext(trainer=trainer, episode=ep, step=0)
        assert cb.should_skip_q_update(ctx) is False
    assert cb._last_applied_phase == "qmix"
    assert cb.world_model.training is False
    assert trainer.agents[0].actor.training is True
    assert trainer.agents[0].jammer_predictor.training is True
    assert trainer.agents[0].target_jammer_predictor.training is True

    # ep 2,3：wm phase → skip；WM train；agents eval
    for ep in (2, 3):
        ctx = TrainHookContext(trainer=trainer, episode=ep, step=0)
        assert cb.should_skip_q_update(ctx) is True
    assert cb._last_applied_phase == "wm"
    assert cb.world_model.training is True
    assert trainer.agents[0].actor.training is False
    assert trainer.agents[0].jammer_predictor.training is False
    assert trainer.agents[0].target_jammer_predictor.training is False

    # ep 4：回到第一个 block，qmix phase
    ctx = TrainHookContext(trainer=trainer, episode=4, step=0)
    assert cb.should_skip_q_update(ctx) is False
    assert cb._last_applied_phase == "qmix"

    # state_dict 含扩展字段
    state = cb.state_dict()
    assert {
        "qmix_block_episodes",
        "wm_block_episodes",
        "current_episode",
        "phase_history",
        "last_applied_phase",
        "last_wm_result",
    }.issubset(state)
    assert state["current_episode"] == 4
    assert state["last_wm_result"] == []

    # Reload 后不能直接复用 last_applied_phase，否则 fresh callback 首次进入同 phase
    # 时会跳过 .train()/.eval() 切换。
    fresh = WMBlockAlternatingCallback(
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        shared={"wm_cfg": shared["wm_cfg"], "td_cfg": shared["td_cfg"]},
        seq_len=4,
        lr=1e-3,
    )
    fresh.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    fresh.load_state_dict(state, strict=True)
    assert fresh.current_episode == 4
    assert fresh._last_applied_phase is None
    fresh.world_model.train()
    assert fresh.should_skip_q_update(TrainHookContext(trainer=trainer, episode=4, step=0)) is False
    assert fresh.world_model.training is False


def test_wm_block_alternating_wm_phase_runs_wm_update_only():
    env_cfg = load_env_config()
    n_agents = int(env_cfg.n_ch)
    global_state_dim = n_agents * specs.state_dim(env_cfg)
    action_dim = exec_action_dim(
        n_agents=n_agents,
        n_des=specs.n_des(env_cfg),
        n_channel=env_cfg.n_channel,
        param_dim=specs.param_dim_per_action(env_cfg),
    )
    shared = {
        "wm_cfg": JointWorldModelConfig(
            state_dim=global_state_dim,
            action_dim=action_dim,
            hidden_dim=8,
            stochastic_dim=4,
        ),
        "td_cfg": TDlambdaConfig(gamma=0.99, lam=0.8, rollout_k=1),
    }
    algo_cfg = SimpleNamespace(
        gamma=0.99,
        buffer_capacity=16,
        value_expansion_seq_len=4,
        value_expansion_td_lambda=0.8,
        value_expansion_rollout_k=1,
        wm_batch_size=1,
        wm_updates_per_learn=1,
        wm_vc_eta_max=0.0,
        wm_vc_warmup_ep=0,
        wm_vc_ramp_end_ep=1,
        wm_block_qmix_episodes=1,
        wm_block_wm_episodes=1,
    )
    trainer = SimpleNamespace(device=torch.device("cpu"), batch_size=1, agents=[])
    cb = WMBlockAlternatingCallback(env_cfg=env_cfg, algo_cfg=algo_cfg, shared=shared, seq_len=4, lr=1e-3)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    _populate_wm_replay(cb, env_cfg, global_state_dim, n_agents)

    from src.training.callbacks.base import TrainHookContext

    wm_calls = []
    original = cb._run_wm_updates
    def _spy(*, episode):
        wm_calls.append(episode)
        return original(episode=episode)
    cb._run_wm_updates = _spy

    # qmix phase (ep 0): after_train_step 即使 result=None 也不跑 WM
    cb.should_skip_q_update(TrainHookContext(trainer=trainer, episode=0, step=0))
    cb.after_train_step(TrainHookContext(trainer=trainer, episode=0, step=0), result=None)
    assert wm_calls == []

    # wm phase (ep 1): result 是 None（被短路），但 after_train_step 跑 WM
    cb.should_skip_q_update(TrainHookContext(trainer=trainer, episode=1, step=0))
    cb.after_train_step(TrainHookContext(trainer=trainer, episode=1, step=0), result=None)
    assert wm_calls == [1]
    assert cb.current_episode == 1
    assert len(cb.state_dict()["last_wm_result"]) == 1


# ----------------------------------------------------------------------
# Stage 8 — JammerPredictionCallback
# ----------------------------------------------------------------------


def _build_jp_qmix(callbacks):
    """构造一个真实 QMIX trainer + JP callback（attach 时 trainer.agents 是 JammerAwareMPDQNAgent）。"""
    from src.algorithms import build_trainer
    from src.config.loader import load_algo_config

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
    trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
    return env_cfg, algo_cfg, trainer


def test_jp_callback_attach_requires_jp_aware_trainer():
    """plain QMIXTrainer (no jp callback in cfg) → attach raise。"""
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix([])   # plain MPDQNAgent
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    with pytest.raises(RuntimeError, match="JP-aware trainer"):
        cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)


def test_jp_callback_reset_jp_state_sets_history_and_feature_scale():
    """reset_jp_state 后 history 是 (n_envs, n_agents, H, C) repeat 初始 sensing；feature_scale 设到每个 agent。"""
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"],
    )
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=2)

    n_envs = 2
    n_agents = int(env_cfg.n_ch)
    n_channel = int(env_cfg.n_channel)
    state_dim = trainer.state_dim
    H = cb.history_len
    rng = np.random.default_rng(0)
    states = rng.standard_normal((n_envs, n_agents, state_dim)).astype(np.float32)

    # warmup_episodes 取 baseline 默认 200；ep=0 → scale=0; ep=200 → scale=1
    cb.reset_jp_state(states, episode=0)
    assert cb.current_sensing_histories.shape == (n_envs, n_agents, H, n_channel)
    expected_slice = states[..., -n_channel:]
    # 全 H 都是同一切片重复
    for h in range(H):
        np.testing.assert_allclose(cb.current_sensing_histories[:, :, h, :], expected_slice)
    for agent in trainer.agents:
        assert agent.feature_scale == 0.0

    cb.reset_jp_state(states, episode=cb.warmup_episodes)
    for agent in trainer.agents:
        assert agent.feature_scale == 1.0


def test_jp_callback_warmup_ramp_at_reset_jp_state():
    """ramp endpoints + midpoint。"""
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"],
    )
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    cb.warmup_episodes = 200  # 显式
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    states = np.zeros((1, int(env_cfg.n_ch), trainer.state_dim), dtype=np.float32)
    for ep, expected in [(0, 0.0), (100, 0.5), (200, 1.0), (300, 1.0)]:
        cb.reset_jp_state(states, episode=ep)
        for agent in trainer.agents:
            assert agent.feature_scale == pytest.approx(expected), f"ep={ep}"


def test_jp_callback_on_aux_loss_returns_weighted_bce_broadcast():
    """on_aux_loss(trainer, batch, ctx) 返回 ≈ aux_weight × mean BCE 跨 agent。"""
    from src.training.callbacks.base import TrainHookContext
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"],
    )
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)

    B = 4
    N = int(env_cfg.n_ch)
    H = cb.history_len
    C = int(env_cfg.n_channel)
    history = torch.randn(B, N, H, C)
    target = torch.rand(B, C)
    batch = {"sensing_history": history, "jammer_target": target}
    ctx = TrainHookContext(trainer=trainer, episode=0, step=0)
    out = cb.on_aux_loss(trainer, batch, ctx)
    assert out is not None
    assert torch.isfinite(out).all()
    # 手算 weighted BCE
    expected_sum = None
    for i in range(N):
        logits = trainer.agents[i].jammer_predictor(history[:, i])
        loss_i = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        expected_sum = loss_i if expected_sum is None else (expected_sum + loss_i)
    expected = float(cb.aux_weight) * expected_sum / float(N)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-6)


def test_jp_callback_on_aux_loss_missing_keys_raises_not_silent_noop():
    """active JP callback 缺 batch 字段必 raise（不静默 no-op）。"""
    from src.training.callbacks.base import TrainHookContext
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"],
    )
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    cb.attach(trainer=trainer, env_cfg=env_cfg, algo_cfg=algo_cfg, n_envs=1)
    ctx = TrainHookContext(trainer=trainer, episode=0, step=0)
    with pytest.raises(ValueError, match="missing keys"):
        cb.on_aux_loss(trainer, {"sensing_history": torch.zeros(1, 1, 1, 1)}, ctx)
    with pytest.raises(ValueError, match="batch=None"):
        cb.on_aux_loss(trainer, None, ctx)


def test_jp_callback_load_state_dict_strict_after_save():
    """non-empty state_dict 必须能 strict reload（基类 default 会 raise）。"""
    from src.training.callbacks.jammer_prediction import JammerPredictionCallback

    env_cfg, algo_cfg, trainer = _build_jp_qmix(
        ["value_expansion", "wm_block_alternating", "jammer_prediction"],
    )
    cb = JammerPredictionCallback(env_cfg=env_cfg, algo_cfg=algo_cfg)
    state = cb.state_dict()
    assert set(state) == {"history_len", "aux_weight", "warmup_episodes", "use_feature"}
    # change values, then reload
    cb.history_len = 99
    cb.aux_weight = 999.0
    cb.warmup_episodes = 999
    cb.use_feature = False
    cb.load_state_dict(state, strict=True)
    assert cb.history_len == int(state["history_len"])
    assert cb.aux_weight == float(state["aux_weight"])
    # strict reject unknown key
    bad = dict(state)
    bad["bogus"] = 1
    with pytest.raises(ValueError, match="unexpected callback state keys"):
        cb.load_state_dict(bad, strict=True)
