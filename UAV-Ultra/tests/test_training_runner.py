from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.config import specs
from src.config.loader import load_env_config
from src.training.callbacks import CallbackManager
from src.training.runner import main, run_dqn_loop, run_training
from src.training.vec_env import SubprocVecEnv


def test_dqn_loop_learns_between_step_async_and_step_wait_then_stores():
    env_cfg = load_env_config()
    n_agents = int(env_cfg.n_ch)
    state_dim = int(specs.state_dim(env_cfg))
    base_param_dim = int(specs.total_param_dim(env_cfg))
    events: list[str] = []

    class _VecEnv:
        def reset(self, p_trans=None):
            events.append("reset")
            return np.zeros((1, n_agents, state_dim), dtype=np.float32)

        def step_async(self, actions):
            events.append("step_async")
            self.waiting = True

        def step_wait(self):
            events.append("step_wait")
            self.waiting = False
            return (
                np.ones((1, n_agents, state_dim), dtype=np.float32),
                np.ones((1, n_agents), dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                [{"jammer_channels_current": [0]}],
            )

        def get_metrics(self):
            events.append("metrics")
            return (
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([float(n_agents * specs.n_des(env_cfg))], dtype=np.float32),
            )

    vecenv = _VecEnv()

    class _Agent:
        def select_action_batch(self, states, epsilon):
            return (
                np.zeros((states.shape[0],), dtype=np.int64),
                np.zeros((states.shape[0], base_param_dim), dtype=np.float32),
            )

    class _Trainer:
        def __init__(self):
            self.agents = [_Agent() for _ in range(n_agents)]
            self.batch_size = 2

        def train_step(self, *, hook_context=None):
            events.append("learn")
            assert getattr(vecenv, "waiting", False) is True
            assert "store" not in events
            return {"loss_q": 1.0, "loss_actor": 2.0}

        def store_transition_batch(self, **kwargs):
            events.append("store")
            assert events[-2] == "step_wait"

    algo_cfg = SimpleNamespace(
        n_episode=1,
        n_steps=1,
        num_envs=1,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learn_every=1,
        updates_per_learn=1,
    )

    metrics, train_results = run_dqn_loop(
        trainer=_Trainer(),
        vecenv=vecenv,
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        callbacks=CallbackManager([]),
        p_trans=None,
    )

    assert events == ["reset", "step_async", "learn", "step_wait", "store", "metrics"]
    assert train_results == [{"loss_q": 1.0, "loss_actor": 2.0}]
    assert len(metrics["reward"]) == 1


@pytest.mark.parametrize("algorithm", ["iql", "vdn", "qmix", "qplex", "mappo"])
def test_training_runner_tiny_smoke_no_save(algorithm):
    algo_overrides = {
        "n_episode": 1,
        "n_steps": 2,
        "seed": 0,
        "device": "cpu",
    }
    if algorithm != "mappo":
        algo_overrides.update(
            {
                "num_envs": 1,
                "batch_size": 2,
                "buffer_capacity": 8,
                "learn_every": 1,
                "updates_per_learn": 1,
                "start_method": "fork",
            }
        )
    else:
        algo_overrides["minibatch_size"] = 8
    result = run_training(algorithm, algo_overrides=algo_overrides, no_save=True)
    assert len(result.metrics["reward"]) == 1
    assert len(result.metrics["success_rate"]) == 1


def test_qmix_runner_smoke_executes_at_least_one_learn_step():
    result = run_training(
        "qmix",
        algo_overrides={
            "n_episode": 1,
            "n_steps": 20,
            "num_envs": 1,
            "batch_size": 2,
            "buffer_capacity": 64,
            "learn_every": 1,
            "updates_per_learn": 1,
            "seed": 123,
            "device": "cpu",
            "start_method": "fork",
        },
        no_save=True,
    )
    assert any(np.isfinite(r.get("loss_q", np.nan)) for r in result.train_results)


def test_qmix_runner_smoke_with_baseline_preset():
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
            "seed": 11,
            "device": "cpu",
            "start_method": "fork",
        },
        no_save=True,
    )
    assert len(result.metrics["reward"]) == 1


def test_training_cli_accepts_baseline_preset(capsys):
    main(
        [
            "qmix",
            "--preset",
            "qmix_plain_baseline",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--num-envs",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "12",
            "--device",
            "cpu",
            "--start-method",
            "fork",
            "--no-save",
        ]
    )
    assert "Training completed for qmix" in capsys.readouterr().out


def test_qmix_runner_smoke_with_jp_full_combo():
    """Stage 8 端到端 smoke：baseline 全开组合 [VE + block_alt + JP + CS] 跑 3 ep 不崩。"""
    result = run_training(
        "qmix",
        algo_overrides={
            "callbacks": ["value_expansion", "wm_block_alternating", "jammer_prediction", "critic_stable"],
            "n_episode": 3,
            "n_steps": 20,
            "num_envs": 2,
            "batch_size": 4,
            "buffer_capacity": 32,
            "learn_every": 1,
            "updates_per_learn": 1,
            "seed": 17,
            "device": "cpu",
            "start_method": "fork",
            "wm_block_qmix_episodes": 1,
            "wm_block_wm_episodes": 1,
            "wm_batch_size": 4,
            "wm_updates_per_learn": 1,
            "wm_vc_eta_max": 0.0,
            "wm_vc_warmup_ep": 0,
            "wm_vc_ramp_end_ep": 1,
            "jammer_warmup_episodes": 1,
        },
        no_save=True,
    )
    assert len(result.metrics["reward"]) == 3
    # Q-phase episodes 应该有非空 train_results
    assert any(np.isfinite(r.get("loss_q", np.nan)) for r in result.train_results)


def test_vecenv_close_terminates_processes_that_survive_join():
    class _Remote:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    class _Proc:
        def __init__(self):
            self.join_calls = 0
            self.terminated = False
            self.killed = False

        def join(self, timeout=None):
            self.join_calls += 1

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    vecenv = SubprocVecEnv.__new__(SubprocVecEnv)
    remote = _Remote()
    proc = _Proc()
    vecenv.closed = False
    vecenv.remotes = [remote]
    vecenv.ps = [proc]

    vecenv.close()

    assert remote.sent == [("close", None)]
    assert proc.terminated is True
    assert proc.join_calls >= 2
