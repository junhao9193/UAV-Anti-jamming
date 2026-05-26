"""Stage 8 JammerAwareMPDQNAgent 单测（plan §test_plan）。

关键覆盖：
- ``agent.state_dim`` 保持 **raw**（baseline 同款）。
- actor/q_net 第一层 ``in_features = raw + n_channel``（JP-aware 子类内部消费）。
- 梯度边界：augment_state 后从 augmented 反向 backward，predictor 参数无梯度（detach 起效）；
  单独 BCE 路径有梯度。
- target_jammer_predictor backward 后无梯度（requires_grad=False，不靠静态断言）。
- sensing_history=None 走 ``_default_history`` 兜底，不 raise。
- ``set_feature_scale(scale)`` 后 augmented 末尾 n_channel 列 == scale × probs.detach()。
"""

from __future__ import annotations

import numpy as np
import torch

from src.algorithms.common.agents.jammer_aware_mpdqn_agent import JammerAwareMPDQNAgent


def _make_agent(*, n_channel=6, history_len=4, use_jammer_feature=True):
    return JammerAwareMPDQNAgent(
        state_dim=18,
        n_actions=4,
        param_dim=2,
        n_channel=n_channel,
        history_len=history_len,
        jammer_hidden_dim=16,
        use_jammer_feature=use_jammer_feature,
        lr_jammer=None,
        batch_size=4,
        gamma=0.99,
        lr_actor=1e-3,
        lr_q=1e-3,
        target_update_interval=100,
        use_amp=False,
        max_grad_norm=10.0,
        device="cpu",
    )


def test_agent_state_dim_is_raw_and_jp_metadata_exposed():
    agent = _make_agent(n_channel=6, history_len=4)
    assert agent.state_dim == 18          # ★ raw, baseline-faithful
    assert agent.n_channel == 6
    assert agent.jammer_history_len == 4


def test_actor_q_net_first_linear_in_features_is_raw_plus_n_channel():
    agent = _make_agent(n_channel=6)
    # actor: JammerAwareMPDQNActor.net[0]: Linear(raw + n_channel, hidden)
    assert agent.actor.net[0].in_features == 18 + 6
    # q_net.state_encoder[0]: Linear(raw + n_channel, hidden)
    assert agent.q_net.state_encoder[0].in_features == 18 + 6
    # target nets 同样
    assert agent.target_actor.net[0].in_features == 18 + 6
    assert agent.target_q_net.state_encoder[0].in_features == 18 + 6


def test_target_jammer_predictor_params_require_grad_false():
    agent = _make_agent()
    for p in agent.target_jammer_predictor.parameters():
        assert p.requires_grad is False
    # target actor/q_net 同处理
    for p in agent.target_actor.parameters():
        assert p.requires_grad is False
    for p in agent.target_q_net.parameters():
        assert p.requires_grad is False


def test_augment_state_default_history_fallback_when_history_is_none():
    agent = _make_agent(n_channel=6, history_len=4)
    state = torch.randn(3, 18)
    aug, logits, probs = agent.augment_state(state, None, target=False)
    assert aug.shape == (3, 18 + 6)
    assert logits.shape == (3, 6)
    assert probs.shape == (3, 6)


def test_augment_state_feature_scale_zero_yields_zero_appended_columns():
    agent = _make_agent(n_channel=6)
    agent.set_feature_scale(0.0)
    state = torch.randn(2, 18)
    history = torch.randn(2, 4, 6)
    aug, _, probs = agent.augment_state(state, history, target=False)
    # 末尾 6 列 == probs.detach() * 0 == 0
    torch.testing.assert_close(aug[:, 18:], torch.zeros_like(probs), rtol=0.0, atol=1e-12)


def test_augment_state_feature_scale_matches_detached_probs_times_scale():
    agent = _make_agent(n_channel=6)
    agent.set_feature_scale(0.5)
    state = torch.randn(2, 18)
    history = torch.randn(2, 4, 6)
    aug, _, probs = agent.augment_state(state, history, target=False)
    expected = 0.5 * probs.detach()
    torch.testing.assert_close(aug[:, 18:], expected, rtol=0.0, atol=1e-12)


def test_augment_state_use_feature_false_keeps_appended_zero():
    agent = _make_agent(n_channel=6, use_jammer_feature=False)
    agent.set_feature_scale(1.0)
    state = torch.randn(2, 18)
    history = torch.randn(2, 4, 6)
    aug, _, _ = agent.augment_state(state, history, target=False)
    torch.testing.assert_close(aug[:, 18:], torch.zeros(2, 6), rtol=0.0, atol=1e-12)


def test_grad_boundary_q_actor_path_does_not_train_predictor():
    """关键回归：Q/actor loss 通过 feature（probs.detach()）不应回传到 predictor。"""
    agent = _make_agent(n_channel=6)
    agent.set_feature_scale(1.0)
    # state 要求 grad 以便 backward 能 flow；history 不要求 grad 但 predictor 内部 forward
    # 仍会 build graph（除非走 detach 边界）。
    state = torch.randn(4, 18, requires_grad=True)
    history = torch.randn(4, 4, 6)
    for p in agent.jammer_predictor.parameters():
        if p.grad is not None:
            p.grad.zero_()
    aug, _, _ = agent.augment_state(state, history, target=False)
    # 通过 actor forward（含 actor 参数）才能让 (aug**2).sum() 真正 backward。
    # 但更直接：让 surrogate 取决于 aug；state.requires_grad=True → backward 可 flow。
    surrogate = (aug ** 2).sum()
    surrogate.backward()
    # predictor 参数 grad 必须全 None 或全零（detach 起效）
    for p in agent.jammer_predictor.parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad)), (
                "Q/actor loss must not flow back to jammer_predictor (probs must be detached)"
            )
    # state 一端应有梯度（验证 backward path 真的跑了）
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0.0


def test_grad_boundary_bce_logits_path_does_train_predictor():
    """关键回归：BCE aux loss 路径应让 predictor 参数有非零梯度。"""
    agent = _make_agent(n_channel=6)
    state = torch.randn(4, 18)
    history = torch.randn(4, 4, 6)
    for p in agent.jammer_predictor.parameters():
        if p.grad is not None:
            p.grad.zero_()
    _, logits, _ = agent.augment_state(state, history, target=False)
    target = torch.rand(4, 6)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    grads_nonzero = [
        (p.grad is not None) and (p.grad.abs().max().item() > 0.0)
        for p in agent.jammer_predictor.parameters()
    ]
    assert all(grads_nonzero), "BCE on logits must produce non-zero predictor gradients"


def test_target_jammer_predictor_backward_does_not_produce_gradients():
    """target 路径 backward 后 target_jammer_predictor 参数无梯度（requires_grad=False 运行时验证）。"""
    agent = _make_agent(n_channel=6)
    agent.set_feature_scale(1.0)
    state = torch.randn(4, 18, requires_grad=True)
    history = torch.randn(4, 4, 6)
    aug, _, _ = agent.augment_state(state, history, target=True)
    surrogate = (aug ** 2).sum()
    surrogate.backward()
    for p in agent.target_jammer_predictor.parameters():
        # requires_grad=False 时 backward 不会写 grad，p.grad 应保持 None
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_select_action_batch_sensing_history_none_falls_back_to_default():
    agent = _make_agent(n_channel=6)
    states = np.random.randn(5, 18).astype(np.float32)
    ad, ap = agent.select_action_batch(states, epsilon=0.0)  # 不传 sensing_history
    assert ad.shape == (5,)
    assert ap.shape == (5, 4 * 2)
