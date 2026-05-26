"""值分解 mixer 回归：VDN 公式直接断言；QMIX / QPLEX state_dict 拷贝。

plan 通过标准：CPU + float64 + 质数维度 (B=7, N=3, S=12, n_heads=2) + ``strict=True``
+ ``atol=1e-12``。
"""

from __future__ import annotations

import torch

from src.algorithms.common.networks import QMIXMixer, QPLEXMixer, VDNMixer


B, N, S = 7, 3, 12  # 质数维度
HIDDEN, HYPER = 8, 16
N_HEADS = 2


def test_vdn_mixer_is_sum_along_agents_dim():
    """VDN 公式断言：Q_tot = sum_i Q_i（无可学参）。"""
    mixer = VDNMixer()
    qs = torch.randn(B, N, dtype=torch.float64)
    out = mixer(qs)
    assert out.shape == (B, 1)
    torch.testing.assert_close(out, qs.sum(dim=1, keepdim=True), rtol=0, atol=1e-12)


def test_vdn_mixer_ignores_global_state():
    """global_state 被忽略，传任意值结果一致。"""
    mixer = VDNMixer()
    qs = torch.randn(B, N, dtype=torch.float64)
    gs1 = torch.randn(B, N * S, dtype=torch.float64)
    gs2 = torch.randn(B, N * S, dtype=torch.float64)
    torch.testing.assert_close(mixer(qs, gs1), mixer(qs, gs2), rtol=0, atol=1e-12)


def test_vdn_mixer_accepts_3d_input_squeezing_last_dim():
    mixer = VDNMixer()
    qs = torch.randn(B, N, 1, dtype=torch.float64)
    out = mixer(qs)
    assert out.shape == (B, 1)


def test_qmix_mixer_state_dict_copy_regression(baseline_import):
    baseline_mod = baseline_import("algorithms.mpdqn.qmix.mixer")
    torch.manual_seed(0)

    old = baseline_mod.QMIXMixer(N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER).double().cpu().eval()
    new = QMIXMixer(N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER).double().cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    qs = torch.randn(B, N, dtype=torch.float64)
    gs = torch.randn(B, N * S, dtype=torch.float64)
    with torch.no_grad():
        out_old = old(qs, gs)
        out_new = new(qs, gs)
    assert out_old.shape == (B, 1)
    assert out_new.shape == out_old.shape
    torch.testing.assert_close(out_new, out_old, rtol=0, atol=1e-12)


def test_qmix_mixer_state_dict_keys_match_baseline(baseline_import):
    baseline_mod = baseline_import("algorithms.mpdqn.qmix.mixer")
    new = QMIXMixer(N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER)
    old = baseline_mod.QMIXMixer(N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER)
    assert set(new.state_dict().keys()) == set(old.state_dict().keys())


def test_qplex_mixer_state_dict_copy_regression(baseline_import):
    baseline_mod = baseline_import("algorithms.mpdqn.qplex.mixer")
    torch.manual_seed(0)

    old = baseline_mod.QPLEXMixer(
        N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER, n_heads=N_HEADS
    ).double().cpu().eval()
    new = QPLEXMixer(
        N, N * S, mixing_hidden_dim=HIDDEN, hypernet_hidden_dim=HYPER, n_heads=N_HEADS
    ).double().cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    qs = torch.randn(B, N, dtype=torch.float64)
    max_qs = torch.randn(B, N, dtype=torch.float64)
    gs = torch.randn(B, N * S, dtype=torch.float64)
    with torch.no_grad():
        out_old = old(qs, max_qs, gs)
        out_new = new(qs, max_qs, gs)
    assert out_old.shape == (B, 1)
    torch.testing.assert_close(out_new, out_old, rtol=0, atol=1e-12)
