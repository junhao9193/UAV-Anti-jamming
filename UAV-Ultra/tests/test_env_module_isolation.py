"""每个 envs 子模块独立可调用：用 synthetic 输入喂入、断言基本契约。

不追求穷举数值正确性（那由 ``test_env_contract`` 保证）；这里只验证模块边界清晰、
可在不构造完整 ``Environ`` 的情况下被单元化使用。
"""

from __future__ import annotations

import numpy as np

from src.envs import channel as channel_module
from src.envs import jammer_model
from src.envs import mobility


def test_pathloss_matrix_3d_distance_formula():
    """两点间路损 = 103.8 + 20.9 * log10((distance_m + 0.001) * 1e-3)。"""
    tx = [[0.0, 0.0, 0.0]]
    rx = [[1000.0, 0.0, 0.0]]
    loss = channel_module.pathloss_matrix(tx, rx)
    # 距离 ≈ 1000m → 1.000001 km → log10(1.000001) ≈ 4.3e-7
    assert loss.shape == (1, 1)
    expected = 103.8 + 20.9 * np.log10(1000.001 * 1e-3)
    assert abs(loss[0, 0] - expected) < 1e-9


def test_uavchannels_update_positions_and_pathloss():
    ch = channel_module.UAVchannels(n_uav=3, n_channel=4, BS_position=[0, 0, 0])
    positions = [[0, 0, 0], [10, 0, 0], [0, 10, 0]]
    ch.update_positions(positions)
    ch.update_pathloss()
    assert ch.PathLoss.shape == (3, 3)
    # 对角线（self-pair）必为最小路损（距离接近 0.001m）
    diag = np.diagonal(ch.PathLoss)
    off_diag = ch.PathLoss - np.diag(diag)
    assert (diag <= off_diag[off_diag > 0].min()).all()


def test_sample_complex_gaussian_shape_and_dtype():
    rng = np.random.default_rng(42)
    h = channel_module.sample_complex_gaussian((2, 3, 4), rng)
    assert h.shape == (2, 3, 4)
    assert h.dtype == np.complex64


def test_generate_p_trans_requires_explicit_rng():
    """plan locked decision #4：``rng=None`` 必须报错。"""
    import pytest
    with pytest.raises(TypeError, match="rng=None is forbidden"):
        jammer_model.generate_p_trans(jammer_state_dim=12, rng=None)


def test_generate_p_trans_rows_sum_to_one():
    rng = np.random.default_rng(0)
    p = jammer_model.generate_p_trans(jammer_state_dim=12, rng=rng)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_build_mobility_strategy_dispatch():
    class FakeCfg:
        uav_mobility_control = "gauss_markov"
        jammer_mobility_model = "uav_guided_markov"

    cfg = FakeCfg()
    uav_strategy = mobility.build_uav_mobility_strategy(cfg)
    jammer_strategy = mobility.build_jammer_mobility_strategy(cfg)
    assert isinstance(uav_strategy, mobility.GaussMarkovUAVStrategy)
    assert isinstance(jammer_strategy, mobility.UAVGuidedMarkovJammerStrategy)


def test_build_mobility_strategy_rejects_unknown_mode():
    import pytest

    class BadUAVCfg:
        uav_mobility_control = "??"
        jammer_mobility_model = "gauss_markov"

    with pytest.raises(ValueError, match="uav_mobility_control"):
        mobility.build_uav_mobility_strategy(BadUAVCfg())

    class BadJammerCfg:
        uav_mobility_control = "gauss_markov"
        jammer_mobility_model = "??"

    with pytest.raises(ValueError, match="jammer_mobility_model"):
        mobility.build_jammer_mobility_strategy(BadJammerCfg())
