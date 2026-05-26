"""``HybridActor`` / ``CentralValueNet`` state_dict 拷贝回归。"""

from __future__ import annotations

import torch

from src.algorithms.common.networks import CentralValueNet, HybridActor


B, OBS_DIM, N_ACTIONS, CONT_DIM, N_AGENTS = 5, 13, 7, 4, 3
GS_DIM = 17


def test_hybrid_actor_state_dict_copy_regression(baseline_import):
    baseline_model_mod = baseline_import("algorithms.mappo.model")
    torch.manual_seed(0)

    old = baseline_model_mod.HybridActor(OBS_DIM, N_ACTIONS, CONT_DIM, N_AGENTS).double().cpu().eval()
    new = HybridActor(OBS_DIM, N_ACTIONS, CONT_DIM, N_AGENTS).double().cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    obs = torch.randn(B, OBS_DIM, dtype=torch.float64)
    agent_onehot = torch.zeros(B, N_AGENTS, dtype=torch.float64)
    agent_onehot[:, 1] = 1.0
    with torch.no_grad():
        logits_o, alpha_o, beta_o = old(obs, agent_onehot)
        logits_n, alpha_n, beta_n = new(obs, agent_onehot)
    torch.testing.assert_close(logits_n, logits_o, rtol=0, atol=1e-12)
    torch.testing.assert_close(alpha_n, alpha_o, rtol=0, atol=1e-12)
    torch.testing.assert_close(beta_n, beta_o, rtol=0, atol=1e-12)


def test_central_value_net_state_dict_copy_regression(baseline_import):
    baseline_model_mod = baseline_import("algorithms.mappo.model")
    torch.manual_seed(0)

    old = baseline_model_mod.CentralValueNet(GS_DIM, N_AGENTS).double().cpu().eval()
    new = CentralValueNet(GS_DIM, N_AGENTS).double().cpu().eval()
    new.load_state_dict(old.state_dict(), strict=True)

    gs = torch.randn(B, GS_DIM, dtype=torch.float64)
    agent_onehot = torch.zeros(B, N_AGENTS, dtype=torch.float64)
    agent_onehot[:, 0] = 1.0
    with torch.no_grad():
        v_old = old(gs, agent_onehot)
        v_new = new(gs, agent_onehot)
    assert v_old.shape == (B,)
    torch.testing.assert_close(v_new, v_old, rtol=0, atol=1e-12)


def test_mappo_submodule_attribute_names_match_baseline(baseline_import):
    baseline_model_mod = baseline_import("algorithms.mappo.model")
    new_actor = HybridActor(OBS_DIM, N_ACTIONS, CONT_DIM, N_AGENTS)
    old_actor = baseline_model_mod.HybridActor(OBS_DIM, N_ACTIONS, CONT_DIM, N_AGENTS)
    assert set(new_actor.state_dict().keys()) == set(old_actor.state_dict().keys())

    new_v = CentralValueNet(GS_DIM, N_AGENTS)
    old_v = baseline_model_mod.CentralValueNet(GS_DIM, N_AGENTS)
    assert set(new_v.state_dict().keys()) == set(old_v.state_dict().keys())
