from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "length": 500,
    "width": 250,
    "low_height": 60,
    "high_height": 120,
    # Optional root seed for environment-private RNG streams. When set, the
    # environment trajectory is reproducible without relying on global RNG state.
    "env_seed": None,
    # Gauss-Markov mobility model used by both UAVs and jammers:
    # x_t = k*x_{t-1} + (1-k)*x_mean + sqrt(1-k^2)*N(0, sigma).
    # x_mean is the fixed target motion value sampled when the entity is created.
    # Larger k means smoother, more inertial motion; larger sigma means stronger random perturbations.
    "k": 0.8,
    "sigma": 0.2,
    "uav_power_min": 23.0,
    "uav_power_max": 36.0,
    "jammer_power": 50,
    "sig2_dB": -114,
    "uavAntGain": 3,
    "uavNoiseFigure": 9,
    "jammerAntGain": 3,
    "bandwidth": 1.8 * 1e6,
    # Scale factor for UAV-UAV interference term in SINR denominator (difficulty knob).
    "uav_interference_scale": 3.0,
    # data_size is in bits (uav_rate is in bit/s).
    "data_size": 0.8 * 1024**2 * 8,
    "t_Rx": 0.9,
    "t_collect": 0.5,
    "timestep": 0.2,
    "max_episode_steps": 1000,
    "jammer_start": 0.2,
    "t_dwell": 2.28,
    "n_ch": 4,
    "n_cm_for_a_ch": 2,
    "n_jammer": 4,
    "n_channel": 6,
    "states_observed": 2,
    "p_md": 0,
    "p_fa": 0,
    "pn0": 20,
    # Per-channel static loss offset (dB), length must equal n_channel when provided.
    "channel_loss_db": [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
    # Link-level frequency selectivity (static across an entire run when seed is fixed).
    "channel_selectivity_std_db": 2.0,
    "channel_selectivity_seed": 0,
    # Fast fading (small-scale): temporally correlated Rayleigh via AR(1) on complex coefficients.
    # If `fast_fading_seed` is None, a private seed is derived from the worker/global RNG state.
    "enable_fast_fading": True,
    "fast_fading_rho": 0.95,
    "fast_fading_seed": None,
    "fast_fading_eps": 1.0e-12,
    "fast_fading_db_clip_low": -30.0,
    "fast_fading_db_clip_high": 10.0,
    # Spectrum sensing energy map weights and noise.
    "sensing_w_jammer": 1.0,
    "sensing_w_uav": 0.8,
    "sensing_noise_std": 0.1,
    "max_distance1": 99,
    "max_distance2": 1,
    # Spectrum sensing range clipping (meters). Use a very large value to approximate global info.
    "sensing_jammer_range": 250.0,
    "sensing_uav_range": 200.0,
    "is_jammer_moving": True,
    # Fix Markov transition matrix generation across runs (for fair IQL vs QMIX comparison).
    "p_trans_seed": 0,
    # Markov base preference: each row gets this many preferred next joint states.
    "p_trans_preferred_next_states": 2,
    # Added weight per preferred next joint state, as a multiple of that row's pre-normalization sum.
    "p_trans_preference_strength": 0.5,
    # Reactive bias: >0 makes jammer prefer next-states overlapping with observed UAV channels.
    # 0.0 keeps only the Markov base preferences.
    "jammer_reactive_beta": 1.0,
    # Number of recent partially observed UAV-channel sets used by the reactive jammer.
    # 1 preserves instantaneous reactive behavior; 4 is the default memoryful jammer.
    "jammer_memory_window": 4,
    # Per-link probability that a reactive jammer observes each UAV channel choice.
    # Values below 1.0 force partial observation, never a full oracle view.
    "jammer_reactive_observe_prob": 0.5,
    # Optional RNG seed for jammer state sampling and partial observation masks.
    "jammer_seed": None,
    "reward_energy_weight": 1.0,
    "reward_jump_weight": 0.1,
    # Fairness (team penalty): if any cluster's success rate < threshold, penalize the whole team.
    "fairness_min_success_rate": 0.5,
    "fairness_weight": 1.0,
    "csi_pathloss_offset": 80.0,
    "csi_pathloss_scale": 60.0,
    # Gaussian observation noise added after CSI normalization.
    "csi_noise_std": 0.05,
    "csi_clip": True,
}


def load_env_config(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(DEFAULT_ENV_CONFIG)

    if config_path is None:
        default_path = Path(__file__).resolve().parents[1] / "configs" / "env.yaml"
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("env config must be a mapping/dict")
            merged.update(loaded)

    if config:
        merged.update(config)

    return merged
