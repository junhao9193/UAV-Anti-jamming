from __future__ import division

from pathlib import Path
from typing import Any, Dict, Optional, Union


DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "length": 500,
    "width": 250,
    "low_height": 60,
    "high_height": 120,
    "k": 0.8,
    "sigma": 0.2,
    "uav_power_min": 23.0,
    "uav_power_max": 36.0,
    "jammer_power": 30,
    "sig2_dB": -114,
    "uavAntGain": 3,
    "uavNoiseFigure": 9,
    "jammerAntGain": 3,
    "bandwidth": 1.8 * 1e6,
    # Scale factor for UAV-UAV interference term in SINR denominator (difficulty knob).
    "uav_interference_scale": 1.5,
    "data_size": 0.8 * 1024**2,
    "t_Rx": 0.98,
    "t_collect": 0.5,
    "timestep": 0.2,
    "jammer_start": 0.2,
    "t_dwell": 2.28,
    "n_ch": 4,
    "n_cm_for_a_ch": 2,
    "n_jammer": 3,
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
    "max_distance1": 99,
    "max_distance2": 1,
    # Spectrum sensing range clipping (meters). Use a very large value to approximate global info.
    "sensing_jammer_range": 250.0,
    "sensing_uav_range": 200.0,
    "is_jammer_moving": True,
    "type_of_interference": "markov",
    "step_forward": 1,
    "p_trans_mode": 1,
    # Fix Markov transition matrix generation across runs (for fair IQL vs QMIX comparison).
    "p_trans_seed": 0,
    "reward_energy_weight": 1.0,
    "reward_jump_weight": 0.1,
    # Fairness (team penalty): if any cluster's success rate < threshold, penalize the whole team.
    "fairness_min_success_rate": 0.5,
    "fairness_weight": 1.0,
    "csi_pathloss_offset": 80.0,
    "csi_pathloss_scale": 60.0,
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
            try:
                import yaml
            except ModuleNotFoundError:
                yaml = None

            if yaml is None:
                raise ModuleNotFoundError("PyYAML is required to load YAML config files")

            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("env config must be a mapping/dict")
            merged.update(loaded)

    if config:
        merged.update(config)

    return merged
