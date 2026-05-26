"""Stage 0 golden-master fixture 自检。

这个测试不启动旧环境，只检查已经生成的标准答案文件是否完整、元信息是否指向固定
基准提交、关键数组是否存在且形状一致。真正的新旧环境数值对齐会在 Stage 3 的
`test_env_contract.py` 中完成。
"""

import json
from pathlib import Path

import numpy as np


BASELINE_COMMIT = "2360ab92ec438528f6e194feda2405f9e943179d"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_master"
JSON_PATH = FIXTURE_DIR / "env_trace_2360ab92.json"
NPZ_PATH = FIXTURE_DIR / "env_trace_2360ab92.npz"


def test_golden_master_metadata_points_to_fixed_commit():
    """元信息必须固定到明确提交，不能使用会漂移的 HEAD。"""

    metadata = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    assert metadata["baseline_commit"] == BASELINE_COMMIT
    assert metadata["npz_file"] == NPZ_PATH.name
    assert metadata["env_seeds"] == [0, 1, 2]
    assert metadata["steps"] == 32
    assert len(metadata["config_sha256"]) == 64


def test_golden_master_arrays_have_expected_trace_contract():
    """每个 seed 都应包含 state/reward/action/delivery 等 Stage 3 对齐所需数组。"""

    metadata = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    arrays = np.load(NPZ_PATH)

    for trace in metadata["traces"]:
        prefix = f"seed_{trace['env_seed']}"
        steps = trace["steps_recorded"]
        n_ch = trace["n_ch"]
        n_des = trace["n_des"]
        total_param_dim = trace["total_param_dim"]

        assert arrays[f"{prefix}_states"].shape[0] == steps + 1
        assert arrays[f"{prefix}_rewards"].shape == (steps, n_ch)
        assert arrays[f"{prefix}_dones"].shape == (steps,)
        assert arrays[f"{prefix}_actions_discrete"].shape == (steps, n_ch)
        assert arrays[f"{prefix}_actions_params"].shape == (steps, n_ch, total_param_dim)
        assert arrays[f"{prefix}_deliveries"].shape == (steps, n_ch, n_des)
        assert arrays[f"{prefix}_transmit_times"].shape == (steps, n_ch, n_des)
        assert arrays[f"{prefix}_uav_channels"].shape == (steps, n_ch, n_des)
        assert arrays[f"{prefix}_uav_powers"].shape == (steps, n_ch, n_des)
        assert arrays[f"{prefix}_p_trans"].shape == tuple(trace["p_trans_shape"])


def test_golden_master_numeric_outputs_keep_float64_precision():
    """数值型输出必须以 float64 存储。

    旧环境用 float64 计算 state/reward；若 fixture 降精度成 float32，Stage 3 的
    严格容差对齐会因参考值本身被截断而误判。这里钉死存储精度。
    """

    metadata = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    arrays = np.load(NPZ_PATH)

    for trace in metadata["traces"]:
        prefix = f"seed_{trace['env_seed']}"
        for name in ("states", "rewards", "transmit_times", "uav_powers", "reward_details"):
            assert arrays[f"{prefix}_{name}"].dtype == np.float64, name
