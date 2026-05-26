# UAV-Ultra

`UAV-Ultra` 是从 `UAV-Jammer-RL/` 拆出的模块化实现。已完成阶段：

- **Stage 0**：从 baseline `2360ab9` 生成 golden-master 环境轨迹。
- **Stage 1**：包骨架、`pyproject.toml` 与纯领域实体 `entities/`。
- **Stage 2**：类型化配置层 `src/config/`（schema / loader / specs + 分层 YAML defaults）。
- **Stage 3**：拆分 baseline 1231 行 `core.py` 为 8 个 `src/envs/` 子模块（mobility / channel / link_budget / jammer_model / sensing / reward / observation / action_space）+ `environment.py` 编排器；通过 golden-master 等价测试（seed 0/1/2 × 32 步逐数值对齐，`rtol=1e-9`）。
- **Stage 3.5**：opt-in 移动控制扩展。新增 9 个配置字段、4 个移动策略（GaussMarkov / Policy UAV + GaussMarkov / UAVGuidedMarkov Jammer）、policy 模式下 3 维 mobility delta 动作、观测层可选 6 维 CH mobility 特征、mobility 惩罚 reward 项。默认配置下 100% 等价 baseline。
- **Stage 4**：算法层 `src/algorithms/` 迁移完成。8 个子包（`common` + `iql / qmix / vdn / qplex / mappo / world_model / heuristic`）共 46 个 source 文件 + 22 个 test 文件；6 个算法注册到统一工厂；4 个 DQN 族 trainer 一步更新与 baseline 逐位等价（state_dict 拷贝回归，atol=1e-7）；world_model 作为组件提供（无独立训练器，留给 Stage 5 callback）；MAPPO 超参全部从 config 取，无硬编码。
- **Stage 5**：训练侧 `src.training` 落地。统一 Runner 覆盖 `iql / vdn / qmix / qplex / mappo` smoke；DQN 族使用 `SubprocVecEnv` 与 `step_async → learn(old replay) → step_wait → store` 顺序；MAPPO 保持 baseline 单 `Environ` on-policy 路径；QMIX callback 框架支持 `policy_mobility / value_expansion / wm_alternating / jammer_prediction / critic_stable`；checkpoint 保存 callback state；训练产物写入 `runs/experiment-data/{algorithm}_expN`。

## Stage 0 生成 golden-master

项目使用 conda 环境，当前验证命令默认在 `marl` 环境中运行。

```bash
git worktree add --detach /tmp/uav-ultra-baseline-2360 2360ab92ec438528f6e194feda2405f9e943179d
conda run -n marl python -B scripts/generate_golden_master.py \
  --baseline-root /tmp/uav-ultra-baseline-2360
```

输出文件位于 `tests/fixtures/golden_master/`：

- `env_trace_2360ab92.json`：提交、配置哈希、seed、数组 schema 等元信息。
- `env_trace_2360ab92.npz`：固定 action trace 下的 state/reward/delivery 等数值标准答案。

## Stage 1 验证

在 `UAV-Ultra/` 目录下运行：

```bash
conda run -n marl python -c "from src.entities import UAV, Jammer, RP; print(UAV, Jammer, RP)"
```

如果当前 conda 环境安装了 `pytest`，也可以运行：

```bash
conda run -n marl python -m pytest
```

这一阶段不迁移环境、算法和训练循环，只保证新包可安装、可导入，且实体属性与
基准提交 `2360ab92ec438528f6e194feda2405f9e943179d` 中的旧实现保持一致。

## Stage 2 配置层用法

类型化配置 dataclass + 分层 YAML defaults，加载时由 loader 做显式校验
（unknown key / missing key / 类型 / 跨字段约束）。**项目内只通过 loader 构造配置**，
dataclass 不携带业务默认值。

```python
from src.config import (
    load_env_config, load_train_config, load_algo_config,
    specs, env_run_summary,
)

# 1) 加载环境配置：packaged env.yaml ← yaml_path ← overrides
cfg = load_env_config()                                  # 全用 packaged 默认
cfg = load_env_config(yaml_path="my_exp.yaml")           # 叠加部分覆盖
cfg = load_env_config(overrides={"env_seed": 42})        # Python 字典覆盖

# 2) 维度规格（algorithms 层独立可用，不需要 import envs）
specs.state_dim(cfg)            # = (n_des + 1) * n_channel
specs.action_dim(cfg)           # = n_channel ** n_des
specs.total_param_dim(cfg)      # = action_dim * param_dim_per_action

# 3) 加载算法配置
#    DQN 族：train/default.yaml ← algo/<name>.yaml ← yaml_path ← overrides
#    MAPPO ：algo/mappo.yaml ← yaml_path ← overrides （不合并 train/default）
qmix = load_algo_config("qmix")                          # lr_mixer null 落定为 lr_q
qmix = load_algo_config("qmix", overrides={"num_envs": 8})
mappo = load_algo_config("mappo")

# 4) 运行时元信息快照（用于 runs/experiment-data 目录命名 & 复现）
summary = env_run_summary(cfg, yaml_path=".../env.yaml", overrides={"env_seed": 42})
```

注册表 `ALGO_CONFIG_TYPES = {"iql", "qmix", "vdn", "qplex", "mappo"}` 用于按名字查算法
dataclass 类型，Stage 4/5 的 Runner 通过它分发。

### Stage 2 验证

```bash
# 全量 pytest
conda run -n marl python -m pytest UAV-Ultra/tests/ -q

# 单独 Stage 2 四个测试文件
conda run -n marl python -m pytest UAV-Ultra/tests/test_config_schema.py -v
conda run -n marl python -m pytest UAV-Ultra/tests/test_config_loader.py -v
conda run -n marl python -m pytest UAV-Ultra/tests/test_config_specs.py -v
conda run -n marl python -m pytest UAV-Ultra/tests/test_config_import_purity.py -v

# env.yaml 是否包含 baseline 全部 63 键（Stage 3.5 后 packaged env.yaml 已扩展，sha 不再
# 等于 baseline；包含关系由 test_config_loader 中的子集校验保证）
conda run -n marl python -m pytest UAV-Ultra/tests/test_config_loader.py::test_load_env_config_baseline_keys_subset_equal -v

# 手动 smoke（不依赖 pytest 的 fallback）
conda run -n marl python -c "
from src.config import load_env_config, load_algo_config, specs
cfg = load_env_config()
print('state_dim=', specs.state_dim(cfg), 'action_dim=', specs.action_dim(cfg))
qmix = load_algo_config('qmix')
print('qmix.lr_mixer=', qmix.lr_mixer, '(== lr_q ?', qmix.lr_mixer == qmix.lr_q, ')')
mappo = load_algo_config('mappo')
print('mappo.gae_lambda=', mappo.gae_lambda, 'mappo.n_steps=', mappo.n_steps)
"
```

## Stage 3 + 3.5 环境层用法

### 默认配置（与 baseline 等价）

```python
import numpy as np
from src.config import load_env_config, specs
from src.envs import Environ

cfg = load_env_config()
env = Environ(config={"env_seed": 0})
state = env.reset()
print(np.asarray(state).shape)        # (n_ch=4, state_dim=18)
print(specs.state_dim(cfg))           # 18

# 一条合法动作：(discrete_action, params)，每个 CH 一份
action = [(0, np.zeros(env.total_param_dim, dtype=np.float32)) for _ in range(env.n_ch)]
next_state, reward, done, info = env.step(action)
print(env.last_link_metrics["delivery"].shape)        # (n_ch, n_des)
```

### Stage 3.5 opt-in（policy mobility + uav_guided_markov + mobility 观测）

```python
env = Environ(config={
    "env_seed": 0,
    # UAV 簇头按 policy 行动；CM 继续跟随
    "uav_mobility_control": "policy",
    "uav_velocity_delta_max": 1.0,
    "uav_direction_delta_max": 0.1,
    "uav_p_delta_max": 0.05,
    # 干扰机方向均值朝最近 CH 偏置（g ∈ [0, 1]，0 严格退化为 gauss_markov）
    "jammer_mobility_model": "uav_guided_markov",
    "jammer_guidance_strength": 0.3,
    # 观测追加 CH 自身归一化 (pos_xyz, vel, dir, p) 6 维
    "observation_include_mobility": True,
    # 可选移动 reward 形状（默认 0 等价 baseline）
    "mobility_oob_penalty_weight": 0.0,
    "mobility_energy_weight": 0.0,
})
state = env.reset()
print(np.asarray(state).shape)        # (n_ch, 24) —— 18 + 6 mobility feats

# policy 模式下，每个 CH 的 params 末尾追加 3 维 [vel_delta, dir_delta, p_delta] ∈ [-1, 1]
per_ch = env.total_param_dim + 3
action = [(0, np.zeros(per_ch, dtype=np.float32)) for _ in range(env.n_ch)]
env.step(action)
```

### 关键契约

- **golden master 等价**：默认配置下，`Environ.step` 输出与 `tests/fixtures/golden_master/env_trace_2360ab92.npz` 在 `rtol=1e-9` 内逐位一致（seed 0/1/2 × 32 步）。
- **公开调试字段** `env.last_link_metrics`：包含 `delivery / success_flags / transmit_times` 三个 `(n_ch, n_des)` numpy 数组，替代 baseline Stage 0 generator 的 monkey-patch 模式。
- **铁律一**：`src.envs.*` 不 import `torch`，由 `test_env_no_torch_import.py` 静态 + 运行期双确认。
- **维度同步**：`np.asarray(env.get_state()).shape == (cfg.n_ch, specs.state_dim(cfg))` 是强制断言；观测特征增删必须同步更新 `specs.state_dim`。

### Stage 3 + 3.5 验证

```bash
# Stage 3 核心契约（必须先过）
conda run -n marl python -m pytest UAV-Ultra/tests/test_env_contract.py -v

# envs/ 不含 torch
conda run -n marl python -m pytest UAV-Ultra/tests/test_env_no_torch_import.py -v

# Stage 3.5 扩展
conda run -n marl python -m pytest \
    UAV-Ultra/tests/test_uav_mobility_strategies.py \
    UAV-Ultra/tests/test_jammer_mobility_dispatch.py \
    UAV-Ultra/tests/test_action_space_extension.py \
    UAV-Ultra/tests/test_observation_extension.py \
    UAV-Ultra/tests/test_reward_extension.py -v

# 全量
conda run -n marl python -m pytest UAV-Ultra/tests/ -q
```

## Stage 4 算法层用法

```python
from src.algorithms import (
    ALGO_NAMES, build_trainer, build_evaluator,
)
from src.config import load_env_config, load_algo_config

env_cfg = load_env_config()

# 5 个学习算法：iql / vdn / qmix / qplex / mappo
algo_cfg = load_algo_config("qmix")
trainer = build_trainer("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, device="cpu")
evaluator = build_evaluator("qmix", env_cfg=env_cfg, algo_cfg=algo_cfg, trainer=trainer)

# heuristic：无 trainer（约定 trainer=None），只需 evaluator
heuristic_eval = build_evaluator("heuristic", env_cfg=env_cfg)
```

### 算法层关键契约

- **统一工厂**：`build_trainer(name, *, env_cfg, algo_cfg=None, device=None, **kwargs)`、`build_evaluator(name, *, env_cfg, algo_cfg=None, trainer=None, **kwargs)`。
- **heuristic 特例**：`build_trainer("heuristic", ...)` 始终返回 `None`；其余 5 个学习算法 `algo_cfg` 必传。
- **`ValueDecompTrainerBase`**：VDN / QMIX / QPLEX 共享 critic + actor + target_sync 更新框架；子类只实现 `_build_mixer`，QPLEX 额外覆盖 `_mix / _target_mix / _collect_*_extras` 注入 `max_agent_qs`。
- **`MPDQNAgent` 不持 replay**：所有 replay 在 trainer 侧；IQL 用 `JointReplayBuffer(per_agent_reward=True)`，VDN/QMIX/QPLEX 用 `per_agent_reward=False`。
- **`agents: list[MPDQNAgent]` 跨 4 个 DQN 族 trainer 统一**：保住 `QMIXValueTeacher` 等 adapter 的字段访问路径。
- **world_model 组件**（无独立 trainer）：`JointWorldModel` / `encode_joint_action_exec` / `compute_wm_losses` / `WorldModelSequenceReplayBuffer` / `imagine_rollout` / `rollout_td_lambda_return` 供 Stage 5 callback 装配。
- **`qmix/world_model_adapter.py`**：QMIX 侧 adapter，避免 `world_model → qmix` 反向 import。
- **Stage 4 网络只接 baseline action head**（plan locked decision #4）：policy mobility 末尾 3 维由 Stage 5 适配器处理，Stage 4 网络只消费 `total_param_dim`。

### Stage 4 验证

```bash
# 注册表 + 全 6 算法构造
conda run -n marl python -m pytest UAV-Ultra/tests/test_algo_registry.py -v

# common 共享层
conda run -n marl python -m pytest UAV-Ultra/tests/test_common_*.py -v

# 4 个 DQN 族 baseline one-step 等价回归
conda run -n marl python -m pytest \
    UAV-Ultra/tests/test_algorithm_train_step_baseline_regression.py -v

# 各算法
conda run -n marl python -m pytest UAV-Ultra/tests/test_iql_*.py \
    UAV-Ultra/tests/test_vdn_trainer.py UAV-Ultra/tests/test_qmix_trainer.py \
    UAV-Ultra/tests/test_qplex_trainer.py UAV-Ultra/tests/test_mappo_trainer.py \
    UAV-Ultra/tests/test_world_model_*.py UAV-Ultra/tests/test_qmix_world_model_adapter.py \
    UAV-Ultra/tests/test_heuristic_policies.py -v

# 依赖纯度（algorithms 不 import envs/training/evaluation）
conda run -n marl python -m pytest UAV-Ultra/tests/test_algorithms_dependency_purity.py -v

# 全量
conda run -n marl python -m pytest UAV-Ultra/tests/ -q
```

## Stage 5 训练 Runner 用法

统一入口：

```bash
cd UAV-Ultra
conda run -n marl python -m src.training.runner qmix \
  --episodes 1 --steps 20 --num-envs 1 --batch-size 2 \
  --device cpu --start-method fork --no-save

# 内置 preset：先加载完整实验默认值，再由 CLI flags 覆盖
conda run -n marl python -m src.training.runner qmix \
  --preset qmix_plain_baseline \
  --episodes 1 --steps 2 --num-envs 1 --batch-size 2 \
  --device cpu --start-method fork --no-amp --no-save
```

`--amp / --no-amp` 会覆盖 preset 中的 `use_amp`；`--resume PATH` 只恢复模型权重和 callback state，并把 checkpoint path/sha256 写入新 run 的 `training_data.json`。

Python API：

```python
from src.training import run_training

result = run_training(
    "qmix",
    preset="qmix_plain_baseline",
    algo_overrides={
        "n_episode": 1,
        "n_steps": 20,
        "num_envs": 1,
        "batch_size": 2,
        "device": "cpu",
        "start_method": "fork",
    },
    no_save=True,
)
print(result.metrics["reward"])
```

QMIX callback 只能写在 `QMIXConfig.callbacks`，其它算法写 `callbacks` 会被 schema unknown-key 拒绝。用户列表顺序会被忽略，Runner 固定按：

```text
policy_mobility -> value_expansion -> wm_concurrent -> wm_block_alternating -> jammer_prediction -> critic_stable
```

`value_expansion` 必须与 `wm_concurrent` 或 `wm_block_alternating` 之一成对启用，避免 value-expansion 使用未训练 world model；`jammer_prediction` 与 `critic_stable` 可单独用于 plain QMIX。若环境启用 `uav_mobility_control="policy"`，必须显式加 `policy_mobility` callback；Stage 5 该 callback 只追加 3 维 zero-delta mobility 参数，不学习 mobility head。

### 内置 Presets

内置 preset 位于 `src/config/defaults/presets/`，顶层包含
`algorithm / description / source / env / algo`。`env.yaml` 继续只放环境字段；
preset 是实验覆盖层，不是新的环境默认。DQN 族与 QMIX+WM 系列 preset 与 baseline
`train_*.py` 字段级对齐：`n_episode=3000`、`num_envs=16`、`batch_size=512`、
`updates_per_learn=1`、`lr_actor=1e-3`、`lr_q=1e-3`、`lr_mixer=1e-3`、`max_grad_norm=10.0`。
（`lr_*` 与 `max_grad_norm` 取 baseline 值；`num_envs/batch_size/n_episode/start_method` 保留 5090 实测吞吐设置。）

可用内置名：

```text
iql_baseline
vdn_baseline
qplex_baseline
mappo_baseline
qmix_plain_baseline
qmix_wm_concurrent_baseline
qmix_wm_concurrent_jp_baseline
qmix_wm_block_baseline
qmix_wm_block_jp_baseline
qmix_wm_block_jp_cs_baseline
```

`--preset NAME_OR_PATH` 的解析规则：

- 不含 `/`、`\`、`.` 的字符串视作内置名，解析到 packaged `defaults/presets/{name}.yaml`；内置名允许字母、数字、下划线和连字符。
- 含路径分隔符或 `.yaml/.yml` 的值视作文件路径；相对路径按当前工作目录解析。
- 两种模式不 fallback，避免同名歧义。

CLI flags 和 Python `*_overrides` 总是覆盖 preset；`--callback` 会整体替换
`preset.algo.callbacks`，不是追加。例如 `--callback critic_stable` 会把 callback 集合
替换成仅 `critic_stable`，而 `--callback wm_concurrent` 会因为缺少
`value_expansion` 被 loader 拒绝。

使用 preset 保存训练产物时，`training_data.json` / `run_config.json` 的
`config.preset` 会记录 `name/path/sha256/description/source`；未使用 preset 时该字段为
`null`。

保存时默认写入 `UAV-Ultra/runs/experiment-data/`：

```text
runs/experiment-data/{algorithm}_expN/
  training_data.json
  training_data.npz
  {algorithm}_weights.pth
```

其中 `expN` 是递增实验次数，例如 `qmix_exp1`、`qmix_exp2`。`runs/` 已由 `.gitignore` 忽略。
每次训练还会在同一实验目录写入本次运行日志，例如 `qmix_wm_block_jp_3000_seed0.out`；中途 traceback 也会进入这个文件。外层 `nohup` 的 launcher 日志可临时放到 `UAV-Ultra/runs/logs/`。

Stage 6 补齐了 `evaluation/` runner、trainer 权重对称加载
（`load_trainer_state_dict`）、world-model callback 的 `wm_cfg/td_cfg` reload 校验，
并把 vecenv worker seed 切到 `SeedSequence.spawn(n_envs)`。

### Stage 6：评估 Runner

训练后可直接用保存的 Stage 5+ checkpoint 做评估：

```bash
python -m src.training.runner qmix --episodes 1 --steps 2 --num-envs 1
python -m src.evaluation.runner qmix \
  --checkpoint runs/experiment-data/qmix_*/qmix_weights.pth \
  --episodes 1 --steps 2 --num-envs 1 --no-save
```

启发式评估不需要 checkpoint：

```bash
python -m src.evaluation.runner heuristic \
  --policy greedy_sensing \
  --power-mode quality_adaptive \
  --episodes 1 --steps 2 --num-envs 1 --no-save
```

QMIX callback checkpoint 必须在 eval 时复现同一 callback 集合；例如：

```bash
python -m src.evaluation.runner qmix \
  --checkpoint runs/experiment-data/qmix_*/qmix_weights.pth \
  --callback value_expansion \
  --callback wm_alternating
```

评估保存沿用同一实验根目录，但文件名区分为：

```text
runs/experiment-data/{algorithm}_expN/
  evaluation_data.json
  evaluation_data.npz
```

旧 `UAV-Jammer-RL/Main/evaluate/evaluate_mpdqn.py`、
`evaluate_mappo.py`、`run_heuristic.py` 的新入口统一为
`python -m src.evaluation.runner ...`。Stage 6 不删除 baseline 目录，也不复现
旧脚本额外写出的 `eval_summary.json`。

### Stage 5 验证

```bash
conda run -n marl python -m pytest \
  UAV-Ultra/tests/test_training_callbacks.py \
  UAV-Ultra/tests/test_training_runner.py \
  UAV-Ultra/tests/test_training_persistence.py -v

conda run -n marl python -m pytest UAV-Ultra/tests/ -q
```
