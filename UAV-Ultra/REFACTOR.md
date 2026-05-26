# UAV-Ultra 重构方案

将 `UAV-Jammer-RL/` 干净重构为 `UAV-Ultra/`：实体、环境、配置、算法、训练彻底模块化，实现**解耦、内聚、单向依赖**。

---

## 一、约束（Constraints）

### 1. 行为等价约束（最硬）
- 对齐基准固定为 `UAV-Jammer-RL/` 提交 `2360ab92ec438528f6e194feda2405f9e943179d`。工作区里尚未提交的实验性修改，先不作为重构对齐目标；后续如要迁入，按独立特性显式合并。
- 环境物理过程必须与已提交旧版**数值等价**：固定 seed + 固定 action trace 下，`step()` 的 reward、链路投递判定（delivery / transmit_time）、状态向量用严格浮点容差对齐（例如 `np.testing.assert_allclose`），不要求所有中间变量逐位一致。SINR 是中间量、旧环境不对外暴露，不单独对齐 —— 它的偏差必然显形在 delivery / transmit_time / reward 上，由这些可观测输出间接覆盖。
- 等价性只约束环境**物理过程本身**；不要求重构后由 RNG 派生的对象（如 `generate_p_trans` 生成的转移矩阵、运动噪声序列）与旧版逐位一致 —— 这类生成函数只需功能正确。golden-master 等价测试靠**回放存好的输入**（如固定 `p_trans`）来隔离物理过程，与此不矛盾。
- 重构是结构调整，**不改变实验语义**。任何数值变化必须是显式、有记录的决策，不得作为重构副作用混入。
- 训练曲线不作为逐点一致硬约束；训练层用短跑 smoke test、多 seed 统计区间、关键指标趋势和差异记录来验收。
- 算法层不读取旧 `.pth` 权重，重构后**重新训练**；但必须保留轻量回归：无权重路径（action encoding、target/loss、eval action selection）直接喂 synthetic 输入做公式级对齐；带权重模块（如 mixer）用显式构造权重或从旧模块 copy `state_dict` 后喂同一 batch 对齐。网络结构（层次、张量形状、参数量）与旧版一致。

### 2. 依赖方向约束
- 依赖只能单向流动，不允许反向或环形 import：

  ```
  utils       →  (无依赖；叶子层，可被任意上层依赖)
  entities    →  (无依赖)
  config      →  (无依赖)
  envs        →  依赖 entities + config
  algorithms  →  依赖 config（经公共 specs 拿维度，不 import 具体 Env 类）
  training    →  依赖 envs + algorithms + config（training/callbacks/ 同此约束）
  evaluation  →  依赖 envs + algorithms + config（与 training 平级）
  ```

- 铁律一：`envs/` 不得 import `torch`（环境是纯物理仿真）。
- 铁律二：`algorithms/` 不得 import `envs/` 的内部实现，只通过 `config` 与 `config/specs.py`（状态维、动作维、参数维等公共规格）通信。
- `algorithms/` 层内允许单向复用：具体算法（如 `qmix/`、`vdn/`）可以 import `common/`；`qmix+wm` 相关 trainer/callback 可以 import `world_model/`。`common/` 与 `world_model/` 不得反向 import 任何具体算法目录。
- 世界模型适配方向必须单向：具体算法侧（如 `qmix/world_model_adapter.py`）或 `training/callbacks/wm_alternating.py` import `world_model/`；`world_model/` 永不 import `qmix/`、`vdn/`、`qplex/`、`mappo/` 等具体算法。
- 公共规格边界：维度信息属于「配置派生出的接口契约」，不属于环境内部实现。`envs/` 和 `algorithms/` 都可以读取 `config/specs.py`，但算法不能为了拿 `state_dim/action_dim` 去 import `envs.environment.Environ` 或 `envs.action_space`。
- `config/specs.py` 只允许写无副作用纯函数，不得 import `envs/`；`envs/observation.py` 是状态向量布局的唯一真相源。任何 observation 特征增删，都必须同步更新 `specs.py`，并用测试钉死 `len(env.get_state()) == specs.state_dim(cfg)`。

### 3. 工程约束
- 唯一可导入包 `src/`，通过 `pyproject.toml` 安装；禁止 `sys.path` 拼接、禁止 `parents[N]` 式路径推断。
- 训练/评估产物（权重、日志、曲线）默认落到仓库根 `Draw/experiment-data/`，保持与旧 baseline 分析脚本兼容；用户通过 `output_root` 覆盖到 `runs/` 或其它目录时也不得进版本库。
- 每个环境子模块（mobility / channel / reward / …）必须可独立单元测试。
- 配置类型化：所有超参经 dataclass schema，加载即校验；不再以「函数默认参数」承载超参。

### 4. 反模式约束（禁止项）
- 禁止「复制整个脚本/trainer」来表达特性差异 —— 正交特性必须用配置开关 + 回调表达。
- 禁止上帝类、万能 `common.py`。
- 禁止重复部件（如多份 `joint_replay_buffer`、散落的 `mixer`）。

---

## 二、目标（Goals）

### 1. 解决现有痛点

| 痛点 | 现状 | 重构目标 |
|---|---|---|
| 上帝类 | `envs/core.py` 1231 行 `Environ` | 拆为 mobility/channel/link_budget/jammer_model/sensing/reward/observation/action_space，`Environ` 只做编排 |
| 万能工具文件 | `Main/common.py` 576 行 | 职责分散到 `training/{vec_env,logging,checkpoint}` 与 `utils/{paths,plotting}` |
| 脚本爆炸 | `Main/train/` 13 个脚本 | 收敛为 1 个通用 `Runner` |
| trainer 变体复制 | qmix 下 4 份 trainer | 收敛为 1 份 + 可组合回调 |
| 重复部件 | 4 份 `joint_replay_buffer`、散落 mixer | 收口到 `algorithms/common/{buffers,networks,optim}/` |
| 配置未类型化 | 仅 `env.yaml`，算法参数硬编码 | dataclass schema + 分层 YAML defaults |
| 产物混入源码 | `logs/` 在仓库内 | 默认统一到仓库根 `Draw/experiment-data/`；自定义 `output_root` 产物由 `.gitignore` 忽略 |

### 2. 关键设计目标

- **正交特性 = 配置开关 + 回调**：`jammer_prediction`、`critic_stable` 是可独立组合的特性；`value_expansion` 与 `wm_alternating` 必须成对启用（避免未训练 world model 注入 TD target）。这些特性做成 `training/callbacks/` 中的可组合组件 + `AlgoConfig` 字段，而非复制脚本。
- **回调包而非回调大文件**：`training/callbacks/` 只定义清晰 hook 与若干小组件，避免 `callbacks.py` 变成新的 `common.py`。
- **算法注册表**：`registry.py` 实现 `"qmix" → 构造器`，新增算法只需注册，训练/评估入口与 Runner 不改。
- **环境职责切片**：`step()` 按固定顺序调用子模块 `mobility → channel → jammer_model → link_budget → sensing → reward → observation`，每段输入输出明确。
- **观测边界**：`sensing.py` 只负责频谱占用/能量估计图与 `sensing_noise_std`；`observation.py` 负责完整状态向量装配与归一化，包括 CSI 特征与 `csi_noise_std`。
- **评估边界**：每个算法的训练文件 `trainer.py` 与评估期动作选择文件 `evaluator.py` 同放在 `algorithms/<algo>/`（同算法的训练/评估策略一起演化，保持内聚）；算法 evaluator 只负责 eval 模式下如何选动作，不创建 env、不跑 rollout、不写日志、不算指标。`evaluation/` 只放**算法无关**的评估循环与指标计算，并调用各算法 evaluator。
- **直接 Python 入口**：不保留额外 `cli.py` 包装层；训练和评估直接用 `python -m src.training.runner ...` 与 `python -m src.evaluation.runner ...`。

### 3. 目标目录结构

```
UAV-Ultra/
├── pyproject.toml
├── README.md
├── REFACTOR.md
├── .gitignore                  # 忽略本地 Draw/ 与 runs/
│
├── src/                  # 唯一可导入包
│   ├── entities/               # 纯领域实体：无算法、无 torch
│   │   ├── uav.py              # UAV, RP
│   │   ├── jammer.py
│   │   └── cluster.py
│   │
│   ├── envs/                   # 拆分后的环境
│   │   ├── environment.py      # Environ：只做编排（reset/step）
│   │   ├── mobility.py         # Gauss-Markov 运动模型
│   │   ├── channel.py          # 路损 / 频率选择性 / 快衰落 AR(1)
│   │   ├── link_budget.py      # SINR / 速率 / 链路投递判定
│   │   ├── jammer_model.py     # Markov 转移 + reactive bias
│   │   ├── sensing.py          # 频谱感知能量图 + 噪声
│   │   ├── reward.py           # 能量 / 跳频 / 公平性奖励
│   │   ├── observation.py      # 状态向量装配 + 归一化
│   │   └── action_space.py     # 参数化动作分解 / 缩放
│   │
│   ├── config/                 # 类型化配置 + 单一加载入口
│   │   ├── schema.py           # @dataclass: EnvConfig/AlgoConfig/TrainConfig
│   │   ├── loader.py           # YAML→dataclass，合并/校验/哈希快照
│   │   ├── specs.py            # 纯函数：从配置派生 state/action/param 等公共维度规格
│   │   └── defaults/
│   │       ├── env.yaml
│   │       ├── algo/{iql,qmix,vdn,qplex,mappo}.yaml
│   │       └── train/default.yaml
│   │
│   ├── algorithms/             # 公共算法积木 + 各算法独立实现
│   │   ├── common/             # 只放跨算法复用代码，不放实验入口
│   │   │   ├── base.py         # Agent / Trainer / EvalPolicy 抽象接口
│   │   │   ├── registry.py     # 名字 → 构造器
│   │   │   ├── buffers/        # replay.py / joint_replay.py
│   │   │   ├── networks/       # mpdqn.py / mixers.py / mappo.py
│   │   │   └── optim/          # 共享 loss、target、update helper
│   │   ├── iql/                # IQL 基线；复用 common/networks/mpdqn.py
│   │   │   ├── agent.py
│   │   │   ├── trainer.py      # 训练逻辑
│   │   │   └── evaluator.py    # 评估期动作选择
│   │   ├── qmix/               # 每个算法：训练 + 评估同目录
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   └── world_model_adapter.py  # qmix 专属 WM 适配；单向 import world_model
│   │   ├── vdn/
│   │   │   ├── trainer.py
│   │   │   └── evaluator.py
│   │   ├── qplex/
│   │   │   ├── trainer.py
│   │   │   └── evaluator.py
│   │   ├── mappo/
│   │   │   ├── trainer.py
│   │   │   └── evaluator.py
│   │   ├── world_model/        # qmix+wm 组件，无独立训练器/脚本
│   │   │   ├── model.py
│   │   │   ├── action_encoding.py
│   │   │   ├── rollout.py      # 模型想象 rollout，非环境 rollout
│   │   │   ├── losses.py
│   │   │   └── value_expansion.py
│   │   └── heuristic/          # 规则基线，无训练器
│   │       ├── policies.py
│   │       └── evaluator.py
│   │
│   ├── training/               # 通用 Runner，替代 13 个脚本
│   │   ├── runner.py           # 通用 episode 循环
│   │   ├── vec_env.py          # SubprocVecEnv
│   │   ├── callbacks/          # wm_alternating / jammer_prediction /
│   │   │   ├── base.py         #   value_expansion 等正交特性的 hook
│   │   │   ├── value_expansion.py
│   │   │   ├── wm_alternating.py
│   │   │   └── jammer_prediction.py
│   │   ├── logging.py          # 指标记录 + 数据落盘
│   │   └── checkpoint.py       # 权重保存 / 加载
│   │
│   ├── evaluation/             # 仅算法无关：评估循环 + 指标
│   │   ├── runner.py           # 通用评估 rollout，调各算法 evaluator
│   │   └── metrics.py          # 成功率 / 能耗 / 跳频
│   └── utils/                  # seeding.py / paths.py / plotting.py
│
├── configs/experiments/*.yaml  # 用户实验配置（覆盖 defaults）
├── scripts/                    # Stage 0 golden-master 生成等一次性脚本
├── tests/                      # 单元 + 集成测试
├── runs/                       # 可选 output_root（.gitignore）
└── docs/
```

默认产物路径不在 `UAV-Ultra/` 包目录内，而在共享仓库根：

```text
../Draw/experiment-data/{algorithm}_{timestamp}/
  training_data.json
  training_data.npz
  {algorithm}_weights.pth
```

---

## 三、总结（Summary）

### 设计核心
重构的本质是把**隐式耦合显式化**：
1. 把 `Environ` 上帝类和 `common.py` 万能文件，按**职责**切成内聚的小模块。
2. 用**单向依赖**（entities → config → envs → algorithms → training）取代当前的相互纠缠。
3. 用**配置开关 + 回调**取代「复制脚本」这一反模式 —— 13 个 train 脚本与多份 trainer 收敛为 1 个 Runner。
4. 用**算法公共层 + 各算法独立目录**取代重复 buffer/mixer/trainer：公共 loss、buffer、network helper 收到 `algorithms/common/`，每个算法的 `trainer.py` 与 `evaluator.py` 同目录维护；算法 evaluator 只做动作选择，`evaluation/` 只放算法无关的评估循环与指标。
5. 用**组件/基线分层**避免硬套模板：`world_model/` 保留模型、动作编码、模型想象 rollout、loss、value expansion 等算法无关组件供 qmix+wm 系列复用，不设独立训练器与训练脚本；具体算法适配放在算法侧（如 `qmix/world_model_adapter.py`）或 callback 侧，避免 `world_model/` 反向依赖具体算法；`heuristic/` 是无训练规则基线。
6. 用**类型化配置 + 注册表**取代硬编码超参与硬编码分发。

### 迁移顺序（自底向上，风险递增）

| 阶段 | 内容 | 验证方式 |
|---|---|---|
| 0 | 冻结基准：从 `2360ab92ec438528f6e194feda2405f9e943179d` 生成 golden-master traces | 记录提交哈希、配置哈希、固定 seed/action trace 下的 state/reward/delivery/transmit_time（旧环境不暴露 SINR，由 delivery/transmit_time 间接覆盖）；数值输出以 float64 存储；不得用会漂移的 `HEAD` 作为基准名 |
| 1 | 包骨架 + `pyproject.toml` + `entities/` | 几乎照搬，import 通过即可 |
| 2 | `config/`（类型化 schema + loader + specs） | 加载 `env.yaml` 与旧 `DEFAULT_ENV_CONFIG` 一致；`specs.py` 为无 `envs` import 的纯函数 |
| 3 | `envs/` 拆分 | `tests/test_env_contract.py`：同种子 + 同 action trace 下环境核心输出在严格容差内对齐旧版。**该对齐测试**用 `reset(p_trans=<golden master 存的矩阵>)` 回放存好的输入（要和某条轨迹比对就得复现其输入）；这不是要求重构后的 `generate_p_trans` 复现旧矩阵 —— 它只需功能正确、允许与旧版不同，环境正常训练时照常自行生成。强制断言 `len(env.get_state()) == specs.state_dim(cfg)` |
| 4 | `algorithms/common/` 共享层（buffers/networks/base/registry/optim）+ 各算法目录 | 单元测试 + 网络结构（层次/张量形状/参数量）一致性检查；不读取旧 `.pth`。无权重路径用 synthetic 输入做公式级回归；带权重模块用显式权重或 copy `state_dict` 后喂同一 batch 对齐 |
| 5 | `training/` Runner + callbacks | 逐算法跑通；短跑 smoke test 和多 seed 指标趋势与旧版可解释地一致 |
| 6 | `evaluation/` + checkpoint 对称加载 + 清理旧脚本入口 | `python -m src.training.runner ...` 与 `python -m src.evaluation.runner ...` 端到端冒烟测试；实现 `load_trainer_state_dict` 供评估 reload 权重；加载 world-model callback 时校验 `wm_cfg/td_cfg` 与当前配置一致；vecenv worker seed 改为 `SeedSequence.spawn(n_envs)`；不新增旧式 train/evaluate wrapper，`scripts/` 仅保留 Stage 0 golden-master 生成工具 |

### 验收标准
- 固定 seed + 固定 action trace 下，环境核心输出与已提交旧版在严格容差内一致。
- `observation.py` 的状态布局与 `config/specs.py` 的维度计算有同步测试保护。
- 算法网络结构（层次/张量形状/参数量）与旧版一致；不迁移旧权重，但关键算法路径有 synthetic batch、显式权重/`state_dict` 复制回归或公式级单元测试保护。
- 任意算法可完成短跑 smoke test；完整训练用多 seed 均值/方差、趋势和差异记录验收，不追求逐 episode 曲线完全重合。
- 新增一个算法 / 一个训练特性，无需新建脚本，只需注册 + 配置。
- 任一环境子模块可脱离整体单独测试。
- 全仓库无环形 import，`envs/` 不含 `torch` 依赖。

---

## 四、后续构想（Future Extension）

### Stage 3.5：移动控制扩展

在完成 Stage 3 环境拆分并通过旧环境等价验证之后，可以作为独立功能扩展加入：
**让模型控制无人机移动，同时让干扰机运动模型可配置**。

这不是重构等价阶段的一部分，不应混入 Stage 1-3 的行为对齐工作。默认配置必须保持旧行为；
新功能只能通过显式配置打开，避免污染旧实验语义和已有结果。

建议边界：
- `envs/mobility.py`：同时支持旧 Gauss-Markov 移动和策略控制移动。
- `envs/jammer_model.py` 或 `envs/mobility.py`：提供可配置的干扰机运动模型，例如 `fixed`、`gauss_markov`、`random_walk`、`scripted`。
- `envs/action_space.py`：在开启策略控制移动时，把速度、航向、爬升角或离散移动动作纳入动作空间。
- `envs/observation.py`：在需要时加入位置、速度、相对几何关系等移动决策相关特征。
- `envs/reward.py`：评估是否需要加入移动能耗、边界惩罚、碰撞/距离/覆盖等奖励项，避免策略学出不可解释轨迹。
- `config/schema.py`：新增显式开关，例如 `uav_mobility_control: "gauss_markov" | "policy"` 与 `jammer_mobility_model: "fixed" | "gauss_markov" | "random_walk" | "scripted"`。

推荐加入时机：
- 先完成 Stage 3，把旧环境拆成清晰模块并锁住 golden-master 行为。
- 再作为 Stage 3.5 增量扩展动作空间、观测和奖励。
- 最后在 Stage 4 算法迁移前固定新接口，减少算法侧返工。
