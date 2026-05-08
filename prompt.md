一、对抗规则 / 建模可改进
1. Reactive jammer 是"oracle"——直接读全局 UAV 信道
jammer_policy.py:119、174：


used_set = set(int(x) for x in np.asarray(env.uav_channels, dtype=np.int32).reshape(-1).tolist())
当 jammer_reactive_beta > 0（默认 1.0），jammer 在选下一信道时会用 当前所有 UAV 真实信道选择（无延迟、无噪声、无距离限制）来加权 Markov 转移。这相当于给 jammer 开了 oracle。

后果：

训练出来的策略实际上是对抗一个"全知 jammer"，而不是真实场景下的 jammer
想换成真实 sensing-limited jammer 就必须重训
现在的"难度"被 beta 这一个参数过度耦合
建议：增加 jammer_observation_mode 配置，至少分 3 档：

oracle（现状，跑 ablation 用）
delayed（用上一时刻或上 K 时刻的 UAV 信道）
sensed（受 jammer_observation_range 距离限制 + 概率丢包）
这是 ablation 的关键变量。Stage 1 之前做过类似的，可以参考但只引入"delayed"这一档（最便宜）。

2. compute_reward 的"前/后半段干扰"靠 m % 2 暗号传递
core.py:592, 603：


for m in idx:
    jammer_idx = self.jammer_index_list[m]
    if m % 2 == 0:   # 后半段
        jammer_interference_from_jammer0 += ...
    if m % 2 == 1:   # 前半段
        jammer_interference_from_jammer1 += ...
而 jammer_channels_list 在 jammer_policy.py:99-105 是按"先后"顺序追加的。信号的"前/后半段身份"通过 list index 的奇偶来传递——这是一个隐式约定，很容易被未来修改打破。

且有一个潜在 bug：当所有 jammer 同时切换时假设它们时间线对齐（jammer_time 是 shape=(2,) 的标量）。如果未来想让不同 jammer 异步切换，这个数据结构就 hardcoded 死了。

建议：把 jammer_channels_list / jammer_index_list / jammer_time 三个分散的 list 合并成一个结构化记录：


JammerEvent = namedtuple("JammerEvent", ["jammer_idx", "channel", "t_start", "t_end"])
self.jammer_events: list[JammerEvent] = []
然后 compute_reward 直接遍历 events 累加 interference。代码更短、不靠 index parity、未来可扩展异步切换。

3. uav_jump_count 的累计语义跨 destination 串味
core.py:993-1000：


for j in range(self.n_des):
    channel_last = self.uav_channels[i][j]
    self.uav_channels[i][j] = int(decoded % self.n_channel)
    ...
    if self.uav_channels[i][j] != channel_last:
        self.uav_jump_count[i] += 1
然后 get_reward 在 core.py:663 把 self.uav_jump_count[tra] 作为每个 (tra, rec) 链路的跳频惩罚——意味着 cluster tra 的 2 条 destination 链路都会被减去同一个累计跳频次数，相当于双倍惩罚。

例：cluster 0 有 dest a 和 dest b，本次只换了 dest a 的信道，jump_count[0]=1。get_reward 会给 dest a 减 0.1，给 dest b 也减 0.1（虽然 b 没换）。

这要么是 bug 要么是有意设计，但读代码时绝对会让人 confused。建议：用 self.last_action_channel_changed[i, j] per-link bit，而不是 cluster-level 累计。

4. 没有 episode 终止条件
core.py:1049：


return state_next, reward, False, {}
done 永远是 False。终止由训练脚本外部通过 n_steps 控制。这意味着：

没有"任务成功"或"任务失败"的语义（比如：所有 cluster 同时 packet failure 5 次 → 终止）
TD bootstrap 跨 episode 边界——如果训练脚本忘了 mark done=True，agent 会以为环境是无穷长的
不能定义"提前胜利"或"提前死亡"
train_qmix.py:141 有 is_last_step 手动 mark done，但这是补丁式的。建议：

env 内部维护 n_step 计数器和 max_cycles 配置
step 返回 truncated=True 当 n_step>=max_cycles
同时定义一个语义 termination（比如所有 cluster 连续 K 步全失败 → 失败终止），让 agent 学到"撑下去"的意义
5. CSI / 频谱感知给 agent 的信息太"干净"
core.py:480、core.py:494：

agent 直接读 UAVchannels_loss_db[tra_id, rec_id, :] —— 包含 fast fading 之后的精确 pathloss
agent 直接读 self.uav_channels[k] —— 看见所有其他 cluster 的信道选择（range 只筛选了哪些 cluster 的占用被聚合，并不影响精度）
实际系统里 CSI 估计是有噪声的、其他 agent 的信道占用是通过 sensing 反推的（不是直接读取）。当前 agent 比真实场景下 capable 太多，可能解释了 reward 收敛得快。

建议：加 csi_noise_std_db 配置（给 CSI 加 dB 噪声），加 other_uav_channel_observability 配置（决定 agent 看到的是真实选择还是 sensing 反推）。

6. p_trans 生成函数 mode 语义混乱
jammer_policy.py:196-225：4 个 mode 的 if/elif 分支风格不一致，没有 docstring。mode=3 是空操作 pass（保持 uniform random），其他 mode 通过加法+归一化制造尖峰分布。

问题：

mode 之间的 distribution shape 没有定量描述
没法选择"想要的 entropy 水平"
mode=4 把 p_trans_sum[i] 全加到一个随机位置，相当于"几乎 deterministic"
默认 mode=1 在 default 配置下究竟代表什么 reactive 强度，没人能从代码看出
建议：换成参数化的 distribution 生成器，比如 dirichlet_alpha 或者 entropy_target，让这个旋钮有可解释含义。

二、代码层面不规范
7. UAV 和 jammer 的位置更新公式系数不一致（疑似 bug）
core.py:751-758：


self.uavs[i].velocity = self.k * v_prev + (1 - self.k) * np.average(...) + (1 - self.k ** 2) ** 0.5 * np.random.normal(0, sigma)
                                                                              ^^^^^^^^^^^^^^^^^^^^
core.py:912-917：


self.jammers[i].velocity = self.k * v_prev + (1 - self.k) * np.average(...) + (1 - self.k) ** 0.5 * np.random.normal(0, sigma)
                                                                                ^^^^^^^^^^^^^^^^^
UAV 用 (1-k²)^0.5，jammer 用 (1-k)^0.5。AR(1) 模型保持稳定方差应该是 sqrt(1-k²)。k=0.8 时 UAV 噪声系数 0.6，jammer 噪声系数 0.447。几乎肯定是 copy-paste typo——两边模型应该一致。

8. Environ.__init__ 末尾调 self.reset() 来反推 state_dim
core.py:252-253：


self.reset(self.generate_p_trans(mode=self.p_trans_mode))
self.state_dim = len(self.get_state()[0])
副作用：构造完一个 Env 实例就已经跑过一次 reset，外部 caller 拿到的对象是"已经在某个状态"的。这把构造和初始化混在一起，单测时也很难只构造不 reset。

建议：根据 cfg 解析地算 state_dim（CSI: n_des × n_channel + sensing: n_channel = n_des·n_channel + n_channel），然后只在外部 caller 显式 reset 时进入 random_game。

9. gym.Env API 不兼容
core.py:1041-1049: step 返回 4-tuple，gymnasium 要求 5-tuple (obs, reward, terminated, truncated, info)
core.py:1035-1039: reset(self, p_trans) 签名是位置参数 p_trans，gymnasium 要求 reset(self, *, seed=None, options=None) 返回 (obs, info)
observation_space / action_space 都是 list，gymnasium 期望单个 Space
继承 gym.Env 但完全不满足契约——要么真接口化（改成 5-tuple + dict obs），要么去掉继承（变成纯 simulator class）。当前是误导性继承。Stage 1 之前的 refactor 做的就是后者，可以保留这个想法但不做大重写。

10. UAV/Jammer entity 的运动历史 list 无界增长
entities.py:7-9、core.py:751-762：


self.uavs[i].uav_velocity.append(self.uavs[i].velocity)  # 每步都 append
...
np.average(self.uavs[i].uav_velocity)  # 每步都全量平均
单 episode 1000 步 → list 长 1000。np.average 复杂度 O(N)，加起来 O(N²)。每次 reset 重建 UAV 对象时清零，所以不是真泄漏，但 episode 内是 O(N²) 操作。

建议：用 EMA mean = α·new + (1-α)·mean 维护，O(1) per step。

11. random 全局 RNG 还在被多处使用
core.py:218 random.sample(self.uav_list, k=self.n_ch) —— 簇头选择
core.py:303-309 UAV 初始位置/方向
core.py:418-419 信道初始化和功率
core.py:459-465 sensing-based observation 的 p_md/p_fa 抽样
core.py:532 np.random.normal 给 sensing 加噪声（全局 numpy RNG）
问题：在 SubprocVecEnv 多进程里，random.seed / np.random.seed 互相覆盖，跨 worker 不可控。同进程内多个 Env 实例共享全局 RNG，互相干扰。

建议：统一所有 Env 内部随机性走 self._rng = np.random.default_rng(seed) 一个流。代价是会改变 trajectory（破坏 .pth 兼容性的训练曲线对比，但 .pth 本身仍能 load）——可以作为"下次重训时一并改"的事项。

12. channels.py 的双层 Python 循环
channels.py:17-19 update_pathloss：


for i in range(len(self.positions)):
    for j in range(len(self.positions)):
        self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
n_uav=12 → 144 次 get_path_loss 调用每次更新。get_path_loss 本身又是 Python 算 np.sqrt(d1²+d2²+d3²)。

建议：向量化：


P = np.asarray(self.positions, dtype=np.float32)  # (n, 3)
diff = P[:, None, :] - P[None, :, :]               # (n, n, 3)
dist = np.linalg.norm(diff, axis=2) + 1e-3          # (n, n)
self.PathLoss = 103.8 + 20.9 * np.log10(dist * 1e-3)
~10× 加速，且 update_pathloss 是 hot path（每 env step 调用）。这一条是真正能缩短 wallclock 的。

13. 死代码 / 不一致命名
channels.py:5-7 h_bs / h_uav / fc 字段定义但 get_path_loss 公式只用距离，BS_position 也存了不用
core.py:89 self.channels = np.zeros([self.n_channel]) 分配但全程不读不写
core.py:211 self.training = True 设了但项目里没人读
core.py:286-296 all_observed_states 维护两个 list（permutations 和 combinations），但 all_observed_states_list (combinations) 看起来是历史遗留，没在 reward / state 里用过
core.py:10 from scipy.special import comb, perm —— 标准库 math.comb / math.perm 就够了
core.py:3 from copy import deepcopy —— 在当前文件里完全没用到
reward_details() 返回 tuple 但 get_reward() 返回 ndarray —— API shape 不一致
三、优先级建议
如果这些问题是为了"研究能拿出来发表"，我会按这个顺序处理：

优先级	项	理由
P0	#7 jammer mobility 系数 typo	是 bug，影响 jammer 运动统计性质，不修就改了又不知道
P0	#3 jump_count 跨 destination 串味	影响 reward 公平性，模糊跳频惩罚的真实强度
P1	#1 jammer reactive 加 delay/range	关系到"agent 对抗的是不是真实 jammer"，关系到 paper claim
P1	#12 update_pathloss 向量化	实测 hot path 30%+ env step 时间在这里，能省真实 wallclock
P2	#4 episode 终止 + #2 jammer event 重构	让 env 接口更标准、更可扩展；不影响数学但影响后续 contribution
P3	#5 CSI/sensing 加噪声	ablation 和真实性，但需要重训对比
P3	#9 gym.Env 接口/#11 RNG 统一	工程整洁，但破坏旧训练 trajectory（.pth 仍兼容）
