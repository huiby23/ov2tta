# Zero-Shot Coordination 问题分析与阶段性实验总结

最后更新：2026-04-30

本文档用于系统整理当前项目中关于 `Zero-Shot Coordination (ZSC)` 的阶段性理解、关键实验结论、概念澄清，以及后续研究方向。文档按“论文式分析”组织，但保留对实验决策直接有用的结论表达。

## 摘要

本文档的核心结论如下：

1. 在经典 `Overcooked / counter_circuit` 设置下，传统 `PPO CNN standard` 的 `ZSC` 失败并不主要来自“不会在测试时适应新伙伴”，而是很大程度上来自训练时 `self-play` 诱导的 `state coverage` 不足。
2. `state augmentation` 对 `XP` 的巨大提升，说明训练分布扩展对经典 ZSC benchmark 的影响远大于单纯测试时微调。
3. 但 `coverage` 并不是全部。即使只看双方训练中都见过的局面，策略之间仍然存在显著的不一致，这说明还存在一个残余的 `partner-specific mismatch`。
4. 因此，`TTA` 并非不适合 ZSC，而是不适合单独解决经典 `PPO` 上的全部 ZSC gap。更合理的定位是：先通过训练阶段方法减轻 `coverage` 问题，再让 `TTA` 去修复剩余的、真正与伙伴有关的失配。
5. 对我们自己的方法线而言，现有 `TTAPPO` 的若干收益主要来自训练阶段对 partner/context 的辅助塑形，而不是当前形式的在线更新。测试时更新这部分尚未形成强、稳、可解释的增益来源。

## 1. 问题定义

### 1.1 什么是 ZSC

在本项目语境中，`ZSC` 指：

- 一个 agent 在训练时没有见过测试时的具体伙伴；
- 测试时需要与新的 partner 协作；
- 评估重点是 `cross-play (XP)` 表现，而非仅仅 `self-play (SP)`。

这里的关键不是“能否独自解决任务”，而是“能否在没有共同训练历史的情况下，与新伙伴形成有效协作”。

### 1.2 为什么 ZSC 在 Overcooked 中会失败

传统直觉往往认为：`ZSC` 失败是因为 agent 学到了固定习惯，测试时无法适应新伙伴。

但基于我们在 `PPO CNN standard` 上的正式诊断，更准确的分解是：

1. 训练分布过窄导致的 `coverage failure`
2. 在双方都见过的局面上，仍然存在的 `partner/convention mismatch`

也就是说，ZSC 失败至少由两层因素组成：

- 第一层：测试时进入了训练中没有充分覆盖的状态区域；
- 第二层：即使进入了双方都熟悉的局面，不同策略对“当前谁该做什么”的理解仍然可能不同。

## 2. 概念澄清

### 2.1 self-play support

`self-play support` 指：

- 某个策略在“自己和自己配合”时，通常会到达的局面集合。

它不是“所有理论可达局面”，而是“这个训练出来的 policy 真正常去的局面区域”。

### 2.2 coverage

`coverage` 指：

- 训练过程中，策略实际覆盖到的局面范围是否足够广。

如果 `coverage` 很差，测试时伙伴稍微换一种行为，轨迹就会进入大量训练时没见过的状态，性能会快速崩掉。

### 2.3 shared-state

`shared-state` 指：

- 对 cross-play 中的一条测试轨迹而言，那些同时属于两个策略各自 `self-play support` 的局面。

更直白地说，就是：

- 这个局面对 A 来说不陌生；
- 对 B 来说也不陌生；
- 双方都“见过”。

### 2.4 shared-state mismatch

`shared-state mismatch` 指：

- 即使在双方都见过的局面里，不同策略仍然可能对“此时应该如何协作”有不同理解。

这部分问题不是“不会做题”，而是“对同一道题的协作含义理解不同”。

### 2.5 action agreement 的合理含义

`action agreement` 当前定义为：

- 在同一个 shared state 上，把相同角色位置的观察分别喂给两个 policy，比较它们最偏好的动作是否一致。

这个指标的重要价值在于：

- 它能刻画不同策略是否学到了相似的角色惯例；
- 因而对 `Other-Play` 所强调的对称性/惯例问题是有参考意义的。

但它的局限也必须明确：

- 在合作游戏中，好的协作并不要求双方做“相同动作”；
- 更合理的要求是“双方动作能否形成互补的联合行为”；
- 因此 `action agreement` 是一个重要但不充分的代理指标。

基于这一点，我们后续补充了 `complementarity` 分析，用于衡量当前动作组合是否对团队目标构成更好的互补配合。

## 3. PPO 主基线的正式诊断

### 3.1 评估对象

当前最重要的主基线是：

- `PPO CNN standard`
- `PPO CNN state_aug`

布局固定为：

- `counter_circuit`

对应正式结果目录：

- `PPO CNN standard`
  `/teams/ius_1663576043/hby/rl/ov2/runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full`
- `PPO CNN state_aug`
  `/teams/ius_1663576043/hby/rl/ov2/runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full`

### 3.2 Figure4 主结果

| 方法 | SP | XP |
|---|---:|---:|
| PPO CNN standard | 167.8600 | 61.4067 |
| PPO CNN state_aug | 201.8360 | 156.8307 |

这说明：

- 经典 `PPO CNN standard` 的 `XP` 非常差；
- 单纯引入 `state augmentation`，`XP` 就能大幅抬高。

### 3.3 正式诊断结果

诊断结果目录：

- `/teams/ius_1663576043/hby/rl/ov2/runs/ppo_zsc_diagnostics_full_fast_20260421-190133`

主要统计如下：

| 指标 | PPO CNN standard | PPO CNN state_aug |
|---|---:|---:|
| self-play unique states | 5713.60 | 11061.70 |
| cross unique states | 14655.90 | 16604.97 |
| support overlap | 0.0555 | 0.0559 |
| out-of-support rate | 0.7935 | 0.5735 |
| in-both-support rate | 0.0823 | 0.1868 |
| action agreement on shared states | 0.5939 | 0.7922 |
| policy TV on shared states | 0.3636 | 0.2273 |
| symmetric KL on shared states | 0.8431 | 0.6194 |
| value shift on shared states | 3.7788 | 2.9614 |

### 3.4 对正式诊断结果的解释

结论一：经典 `PPO CNN standard` 的 ZSC 失败，确实有很大一部分来自 `coverage` 不足。

最直接的证据是：

- `out-of-support rate = 0.7935`

这意味着：

- 在 cross-play 轨迹中，大约 79% 的状态不属于任一方熟悉的自博弈支持区域。

换句话说，测试时大多数时间都在陌生状态里运行。

结论二：`state_aug` 的巨大 XP 提升，与 coverage 改善高度一致。

- `out-of-support rate` 从 `0.7935` 降到 `0.5735`
- `in-both-support rate` 从 `0.0823` 提升到 `0.1868`

这说明：

- `state_aug` 不只是“多见了一些状态”，
- 它还显著增加了 cross-play 时双方都处于熟悉状态区域的时间比例。

结论三：问题不只是 coverage。

即使只看 shared states：

- `action agreement` 仍然不是 1；
- `policy TV` 与 `symmetric KL` 仍然显著；
- `value shift` 也仍然较大。

这说明：

- 双方对同一局面的角色理解、动作分布偏好、价值判断仍然不一致；
- 因而经典 ZSC 失败不能被简单归结为“全都是 OOD 状态”。

### 3.5 为什么 coverage 只改善 0.22 却能带来很大 XP 提升

表面上看：

- `out-of-support rate` 只下降了 `0.2200`
- 但 `XP` 却提升了 `+95.4240`

这个现象是合理的，因为：

1. `out-of-support rate` 是平均比例，不区分状态的重要性。
2. 序列决策中存在关键分叉点。
3. 只要覆盖到了足够多的高杠杆瓶颈状态，整条轨迹就可能从“崩溃支路”回到“正常协作支路”。

所以这里不是线性关系：

- 不是“少 22% 的陌生状态，只会换来 22% 的收益”
- 而是“减少了足够多关键陌生状态，整局任务的配合都被拉回来了”

## 4. 从 PPO 诊断到 TTA 的研究定位

### 4.1 TTA 是否不适合做 ZSC

不是。

更准确的说法是：

- `TTA` 不适合单独解决经典 `PPO` 上由 coverage 崩坏导致的全部 ZSC gap；
- `TTA` 更适合解决 coverage 改善之后，仍然残留的 partner-specific mismatch。

也就是说：

- `TTA` 更像“残差修正器”，不是“基础能力补课器”。

### 4.2 当前更合理的研究问题

基于现有结果，后续研究不应该再表述为：

- “用测试时自适应单独解决 ZSC”

更合理的表述应该是：

- “在训练阶段先减轻 coverage failure 的基础上，测试时自适应是否能进一步修复 residual partner mismatch？”

这个问题定位更准确，也更符合我们现在掌握的证据。

## 5. 与 OvercookedV2 的关系

`OvercookedV2` 的核心贡献之一是：

- 说明经典 `Overcooked` 中很多所谓 ZSC 失败，其实主要来自 `poor state coverage`；
- 然后通过新环境设计，把 benchmark 改造成真正更需要 test-time coordination 与 protocol formation 的任务。

我们的当前结果与这一论点是高度一致的：

1. 经典 `counter_circuit` 上，coverage 确实是主因；
2. 但我们进一步看到，在扣除 coverage 之后，shared-state mismatch 仍然存在；
3. 因此我们比“coverage matters”更往前走了一步：提出了 `coverage failure + residual mismatch` 的双层分解。

这意味着我们未来的工作空间不是去重复证明 `coverage` 很重要，而是：

- 明确 residual mismatch 的结构；
- 研究 TTA 该如何只针对这一残差部分工作。

## 6. TTAPPO 系列方法的消融结论

### 6.1 TTAPPO v1

主结果：

| 方法 | SP | XP |
|---|---:|---:|
| TTAPPO v1 no_adapt | 167.3680 | 65.9587 |
| TTAPPO v1 ce_only | 167.3040 | 65.5778 |
| TTAPPO v1 full | 167.3520 | 65.5680 |

解释：

- 相比 `PPO CNN standard`，`v1` 给出了一个小幅正 `XP` 信号；
- 但 `no_adapt` 并不弱于测试时更新版本；
- 因此当前 `v1` 的收益主要来自训练期的 partner auxiliary，而不是当前形式的 online adaptation。

### 6.2 TTAPPO v2 temporal

主结果：

| 方法 | SP | XP |
|---|---:|---:|
| TTAPPO v2 temporal no_adapt 单条 | 166.3120 | 91.8311 |
| TTAPPO v2 temporal ce_only | 2.4440 | 4.9036 |
| TTAPPO v2 temporal gated | 3.0520 | 5.3538 |

重复实验：

| 设置 | SP | XP |
|---|---:|---:|
| no_adapt repeats mean | 180.0000 | 80.4051 |

解释：

- temporal partner modeling 在训练期确实能带来显著 `XP` 提升；
- 但当前形式的测试时更新会灾难性崩塌；
- 因此这一阶段的经验教训是：直接在会影响 actor 的共享特征上做测试时更新，非常危险。

### 6.3 TTAPPO v3 memory

主结果：

| 方法 | SP Mean | XP Mean |
|---|---:|---:|
| v3 memory_off | 168.1040 | 74.5465 |
| v3 state_adapt | 168.6680 | 76.3305 |
| v3 state_adapt_gated | 168.6040 | 76.4492 |

因果消融：

| 方法 | XP Mean | 相对 `memory_off` |
|---|---:|---:|
| state_adapt | 76.3305 | +1.7840 |
| wrong_partner | 76.3633 | +1.8167 |
| random_partner | 76.3071 | +1.7606 |
| delayed_partner | 76.3776 | +1.8311 |
| state_readout_off | 72.8376 | -1.7089 |

解释：

- `state_adapt > memory_off` 说明 state-only memory update 这条路是真的；
- `state_readout_off` 降得很明显，说明 `partner_memory -> readout -> decision` 是真实有效路径；
- 但 `wrong/random/delayed ≈ true`，说明收益并不依赖“正确 partner 语义”，memory 更像泛化上下文状态，而不是严格的 partner behavior state。

### 6.4 TTAPPO v3.1 semantic memory

主结果：

| 方法 | SP Mean | XP Mean |
|---|---:|---:|
| v3.1 memory_off | 157.5173 | 60.4710 |
| v3.1 state_adapt | 157.8933 | 62.9347 |

解释：

- `state_adapt > memory_off` 仍然存在；
- 但整体 `SP/XP` 明显弱于 `v3`；
- 同时 `wrong/random/delayed` 仍然和真更新差不多；
- 因而这是一条典型的负结果：试图用更强的语义 margin 训练 memory，并没有真正建立起可用的 partner semantics，反而伤害了主策略质量。

## 7. action agreement 与互补协作的双指标视角

### 7.1 为什么保留 action agreement

我们保留 `action agreement`，因为它在当前项目里仍然具有重要价值：

- 它可以刻画“相同角色位置下，不同策略是否学到了相似惯例”
- 对 `Other-Play` 关心的对称性/协议一致性问题有解释力
- 它能帮助判断 cross-play 失败是否部分来自不同 run 之间学出了不同 convention

### 7.2 为什么还需要 complementarity

但单靠 `action agreement` 不够，因为：

- 好的协作不要求动作相同；
- 好的协作要求双方动作形成有效互补；
- 因而更合理的分析还应包括：
  - 当前 joint action 是否接近更优团队组合；
  - 在固定另一方动作时，我的动作是否还有明显单边 regret。

基于这一点，我们已经把 `complementarity` 分析接入现有 PPO 诊断脚本，而且不需要重新训练任何模型。

### 7.3 当前 complementarity 分析的定义

在 sampled shared states 上：

1. 固定当前两个策略的局部观察；
2. 分别取当前两边最偏好的动作；
3. 枚举所有 joint actions；
4. 用一步环境转移后的即时团队回报加下一状态 value 近似，估计每个 joint action 的团队质量；
5. 计算：
   - `joint_optimal_rate`
   - `joint_regret`
   - `unilateral_regret`

它的意义是：

- `action agreement` 在测“惯例是否相似”
- `complementarity` 在测“当前联合动作是否是好的互补搭配”

两者回答的是不同问题，应该同时保留。

### 7.4 正式 complementarity 结果

正式结果目录：

- `/teams/ius_1663576043/hby/rl/ov2/runs/ppo_zsc_diagnostics_full_comp_20260421-211127`

主要统计如下：

| 指标 | PPO CNN standard | PPO CNN state_aug |
|---|---:|---:|
| sampled shared states | 475.31 | 503.76 |
| joint optimal rate | 0.3026 | 0.3471 |
| unilateral regret | 0.1104 | 0.0840 |
| joint regret | 0.2123 | 0.1654 |

这组结果说明：

1. `state_aug` 的提升并不只体现在“更常走到 shared states”，也体现在“到了 shared states 之后，当前联合动作更接近好的团队动作组合”。
2. `joint_optimal_rate` 从 `0.3026` 升到 `0.3471`，说明 `action agreement` 的提升并不是一个误导性信号；在当前任务上，惯例更一致时，互补协作质量也同步改善。
3. 但这个值仍然不高。即使在 `state_aug` 下，shared states 中也只有大约 `34.7%` 的时刻，其当前联合动作接近一步近似意义下的最优团队动作。

如果把 coverage 和 complementarity 放在一起看，可以得到一个更保守、但也更有解释力的结论：

- `PPO CNN standard` 中，cross-play 全部步骤里，约只有 `0.0823 x 0.3026 ~= 2.5%` 的步骤同时满足“双方都熟悉这个状态”且“当前联合动作接近好的团队组合”。
- `PPO CNN state_aug` 中，这个比例约为 `0.1868 x 0.3471 ~= 6.5%`。

这里的乘法是对上面两个统计量的近似组合推断，不是额外直接测得的原始指标，但它能帮助我们直观看到：

- `state_aug` 已经把“有效协调区间”扩大了约 `2.6x`；
- 可它的绝对值仍然很低，因此当前 benchmark 上的 `ZSC` 还有显著提升空间。

## 8. 当前最稳的阶段性结论

基于现有全部结果，当前最稳的结论是：

1. 经典 `PPO CNN standard` 的 ZSC 失败，主要来源于训练期 `self-play` 带来的 coverage failure。
2. 但 shared-state mismatch 也客观存在，因此问题不能被简单化成“全是 OOD”。
3. `state augmentation` 在经典环境上的巨大成功，说明训练分布扩展是第一优先级。
4. `TTA` 仍然有研究价值，但应该被定位为 residual mismatch 的修正器，而不是 coverage 崩坏下的万能修复器。
5. 对我们自己的方法线而言，目前最可信的收益主要来自训练阶段对 partner/context 的塑形；测试时更新仍未形成强而稳定的语义性增益。

### 8.1 coverage 是否还有明显提升空间

有，而且空间并不小。

最直接的证据是：

- 即使是更强的 `PPO CNN state_aug`，`xp_out_of_sp_support_rate` 仍然有 `0.5735`
- `in_both_support_rate` 也只有 `0.1868`
- `shared_state_rate` 同样只有 `0.1868`

这意味着：

- 在当前 cross-play 轨迹中，仍有超过一半的时间落在双方都不熟悉的区域；
- 真正进入“双方都见过的局面”的时间占比不到五分之一。

换句话说，`state_aug` 已经显著改善了 coverage，但远没有把 coverage 问题解决掉。

此外，`topk_bottleneck_state_mass` 也从 `0.1074` 降到 `0.0596`，说明 cross-play 中最集中的 OOD 瓶颈状态确实被分散了一部分，但这些瓶颈并没有被消灭，只是压力有所缓解。

因此当前阶段不能得出“coverage 已经差不多了，剩下都靠 TTA”这种判断。更准确的结论是：

- 经典 `counter_circuit` 的 ZSC gap，仍然主要受 coverage 限制；
- 只不过在 `state_aug` 之后，我们终于有条件去观察残余 mismatch，而不再被 coverage 崩坏完全淹没。

### 8.2 现有 self-play PPO 的方法性缺陷

如果从机制上看，当前传统 `self-play PPO` 会同时制造 `coverage failure` 和 `shared-state mismatch`，原因并不矛盾，而是同一套训练动力学的两个侧面。

第一，`coverage failure` 的来源是：

1. 训练伙伴分布过于单一。
2. 每个 policy 基本只和自己的拷贝协作。
3. 一旦某条协作分工路径在早期回报略高，on-policy 数据收集就会持续强化这条路径。
4. 没被访问到的替代协作路径得不到数据，也就不会被学会。

这会导致：

- 训练 occupancy 迅速塌缩到少数 convention branch；
- 测试时只要新伙伴把轨迹推离这条 branch，策略就进入自己没有支持的状态区。

第二，`shared-state mismatch` 的来源是：

1. 不同 run 会在对称结构下做不同的惯例破缺。
2. actor 只需要对“自己的训练伙伴”最优，不需要对“别的 run 的伙伴”兼容。
3. critic/value 也只在自己的 convention 下被训练，因此会对 alternative-but-valid continuation 给出失真的价值判断。

这会导致：

- 即使两个策略都见过某个状态，它们对“现在谁该去拿盘子，谁该去送菜”的角色语义仍然可能不一致；
- 因而 shared states 上依然存在 `policy TV / KL / regret`。

所以，当前问题不是单一缺陷，而是两级缺陷：

- 一级缺陷：训练轨迹太窄，导致 coverage 不足；
- 二级缺陷：在有限重叠区域内，各 run 的 convention 和价值判断仍然没有对齐。

### 8.3 ZSC 中的 TTA 应该如何改进

如果接受上面的双层分解，那么 `TTA` 的设计目标也应该重写。

当前更合理的原则是：

1. 不要让 `TTA` 负责从零补 coverage。
2. 让训练阶段方法优先扩大可支持状态区域。
3. 让 `TTA` 专门修复 residual mismatch。

具体来说，未来更值得做的不是“粗暴在线更新整个 backbone”，而是：

- 只更新 partner-specific residual branch、memory readout、feature modulation 这类局部模块；
- 把测试时更新限制为对“当前伙伴兼容性”敏感、但不会破坏基础技能的参数子空间；
- 用 compatibility/complementarity 导向的目标，而不是单纯希望降低某个辅助损失。

从方法设计上，后续更应关注三个条件：

第一，更新位置要受控。

- 直接改共享 backbone，往往会把本来稳定的基本协作能力一起扰乱；
- 更合理的做法是让 backbone 保持稳定，只让 partner-conditioned residual path 发生小幅调整。

第二，更新目标要与“协作修正”真正相关。

- 单纯的 partner action prediction loss 并不等价于更好的 coordination；
- 更有价值的是能反映“当前动作是否让团队更接近兼容联合行为”的 surrogate。

第三，更新触发要有条件。

- 并不是所有状态都需要适应；
- 更合理的做法是在 shared-state 或高置信熟悉区域内做小步修正，而不是在明显 OOD 的状态里盲目在线学习。

因此，TTA 在 ZSC 里的正确角色更像：

- 一个建立在更强 train-time coverage 之上的局部兼容性校正器；
- 而不是一个试图单独解决所有 ZSC 失败来源的万能补丁。

## 9. 当前建议的研究主线

如果后续继续以 `baseline + ZSC + TTA` 为主线，当前更合理的研究顺序是：

1. 先以传统 `PPO CNN` 为主基线，明确拆分 coverage 与 residual mismatch。
2. 继续补 train-time 分析，弄清楚哪些训练机制在决定 coverage，哪些训练机制在决定 convention mismatch。
3. 在训练期引入更强的分布扩展或 partner diversity，优先扩大 self-play support 与 shared-state ratio。
4. 在这个基础上，设计只针对 residual mismatch 的 TTA，而不是粗暴更新主干参数。
5. 同时用双指标体系评估：
   - `action agreement` 看惯例/对称性问题
   - `complementarity` 看真正的互补协作质量
6. 对任何新方法，都要同时回答三个问题：
   - 为什么有收益
   - 为什么收益只有这么大
   - 它究竟修的是 coverage，还是 mismatch，还是两者都有

## 10. 附：当前关键结果目录

- PPO 正式诊断：
  - `/teams/ius_1663576043/hby/rl/ov2/runs/ppo_zsc_diagnostics_full_fast_20260421-190133`
- PPO 互补协作正式诊断：
  - `/teams/ius_1663576043/hby/rl/ov2/runs/ppo_zsc_diagnostics_full_comp_20260421-211127`
- 当前主结果索引：
  - `RESULTS_STAGE.md`
- 现有中文实验结果总表：
  - `.tmp_v31_semantic/EXPERIMENT_RESULTS_ZH.md`

## 11. 2026-04-30 更新：hidden-state-aware MAPPO/PPO 诊断

### 11.1 为什么需要重新诊断

此前的 MAPPO vs PPO state_aug 诊断使用的是 reset hidden-state readout。这个读数对 CNN policy 可以接受，但对 RNN policy 不够公平，因为 RNN 的动作分布依赖历史 hidden state。`MAPPO state_aug` 与 `PPO RNN state_aug` 都是 RNN/state-aug，因此更合理的做法是保留真实 rollout 中的 hidden state，再比较 shared state 上的 convention alignment。

### 11.2 新诊断方法

本轮新增 hidden-state-aware 诊断：

- 对每条 run 先跑 self-play，记录真实 hidden state 下的 `(state, role) -> action distribution`。
- 对每个 cross-play pair，统计 cross 轨迹中哪些状态落在双方 self-play support 中。
- 对 shared states，比较两条 run 在真实 self-play hidden context 下的动作分布差异。

这比 reset readout 更接近 RNN policy 在真实轨迹中的协作惯例。

### 11.3 结果

结果目录：`runs/hidden_state_mismatch_full_20260430-full`

| 方法 | SP | XP | Out-of-support | In-both-support | Hidden agreement | Hidden TV | Hidden KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO RNN state_aug | 162.7560 | 65.6080 | 0.7026 | 0.0843 | 0.5395 | 0.4449 | 1.8202 |
| MAPPO RNN state_aug | 223.1520 | 174.6680 | 0.5723 | 0.1628 | 0.6630 | 0.2994 | 0.9434 |

Pair-level 相关性：

| 方法 | corr(XP, out-of-support) | corr(XP, hidden TV) | corr(XP, hidden agreement) |
|---|---:|---:|---:|
| PPO RNN state_aug | -0.6439 | -0.6936 | 0.6894 |
| MAPPO RNN state_aug | -0.7972 | -0.6437 | 0.5567 |

### 11.4 解释

这轮结果改变了此前 reset-hidden 诊断下比较保守的判断。用真实 hidden-state convention profile 后，`MAPPO RNN state_aug` 的 shared-state mismatch 指标确实明显优于 `PPO RNN state_aug`：

- `hidden action agreement` 更高：`0.6630` vs `0.5395`
- `hidden policy TV` 更低：`0.2994` vs `0.4449`
- `hidden KL` 更低：`0.9434` vs `1.8202`

同时，MAPPO 的 coverage 也更好：

- `out-of-support` 更低：`0.5723` vs `0.7026`
- `in-both-support` 更高：`0.1628` vs `0.0843`

因此当前最稳妥的结论是：MAPPO 的优势可能来自两部分叠加，一是更好的 coverage，二是更一致的 hidden-state convention。centralized critic 是否是造成这两点的直接原因，还需要同架构消融来验证。

### 11.5 下一步实验定位

后续不应再只比较历史 PPO 与 MAPPO checkpoint，因为它们同时改变了太多因素。下一步应该构造同架构消融：

1. `PPO RNN state_aug`：decentralized actor + decentralized critic。
2. `MAPPO-like RNN state_aug without centralized critic`：尽量复用 MAPPO 训练框架，但 critic 只看本地信息。
3. `MAPPO RNN state_aug`：decentralized actor + centralized critic。

如果 2 和 3 的主要差异集中在 hidden TV/agreement，而 coverage 接近，才能更有力地说明 centralized critic 在解决 shared-state mismatch。

