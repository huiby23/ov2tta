# 实验结果记录

最后更新：2026-04-30

本文档用于记录 `ov2` 仓库当前主线实验的结果，以及当前阶段对这些结果的解释。

## 评估约定
- 布局：`counter_circuit`
- 默认使用 figure4 指标：
  - `SP`：self-play 分数
  - `XP`：cross-play 平均分数
- 如果没有特别说明，结果都来自 `runs/` 目录下的 `reward_summary_cross*.csv`。

## 结果总表
| 方法 | 版本/设置 | SP | XP | 备注 |
|---|---:|---:|---:|---|
| PPO | CNN standard | 167.8600 | 61.4067 | 当前主基线 |
| PPO | CNN state_aug | 201.8360 | 156.8307 | 强基线 |
| MAPPO | standard | 202.0680 | 132.3973 | centralized baseline |
| MAPPO | state_aug | 223.1520 | 174.6680 | 当前最强现成基线 |
| TTAPPO v1 | no_adapt | 167.3680 | 65.9587 | 相对 PPO(CNN) 小幅提升 XP |
| TTAPPO v1 | ce_only | 167.3040 | 65.5778 | 当前在线更新无明显收益 |
| TTAPPO v1 | full | 167.3520 | 65.5680 | 当前在线更新无明显收益 |
| TTAPPO v2 temporal | no_adapt 单条 | 166.3120 | 91.8311 | 当前最强单条 TTAPPO 结果 |
| TTAPPO v2 temporal | ce_only | 2.4440 | 4.9036 | 当前在线适应崩塌 |
| TTAPPO v2 temporal | gated | 3.0520 | 5.3538 | 当前在线适应崩塌 |
| TTAPPO v2 temporal | no_adapt r1 | 200.0000 | 70.3574 | base seed=42 |
| TTAPPO v2 temporal | no_adapt r2 | 180.0000 | 86.9509 | base seed=43 |
| TTAPPO v2 temporal | no_adapt r3 | 160.0000 | 83.9069 | base seed=44 |
| TTAPPO v2 temporal | no_adapt repeats mean | 180.0000 | 80.4051 | SP std=16.3299, XP std=7.2126 |

## 主要基线

### PPO (CNN) standard
- 目录：`runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full`
- 结果：`SP = 167.8600`，`XP = 61.4067`
- 文件：`runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full/reward_summary_cross.csv`

### PPO (CNN) state_aug
- 目录：`runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full`
- 结果：`SP = 201.8360`，`XP = 156.8307`
- 文件：`runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full/reward_summary_cross.csv`

### MAPPO
- 目录：`runs/figure4_mappo_ippo_budget_match_multigpu/20260404-023250_plvdnkh1_counter_circuit_avs-full`
- 结果：`SP = 202.0680`，`XP = 132.3973`
- 文件：`runs/figure4_mappo_ippo_budget_match_multigpu/20260404-023250_plvdnkh1_counter_circuit_avs-full/reward_summary_cross.csv`

### MAPPO state_aug
- 目录：`runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full`
- 结果：`SP = 223.1520`，`XP = 174.6680`
- 文件：`runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full/reward_summary_cross.csv`

## TTAPPO v1

### TTAPPO v1 no_adapt
- 目录：`runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full`
- 结果：`SP = 167.3680`，`XP = 65.9587`
- 文件：`runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_no_adapt.csv`

### TTAPPO v1 ce_only
- 结果：`SP = 167.3040`，`XP = 65.5778`
- 文件：`runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_ce_only.csv`

### TTAPPO v1 full
- 结果：`SP = 167.3520`，`XP = 65.5680`
- 文件：`runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_full.csv`

### 对 v1 的解释
- 当前 v1 的收益主要来自训练阶段的 partner auxiliary 学习，而不是当前这版 test-time update。
- `no_adapt` 略优于 `ce_only` 和 `full`。
- 相对 `PPO (CNN) standard`，v1 基本保持了 `SP`，同时给出了一个小幅的 `XP` 正信号。

## TTAPPO v2 temporal

### 单条 v2 temporal no_adapt
- 目录：`runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full`
- 结果：`SP = 166.3120`，`XP = 91.8311`
- 文件：`runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_stage2_no_adapt.csv`

### v2 temporal 的在线适应探针
- `no_adapt`：`SP = 166.3120`，`XP = 91.8311`
- `ce_only`：`SP = 2.4440`，`XP = 4.9036`
- `gated`：`SP = 3.0520`，`XP = 5.3538`
- 文件：
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_no_adapt.csv`
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_ce_only.csv`
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_gated.csv`

### 对 v2 在线适应的解释
- temporal partner model 本身是有效的。
- 当前这版在线适应是不可用的。
- 最可能的问题是：在线更新仍然作用到了会影响 actor 的特征路径，导致 partner update 直接破坏了 actor 的表示。
- 当前策略：暂停 `v2` 的在线适应变体。

## TTAPPO v2 temporal no_adapt 重复实验

这组 repeat 使用不同的 base seed，并且全部关闭 online adaptation。

### Repeat r1
- Base seed：`42`
- 目录：`runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r1/20260409-223809_smkh3gfq_counter_circuit_avs-full`
- 结果：`SP = 200.0`，`XP = 70.3574`
- Partner diagnostics 平均准确率：`0.67834`

### Repeat r2
- Base seed：`43`
- 目录：`runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r2/20260410-004754_hs3y18x8_counter_circuit_avs-full`
- 结果：`SP = 180.0`，`XP = 86.9509`
- Partner diagnostics 平均准确率：`0.72703`

### Repeat r3
- Base seed：`44`
- 目录：`runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r3/20260410-025530_c2j5a74c_counter_circuit_avs-full`
- 结果：`SP = 160.0`，`XP = 83.9069`
- Partner diagnostics 平均准确率：`0.75062`

### Repeat 汇总
- `SP mean = 180.0000`，`SP std = 16.3299`
- `XP mean = 80.4051`，`XP std = 7.2126`

### 对 v2 no_adapt repeats 的解释
- `XP` 的正向提升是真实存在的，不是只在单条 run 上出现。
- 但方差仍然不小，尤其是 `SP` 波动明显。
- predictor 质量与性能存在相关性，这说明当前瓶颈更可能在 temporal partner modeling 分支，而不只是 PPO actor 本身。

## 当前阶段结论
1. `PPO (CNN) standard` 仍然是最干净的主基线。
2. `TTAPPO v1` 通过训练期 partner auxiliary 带来了小幅 `XP` 提升，但当前在线适应没有帮助。
3. `TTAPPO v2 temporal no_adapt` 是当前最强的 `TTAPPO` backbone。
4. `TTAPPO v2` 的在线适应变体已经暂停，因为会导致性能灾难性崩塌。
5. 下一版方法必须把 test-time update 从任何共享 actor 特征路径上移开。

## 当前下一步计划
1. 将 `v2 temporal no_adapt` 视为当前主 backbone。
2. 继续使用 partner diagnostics 分析 predictor 质量。
3. 设计 `v3`，使 test-time update 只作用在隔离的 partner-specific branch 或 memory 上。
4. 不再恢复当前这版 `v2` 的在线适应变体。

## 2026-04-20 阶段更新：v3 因果分析与 v4 结论

### TTAPPO v3 memory 因果分析（42/43/44 三条历史 run 汇总）

| 方法 | Seed Block | SP Mean | XP Mean | 结论 |
|---|---:|---:|---:|---|
| TTAPPO v3 memory_off | `42/43/44` | `168.1040` | `74.5465` | 关闭 test-time memory update |
| TTAPPO v3 state_adapt | `42/43/44` | `168.6680` | `76.3305` | 正常 test-time memory update |
| TTAPPO v3 state_adapt_gated | `42/43/44` | `168.6040` | `76.4492` | gated update，收益接近 state_adapt |
| TTAPPO v3 state_adapt_wrong_partner | `42/43/44` | `168.7080` | `76.3633` | 错误 partner action 仍几乎不掉 |
| TTAPPO v3 state_adapt_random_partner | `42/43/44` | `168.6880` | `76.3071` | 随机 partner action 仍几乎不掉 |
| TTAPPO v3 state_adapt_delayed_partner | `42/43/44` | `168.7227` | `76.3776` | 延迟 partner action 仍几乎不掉 |
| TTAPPO v3 state_readout_off | `42/43/44` | `167.2053` | `72.8376` | 关闭 memory readout 明显下降 |

### 对 v3 因果分析的解释

- `state_adapt` 相对 `memory_off` 确实存在稳定正收益，说明 test-time memory update 这条路不是假的。
- 但 `wrong/random/delayed` 和 `state_adapt` 几乎一样，说明当前收益并不依赖“正确的 partner action 语义”。
- `state_readout_off` 稳定下降，说明真正有用的是 `partner_memory -> feature readout` 这条路径。
- 因此当前瓶颈不是“memory 没接到 actor 上”，而是“memory update 语义还不够强”，memory 更像一个泛化的上下文状态，而不是严格的 partner 行为语义状态。

### TTAPPO v4 actor adapter 结果

| 方法 | SP | XP | 结论 |
|---|---:|---:|---|
| TTAPPO v4 actor adapter memory_off | `158.1920` | `62.8843` | 明显弱于 v3 |
| TTAPPO v4 actor adapter state_adapt | `159.1093` | `65.9333` | state_adapt 有增益，但整条线更差 |

### 对 v4 的解释

- 把 partner memory 调制路径前移到 actor adapter 上，确实还能保留一点 test-time adaptation 增益。
- 但整体 `SP/XP` 都明显退化，说明这条结构不适合作为当前主线。
- 因此后续主线继续保留 `v3 memory` 架构，不切到 `v4 actor adapter`。

## 当前新主线：TTAPPO v3.1 semantic memory

为了直接检验“当前 `v3` 的问题是不是 memory update 缺少 partner 语义约束”，下一条新线定为 `ttappo_v3_1_semantic_memory`：

1. 保持 `v3` 的 actor / critic / partner_memory / test-time update 结构不变。
2. 训练时保留原始正样本 forward：
   - 用真实 `partner_action` 更新 memory，并计算原本的 partner CE。
3. 在同一批数据上新增两条负样本 forward：
   - `wrong`：用本 agent 自己的动作替代 `partner_action`
   - `random`：用随机且保证不等于真实动作的动作替代 `partner_action`
4. 新增 `semantic memory margin loss`：
   - 目标是让真实动作驱动下的 partner 预测 CE，显著低于错误/随机动作驱动下的 CE
5. 第一轮验证只看：
   - `memory_off`
   - `state_adapt`
6. 第一轮默认不跑 diagnostics、不扫 gated threshold，先确认新 loss 是否真的把“正确 partner 语义”拉开。

## 2026-04-30 阶段更新：MAPPO vs PPO 的 hidden-state mismatch 诊断

本轮目标是验证：`MAPPO state_aug` 相比 `PPO state_aug` 的优势，是否可能来自 centralized critic 间接改善了不同 seed 策略之间的 hidden-state convention alignment，而不只是单纯提升 SP。

为了避免把 RNN policy 的 hidden state 重置后错误地当成 CNN readout，本轮新增了 hidden-state-aware 诊断：

- 在 self-play 轨迹中保留真实 RNN hidden state，统计每个 `(state, role)` 下的动作分布，形成每条 run 的 hidden convention profile。
- 在 cross-play pair 上只比较双方都覆盖到的 shared states，并用真实 hidden convention profile 计算 mismatch。
- 主要指标包括 `out-of-support`、`in-both-support`、`hidden action agreement`、`hidden policy TV`、`hidden symmetric KL`。

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

阶段性解释：

1. MAPPO RNN state_aug 相比 PPO RNN state_aug，不只是 SP/XP 更高，同时 coverage 与 hidden convention alignment 都更好。
2. `hidden TV` 与 `hidden agreement` 和 pair-level XP 有明显相关性，说明 shared-state mismatch 不是伪问题，确实能解释 XP 差异。
3. 但 MAPPO 同时也显著降低了 out-of-support，因此不能把 MAPPO 的全部收益单独归因于 mismatch；它同时改善了 coverage 与 convention alignment。
4. 下一步如果要验证 centralized critic 的因果作用，需要做同架构消融：保持 actor/RNN/state_aug/训练预算一致，只切换 centralized critic 与 decentralized critic。

