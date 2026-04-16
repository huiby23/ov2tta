# 实验结果记录

最后更新：2026-04-10

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
