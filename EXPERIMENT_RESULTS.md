# Experiment Results

Last updated: 2026-04-10

This file records the main experiment results and the current interpretation of those results for the `ov2` repository.

## Evaluation Conventions
- Layout: `counter_circuit`
- Default figure4 metric:
  - `SP`: self-play score
  - `XP`: cross-play average score
- Unless noted otherwise, scores come from `reward_summary_cross*.csv` artifacts under `runs/`.

## Main Baselines

### PPO (CNN) standard
- Run dir: `runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full`
- Result: `SP = 167.8600`, `XP = 61.4067`
- Artifact: `runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full/reward_summary_cross.csv`

### PPO (CNN) state_aug
- Run dir: `runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full`
- Result: `SP = 201.8360`, `XP = 156.8307`
- Artifact: `runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full/reward_summary_cross.csv`

### MAPPO
- Run dir: `runs/figure4_mappo_ippo_budget_match_multigpu/20260404-023250_plvdnkh1_counter_circuit_avs-full`
- Result: `SP = 202.0680`, `XP = 132.3973`
- Artifact: `runs/figure4_mappo_ippo_budget_match_multigpu/20260404-023250_plvdnkh1_counter_circuit_avs-full/reward_summary_cross.csv`

### MAPPO state_aug
- Run dir: `runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full`
- Result: `SP = 223.1520`, `XP = 174.6680`
- Artifact: `runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full/reward_summary_cross.csv`

## TTAPPO v1

### TTAPPO v1 no_adapt
- Run dir: `runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full`
- Result: `SP = 167.3680`, `XP = 65.9587`
- Artifact: `runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_no_adapt.csv`

### TTAPPO v1 ce_only
- Result: `SP = 167.3040`, `XP = 65.5778`
- Artifact: `runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_ce_only.csv`

### TTAPPO v1 full
- Result: `SP = 167.3520`, `XP = 65.5680`
- Artifact: `runs/figure4_ttappo_cnn_full_true10_a40x2/20260409-135002_yz5ho4c9_counter_circuit_avs-full/reward_summary_cross_full.csv`

### Interpretation for v1
- The useful gain comes from training-time partner auxiliary learning, not from the current test-time update rule.
- `no_adapt` is slightly better than `ce_only` and `full`.
- Relative to `PPO (CNN) standard`, v1 keeps `SP` roughly unchanged and gives a small positive `XP` signal.

## TTAPPO v2 temporal

### Single v2 temporal no_adapt run
- Run dir: `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full`
- Result: `SP = 166.3120`, `XP = 91.8311`
- Artifact: `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_stage2_no_adapt.csv`

### v2 temporal online adaptation probes
- `no_adapt`: `SP = 166.3120`, `XP = 91.8311`
- `ce_only`: `SP = 2.4440`, `XP = 4.9036`
- `gated`: `SP = 3.0520`, `XP = 5.3538`
- Artifacts:
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_no_adapt.csv`
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_ce_only.csv`
  - `runs/figure4_ttappo_v2_temporal_20260409-180552/20260409-192153_sw6lrhes_counter_circuit_avs-full/reward_summary_cross_gated.csv`

### Interpretation for v2 online adaptation
- The temporal partner model itself is strong.
- The current online adaptation variants are not usable.
- The likely failure mode is that the update still touches a feature path that influences the actor, so online partner updates destroy actor-side representations.
- Current policy: pause `v2` online adaptation variants.

## TTAPPO v2 temporal no_adapt repeats

The repeat pipeline was run with different base seeds and no online adaptation.

### Repeat r1
- Base seed: `42`
- Run dir: `runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r1/20260409-223809_smkh3gfq_counter_circuit_avs-full`
- Result: `SP = 200.0`, `XP = 70.3574`
- Partner diagnostics mean accuracy: `0.67834`

### Repeat r2
- Base seed: `43`
- Run dir: `runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r2/20260410-004754_hs3y18x8_counter_circuit_avs-full`
- Result: `SP = 180.0`, `XP = 86.9509`
- Partner diagnostics mean accuracy: `0.72703`

### Repeat r3
- Base seed: `44`
- Run dir: `runs/figure4_ttappo_v2_noadapt_repeat_20260409-223756_r3/20260410-025530_c2j5a74c_counter_circuit_avs-full`
- Result: `SP = 160.0`, `XP = 83.9069`
- Partner diagnostics mean accuracy: `0.75062`

### Repeat summary
- `SP mean = 180.0000`, `SP std = 16.3299`
- `XP mean = 80.4051`, `XP std = 7.2126`

### Interpretation for v2 no_adapt repeats
- The positive `XP` signal is real and is not limited to a single run.
- However, variance is still non-trivial, especially in `SP`.
- The predictor quality correlates with performance, which suggests the current bottleneck is the temporal partner modeling branch rather than the PPO actor alone.

## Current Working Conclusions
1. `PPO (CNN) standard` remains the clean main baseline.
2. `TTAPPO v1` gives a small `XP` gain through training-time partner auxiliary loss, but current online adaptation does not help.
3. `TTAPPO v2 temporal no_adapt` is the strongest current `TTAPPO` backbone.
4. `TTAPPO v2` online adaptation variants are paused because they catastrophically collapse performance.
5. The next method version should move test-time updates off any shared actor feature path.

## Current Next-Step Plan
1. Treat `v2 temporal no_adapt` as the current main backbone.
2. Continue using partner diagnostics to analyze predictor quality.
3. Design `v3` so that test-time updates only affect an isolated partner-specific branch or memory.
4. Do not resume current `v2` online adaptation variants.
