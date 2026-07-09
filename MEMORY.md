# Memory

## Scope
- The authoritative workspace for `ov2tta` lives on the remote server:
  - `/teams/ius_1663576043/hby/rl/ov2`

## Current Stable Facts
- Active server: `ssh -p 26991 root@hz-4.matpool.com`
- Active hardware: `NVIDIA RTX A6000`, 49140 MiB.
- Previous active server: `ssh -p 29183 root@hz-4.matpool.com`
- Previous active hardware: `NVIDIA A40`, 49140 MiB.
- Previous active server: `ssh -p 26728 root@hz-4.matpool.com`
- Previous A6000 server: `ssh -p 26276 root@hz-t3.matpool.com`
- Previous temporary server: `ssh -p 26595 root@hz-t3.matpool.com`
- Previous temporary hardware: `2 x Tesla V100-SXM2 16GB`; do not assume a single 32GB/48GB GPU on that old server.
- Previous main server: `ssh -p 29937 root@hz-4.matpool.com`
- Previous server: `ssh -p 29207 root@hz-4.matpool.com`
- Historical server: `ssh -p 29887 root@hz-t3.matpool.com`
- Main repository: `/teams/ius_1663576043/hby/rl/ov2`
- `C5` best-known code backup is on GitHub:
  - branch: `c5_best_backup`
  - tag: `c5_best_20260408`
- All future code work must happen in `/teams/ius_1663576043/hby/rl/ov2`.
- Code must be implemented directly on the remote authoritative workspace, not written locally and then transferred.
- Local `ov2tta` files should only keep control rules, memory, session logs, and planning notes.
- Version iteration should use Git branches/tags/commits, not new worktree directories.
- The previous `ov2_o2_clean` line should be treated as abandoned exploratory work.
- On the active A6000 server, `64 env / 16 minibatch` remains the current fair baseline config unless explicitly changed.

## Current Experiment Baseline
- `CNN-IPPO standard` is the main baseline.
- `C5` is the strongest known approximate E3T result baseline.
- The next round should restart from the `C5` code state inside `ov2`.
- For repeatable comparison, the default base seed block for the current TTAPPO line is `42,43,44`.
- Do not silently switch to a different seed block such as `42,142,242` unless the user explicitly asks for it.
- Current best TTAPPO-style line is still `v3 memory`, not `v3.1 semantic memory`.
- `v3.1` increased the relative `state_adapt` delta but did not create a meaningful `true vs wrong/random/delayed` causal gap, and it lowered absolute `SP/XP`.

## Terminology Guardrail
- `state-aug` in this project is rollout state-sampling augmentation through `initial_state_buffer`, not `agent_view_size`.
- The actual state-aug flow is: rollout current policies, collect `rollout.state_seq`, sample states at intervals, pass them as `env.ENV_KWARGS.initial_state_buffer`, and continue training.
- `agent_view_size`, `avs-full`, and `obs_view` only describe observation scope.
- Experiment tables and analysis must keep `state_aug` and `obs_view` as separate fields.

## 2026-05-23 13:22:05 Asia/Shanghai
- Corrected the repeated terminology error that conflated `state-aug` with `agent_view_size` / `avs-full`.
- Current remote reports were regenerated with explicit `state_aug` and `obs_view` columns.

## 2026-05-30 Asia/Shanghai
- Earlier temporary server changed to `ssh -p 26595 root@hz-t3.matpool.com` with `2 x V100 16GB`.
- Active server later changed to `ssh -p 26276 root@hz-t3.matpool.com` with `NVIDIA RTX A6000`, 49140 MiB.
- Verified `/teams/ius_1663576043/hby/rl/ov2` exists.
- Implemented TTAC in `experiments/overcooked_v2_experiments/ttac/`: PPO-CNN base policy plus bounded adapter delta trained jointly, with test-time adapter-only update from partner history.
- TTAC formal run `ttac_joint_adapter_state_aug_64_16_10M_seed42_10seeds_20260530_152007` completed training; `base_no_test_adapt` eval produced SP `193.22`, XP off-diagonal `159.99`.
- TTAC `ttac_true_history` eval is much slower than no-adapt because it performs adapter gradient updates during rollout. First validate whether full true-history adaptation improves XP; if it works, follow-up idea is to test practical realtime variants: prefix-only adaptation, update every K steps, shorter history, and adapter-last-layer-only updates.

## 2026-05-31 Asia/Shanghai
- Implemented TTACv2 as a new independent remote method directory:
  - `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v2/`
- Implemented GAMMA-TTAC as a new independent remote method directory:
  - `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/gamma_ttac/`
- Preserved existing `ttac/` and `gamma/` baseline directories.
- TTACv2 defaults:
  - `TTAC_ADAPTER_SCALE=0.5`
  - `TTAC_TRAIN_KL_COEF=0.005`
  - `TTAC_TEST_HIST_KL_COEF=0.0`
  - `TTAC_TEST_EGO_KL_COEF=0.01`
  - `TTAC_TEST_CUR_KL_COEF=0.01`
  - `TTAC_TEST_LR=0.003`
  - `TTAC_TEST_UPDATE_STEPS=3`
- Added four TTACv2 experiment scripts:
  - `run_ttac_v2_pipeline_64_16.sh` for relax self-play.
  - `run_ttac_v2_frozen_population_64_16.sh` for frozen MEP/TrajeDi-style trained population partners.
  - `run_ttac_v2_sync_population_64_16.sh` for MEP-style synchronized population training.
  - `run_gamma_ttac_mappo_state_aug_full.sh` where the GAMMA coordinator is TTACv2, not plain PPO.
- Sync-population definition after correction:
  - ego policy is TTACv2;
  - partner population uses ordinary PPO CNN policies trained synchronously from scratch;
  - TTAC remains the only adaptive control path.
- Smoke tests passed for self-play, frozen population, sync population, and GAMMA-TTAC.
- Full queue started on the A6000:
  - PID file: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v2_full_queue.pid`
  - main log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v2_full_queue_20260531_012443.log`
  - first W&B run: `tmtpxwwa`

## 2026-06-06 Asia/Shanghai
- Active server changed to `ssh -p 26728 root@hz-4.matpool.com`.
- Current authoritative workspace remains:
  - `/teams/ius_1663576043/hby/rl/ov2`
- Clean TTAC posthoc zero-adapter experiment used an existing strong PPO CNN state-aug checkpoint and converted it into TTACv2 with zero adapter:
  - source PPO run: `runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full`
  - converted run: `runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606`
  - conversion check: max actor-logit diff `0.0`, max value diff `0.0`, adapter delta norm `0.0`.
- TTAC posthoc zero-adapter eval results:
  - `base_no_test_adapt` / `adapter_off`: SP `201.792`, XP90 `156.764`.
  - `ttac_advantage_weighted`: SP `201.996`, XP90 `159.366`, about `+2.60` XP over no-adapt.
  - `ttac_projected_confident`: SP `201.668`, XP90 `158.672`, about `+1.91` XP over no-adapt.
- TTAC posthoc diagnostics were written to:
  - `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_posthoc_zero_adapter_diagnostics_20260606/diagnostics_summary.csv`
  - `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_posthoc_zero_adapter_diagnostics_20260606/diagnostics_summary.md`
- Diagnostics conclusion:
  - online AW and projected-confident improve XP without improving coverage;
  - agreement and policy TV move only minimally;
  - joint optimal rate increases and joint regret decreases slightly, so the positive test-time signal looks closer to joint-action quality/complementarity than pure agreement.

## 2026-06-08 Asia/Shanghai
- Active server changed to:
  - `ssh -p 29183 root@hz-4.matpool.com`
- The previous `hz-4.matpool.com:26728` link is no longer valid.
- Verified authoritative workspace:
  - `/teams/ius_1663576043/hby/rl/ov2`
- Verified GPU:
  - `NVIDIA A40`, 49140 MiB, idle at connection time.
- Current TTACv2 surrogate status:
  - `ttac_ego_advantage_weighted` has a small positive reward signal but true-history did not beat wrong/delayed controls, so it is not yet a convincing partner-semantic TTA method.
  - `ttac_contrastive_ego_aw` improved over base/old ego-AW in pair20, but still did not beat wrong/delayed controls; treat it as support-refinement evidence, not final method.
  - `ttac_semantic_ego_aw` was implemented to add a small partner-observation projected CE direction on top of contrastive ego-AW. Its pair20 validation is running on the active A40.
  - Active TTAC posthoc run remains `runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606`.
  - Next semantic-surrogate sweep is now running from `experiments/run_ttac_semantic_lambda_sweep_pair20.sh`.
  - Full 500-seed validation must only run if `reports/ttac_semantic_surrogate_sweep_20260608/lambda_sweep_summary.csv` has a passing lambda.
  - The semantic lambda sweep did not pass the true-history semantic gate; do not treat this CE-weight line as the final partner-specific TTA method.
  - Preserve `ttac_v2` as the AW/support-refinement line. New TTAC exploration should be developed only in `experiments/overcooked_v2_experiments/ttac_v3/`.
- TTAC v3 alignment amplification was implemented remotely under `experiments/overcooked_v2_experiments/ttac_v3/`; do not backport these changes into `ttac_v2`.
- v3 fixed-state target-effect audit result:
  - `v3_gated_margin` and `v3_gated_semantic` both improve Agreement and reduce Policy TV/Joint regret more than plain support AW.
  - `v3_margin_only` reduces Agreement and should be treated as a mechanism ablation, not a main candidate.
- Active v3 pair20 reward sweep:
  - report dir: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v3_alignment_sweep_20260608_173120`
  - latest log pointer: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v3_alignment_pair20_latest_log.txt`
  - If a target passes pair20 true-history gate, run `experiments/run_ttac_v3_alignment_best_full_eval.sh`.
- TTAC v3 pair20 sweep finished with no passing target.
  - Best absolute reward: `gated_semantic tau=10`, XP `162.53`, only `+0.82` over support AW and only `+0.38` over delayed.
  - `gated_margin` improved diagnostics but did not beat delayed controls clearly in reward.
  - Conclusion: v3 preserves and slightly amplifies generic AW/support-refinement, but still does not establish strong partner-specific true-history dependence.
- TTAC v4-direct is the current active experiment line.
  - Directory: `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v4_direct/`
  - Goal: replace CE/gate guessing with direct partner-conditioned joint-action surrogate update.
  - Surrogate heads: `q_eta(a_p | partner obs, previous partner-action evidence)` and `Q_omega(ego obs, partner obs, a_e, a_p)`.
  - Smoke pipeline passed end-to-end on 2026-06-09.
  - Formal pipeline started; check `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v4_direct_formal_latest_log.txt`.
- TTAC v4-direct was corrected to use K-step action history instead of one-step previous-action evidence.
  - Corrected formal report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v4_direct_history_20260609_005431`
  - Corrected formal log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v4_direct_history_formal_20260609_005431.log`
  - Surrogate validation: `q_true_ce=0.9636`, `q_wrong_ce=1.1286`, `q_delayed_ce=1.0560`, `q_random_ce=1.1091`, `joint_val_auc=0.7464`.
  - Pair20 rewards: base `160.84`, support-AW `161.71`, direct-Q+support true `161.97`, wrong `162.27`, delayed `162.24`, random `158.35`.
  - Conclusion: K-step partner history is predictive for `q_eta`, but direct-Q+support still does not establish partner-specific online adaptation because true-history does not beat wrong/delayed controls.
  - Do not run full 500-seed v4-direct eval unless a later variant passes pair20 true-history controls first.

## 2026-06-09 Open Layout Mechanism Line
- Implemented and validated a new OV2 layout `open_cramped_room_v2` on the active remote workspace.
- Motivation: test a cleaner layout with both agents in one connected walkable space, two ingredients, recipe indicator, pot, plate pile, and delivery point, so Agreement/Policy TV/OOS/Joint regret can be revalidated as XP predictors outside `counter_circuit`.
- Layout:
  ```text
  WWWWWWWWW
  W0     1W
  W  A A  W
  W   P   W
  W B   X W
  W   R   W
  WWWWWWWWW
  ```
- Registered env configs for PPO, MEP, TrajeDi, TTAC, TTACv2, TTACv3, and TTACv4-direct.
- Important environment fix: OV2 `observation_space()` previously undercounted channels on multi-ingredient default observations. Correct default channel count is `25 + 5 * num_ingredients` plus optional delivery indicator.
- Stage 0 validation and Stage 1 smoke passed:
  - validation report: `/teams/ius_1663576043/hby/rl/ov2/reports/open_cramped_room_v2_20260609-162717/layout_validation/layout_validation_summary.md`
  - smoke reports: `/teams/ius_1663576043/hby/rl/ov2/reports/open_cramped_room_v2_20260609-162717`
- Full Stage 2 mechanism pre-study was stopped after PPO CNN standard:
  - driver log: `/teams/ius_1663576043/hby/rl/ov2/logs/open_cramped_room_v2_full_20260609-163014/driver.log`
  - first run W&B: `y05gpski`
  - PPO CNN standard result: SP mean `229.91`, off-diagonal XP mean `219.18`, cross mean including self-pair `220.32`.
  - Decision: stop this layout line for now because PPO standard already has very small SP/XP gap on `open_cramped_room_v2`; it does not strongly expose the ZSC convention-mismatch failure mode we need for TTAC.
  - Stopped remote open-layout pipeline processes on 2026-06-09.

## 2026-06-09 TTAC v5 Agreement Estimator Line
- Returned TTAC back to the Agreement/Policy-TV mechanism line after recognizing v3/v4 drifted toward support refinement and joint-regret/direct-Q surrogates.
- New remote directory: `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v5_agreement_estimator/`.
- Core idea:
  - Train an estimator `q_psi(a | s_query, o_partner, h_partner)` to predict a partner checkpoint's full policy distribution on the ego/query observation.
  - This is a supervised proxy for Agreement because the target is the partner policy distribution at the same query state, not the single sampled partner action.
  - Test-time adapter loss can then align adapted ego policy to `q_psi`, plus optional support-AW.
- New scripts:
  - `utils/collect_agreement_dataset.py`
  - `utils/train_agreement_estimator.py`
  - `utils/agreement_heads.py`
  - `experiments/run_ttac_v5_agreement_estimator_pipeline.sh`
- Smoke run passed end-to-end:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_agreement_estimator_smoke2_20260609_174244`
  - smoke history gate failed, especially delayed-history KL slightly beat true-history, but smoke is too small for conclusion.
- Formal run started:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_agreement_estimator_20260609_180926`
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_agreement_estimator_20260609_180926/queue.log`
  - current first stage: agreement dataset collection with `20 pairings x 100 eval seeds`, `max_transitions=250000`.
- Formal run completed:
  - estimator validation passed true-history gate: true KL `0.1466`, wrong KL `0.2264`, delayed KL `0.1528`, random KL `0.2109`.
  - pair20 reward: base `160.84`, support-AW `161.71`, v5 agreement true `167.55`, wrong `167.78`, delayed `167.60`, random `167.70`.
  - target-effect quick audit (4 pairs) baseline metrics: Agreement `0.7270`, Policy TV `0.2520`, Joint optimal `0.2734`, Joint regret `0.2388`.
  - target-effect deltas: v5 agreement true `+0.00625` Agreement, `-0.00105` Policy TV, `+0.00391` Joint optimal, `+0.00170` Joint regret; v5 agreement+support `+0.01211` Agreement, `-0.00085` Policy TV, `+0.00781` Joint optimal, `+0.00109` Joint regret.
  - corrupted histories are weaker mechanistically in quick audit: wrong-history gives `-0.00313` Agreement and `+0.00776` Joint regret, but reward still improves in pair20, suggesting the reward gain partly comes from robust policy projection/refinement and the reward eval is not yet a clean partner-specific test.
- TTAC v5.1 agreement amplification line:
  - directory: `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v5_1_agreement_amp/`
  - adds recent-buffer multi-query agreement loss and confidence-gated agreement loss.
  - reward sweep script: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_1_amp_sweep.sh`
  - history-swap fixed-state audit script: `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v5_1_agreement_amp/utils/history_swap_target_audit.py`
  - formal sweep running in `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_1_agreement_amp_20260609_233042`.
  - early partial results on 8 pairings x 50 seeds: base `175.8`, support-AW `174.75`, mild multiquery true `175.75`, mild multiquery wrong `176.45`.
- Formal v5.1 reward sweep completed:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_1_agreement_amp_20260609_233042`
  - config: `8 pairings x 50 eval seeds`; this subset is not directly comparable to earlier pair20 numbers.
  - base no-adapt: `175.80`; v5 support-AW: `174.75`.
  - best observed reward variants:
    - mild multiquery random: `176.65`
    - conf confident random: `176.65`
    - mild confident support: `176.55`
    - mild multiquery support: `176.50`
    - mild multiquery wrong: `176.45`
    - amp confident delayed: `176.35`
    - amp confident true: `176.30`
  - Decision: v5.1 did not pass the partner-specific criterion. Amplification improves reward by about `+0.5` to `+0.9` over the same-subset base, but true-history does not reliably beat wrong/random/delayed histories.
- v5.1 history-swap fixed-state audit completed:
  - output: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_1_agreement_amp_20260609_233042/history_swap_target`
  - true history deltas: Agreement `-0.00664`, Policy TV `+0.00443`, Joint optimal `+0.00781`, Joint regret `+0.00642`, adapter action-change `0.0578`.
  - swapped history deltas: Agreement `-0.02617`, Policy TV `+0.02338`, Joint optimal `-0.00781`, Joint regret `+0.01803`, adapter action-change `0.0799`.
  - Interpretation: partner history contains semantically meaningful information because swapped history is mechanistically much worse, but the current adapter update converts that signal poorly and does not yield true-history-specific reward gains.

## 2026-06-12 TTAC v5.2 Reward-Critical Agreement Selection
- New remote method directory:
  `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/`
- v5.2 keeps old v5's latest-query agreement update as the main line and treats v5.1 multi-query/confident as failed amplification attempts unless proven otherwise.
- Implemented v5.2 eval modes:
  - `ttac_v5_2_latest` plus wrong/random/delayed controls.
  - `ttac_v5_2_tv_gate` plus controls.
  - `ttac_v5_2_value_tv_gate` plus controls.
  - `ttac_v5_2_change_tv_gate` plus controls.
- Added formal scripts:
  - `experiments/run_ttac_v5_2_same_script_eval.sh`
  - `experiments/run_ttac_v5_2_paired_attribution.sh`
  - `experiments/run_ttac_v5_2_gate_sweep.sh`
  - `experiments/run_ttac_v5_2_pipeline.sh`
  - `experiments/run_ttac_v5_2_best_full_eval.sh`
- Smoke pipeline passed end-to-end on the active RTX 3090:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_state_selection_20260612_v52_smoke2`
  - code checks passed for v5.2 policy, visualizer, and paired attribution audit.

## 2026-07-05 TTAC v5.8 / Latent Partner Decoder Experiment Ledger
- Current July remote used in this phase:
  - `ssh -p 26991 root@hz-4.matpool.com`
  - workspace: `/teams/ius_1663576043/hby/rl/ov2`
- Full local ledger written to:
  - `/Users/huibingyu/code/research/ov2tta/EXPERIMENT_LEDGER_20260705.md`
- Full XP500 results:
  - base no test adapt: SP `201.624`, XP `156.940`, ALL `161.408`.
  - old v5.8 logit_bias: SP `199.788`, XP `163.811`, ALL `167.408`.
  - policy-bank direct blend: SP `201.448`, XP `175.924`, ALL `178.477`; strong XP but same-seed bank has answer-copying / SP-like risk.
  - latent decoder direct blend alpha=0.5: SP `191.312`, XP `161.995`, ALL `164.926`.
  - latent decoder + logit_bias: SP `200.724`, XP `163.264`, ALL `167.010`.
- Full 1-ZSC on healthy Q partners only (`IQL`, `VDN`, `PQN-VDN`):
  - `1-ZSC` is the metric/protocol; `all_q_partners/both_roles` is only the report aggregation over the listed Q partner families.
  - base: 1-ZSC aggregate over healthy Q partners `57.343`.
  - old v5.8 logit_bias: 1-ZSC aggregate `62.265`.
  - latent decoder + logit_bias: 1-ZSC aggregate `62.162`.
  - query-attn qexp4 VAE direct blend: 1-ZSC aggregate `64.194`, with IQL `30.435`, VDN `60.654`, PQN-VDN `101.494`; this is the current best clean full 1-ZSC result.
- Full 5-Q aggregate including broken `QMIX/SHAQ` should not be used as the main metric:
  - 1-ZSC aggregate over the 5 Q families: base `35.861`, old v5.8 `38.884`, VAE direct `40.360`.
  - `qmix_rnn` and `shaq_ps` are near-zero partners and depress the aggregate.
- 20-pair XP highlights:
  - base `161.030`; old v5.8 logit_bias `167.020`, but wrong/random/delayed controls are also about `166.6-167.1`.
  - policy-bank direct blend true `178.330`, wrong/random/delayed `165.460/166.600/166.020`; useful control gap but not a clean strict-ZSC method.
  - query-attn qexp4 VAE direct blend `171.150`; best non-bank XP pair20.
  - query-attn qexp4 direct `168.520`; MLP qexp4 logit_bias `167.850`; latent decoder alpha=0.5 only `164.640`.
- 20-pair strict 1-ZSC highlights:
  - base 1-ZSC aggregate over healthy Q partners `81.427`; old v5.8 `88.243`.
  - query-attn qexp4 VAE direct blend `89.017`, with IQL `15.990`, VDN `55.190`, PQN-VDN `195.870`; best 20-pair strict 1-ZSC among healthy-Q tests.
  - TALENTS-style non-Q data expansion regressed badly: direct `65.790`, logit `68.020`; IQL collapses to about `0`.
  - removing FCP or quality-filtering the non-Q expansion did not fix the IQL collapse.
  - reward-weighted / reward-compatible VAE variants also did not fix the final cooperation mismatch: reward-weighted 1-ZSC aggregate `70.220`, compat 1-ZSC aggregate `67.627`, IQL still `0`.
- Offline estimator sanity is not sufficient for cooperation reward:
  - VAE qexp4 has true KL `0.1566`, true argmax acc `0.7958`, weak delayed gap `+0.0004`, but best full 1-ZSC.
  - expanded non-Q VAE has larger wrong gap `+0.0419` but worse fit true KL `0.2042`, acc `0.5457`, and much worse 1-ZSC.
- QMIX/SHAQ repair status:
  - existing VDN sanity SP100 is `180.000`, so evaluator is valid.
  - old qmix/shaq shaping/persistent/CNN/seq16 and native residual CNN-QMIX all remain SP `0.000` in sanity tests; exclude them from the main partner pool until SP is fixed.
- New partner-source implementation status:
  - implemented `iql_dueling`, `vdn_dueling`, `iql_double`, and non-PPO `a2c`.
  - smoke training and SP5 loader rollouts passed.
  - full 10M single-seed training started in parallel under `runs/partner_variants_10m_seed42_20260705`; results pending as of this memory entry.
- Current main conclusion:
  - Partner modeling is mildly effective, not solved. The best clean full 1-ZSC aggregate-over-healthy-Q gain is from base `57.343` to VAE direct `64.194` (`+6.851`) and over old v5.8 `62.265` to `64.194` (`+1.929`).
  - The improvement is partner-family dependent and often comes through PQN-VDN; IQL remains the hardest case.
  - Follow-up on new 10-seed partner variants:
    - XP500 partner self/cross: `iql_dueling` SP `150.000`, XP `45.111`; `vdn_dueling` SP `174.000`, XP `62.889`; `iql_double` SP `162.000`, XP `89.333`; `a2c` SP/XP `0.000`.
    - Full 1-ZSC / 100 seeds for query-attn qexp4 VAE direct blend on healthy new variants (`iql_dueling`, `vdn_dueling`, `iql_double`) produced aggregate `71.065`.
    - Same-pool base aggregate was `63.886`, so VAE improves by `+7.179`.
    - Per new partner family base -> VAE: `iql_dueling` `49.687 -> 53.325` (`+3.638`), `vdn_dueling` `69.465 -> 78.465` (`+9.000`), `iql_double` `72.505 -> 81.405` (`+8.900`).
    - Reports: VAE `/teams/ius_1663576043/hby/rl/ov2/reports/partner_variants_10seed_vae_only_1zsc_100seed_20260705/mixed_1zsc_summary.md`; base `/teams/ius_1663576043/hby/rl/ov2/reports/partner_variants_10seed_base_only_1zsc_100seed_20260705/mixed_1zsc_summary.md`.
    - `a2c` should remain excluded from healthy partner pools; same-pool old-v5.8 control for the 10-seed new partner variants was not completed in this follow-up.
- Formal Stage 1 same-script result:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_state_selection_20260612_v52_formal/stage1_same_script`
  - base no-adapt: `160.84`
  - old v5 agreement: `167.53`
  - v5.2 latest: `167.53`, exactly matching old v5 in the same script.
  - v5.1 multiquery: `166.22`; v5.1 confident: `166.75`.
  - corrupted controls: delayed `167.58`, random `167.68`, wrong `167.79`.
  - Decision: latest-state v5 is still stronger than v5.1, but true-history still does not beat corrupted histories. Continue to state selection, not loss amplification.
- Formal Stage 2 paired attribution completed:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_state_selection_20260612_v52_formal/stage2_paired_attribution`
  - paired episodes: `2000`; sampled timestep rows: `50000`; mean episode delta: `+6.69`.
  - beneficial / harmful / neutral episode counts: `709 / 376 / 915`.
  - beneficial states have higher target-base TV (`0.264` vs harmful `0.179`) and lower base value (`39.63` vs harmful `41.86`).
  - partner action change is not a clean positive selector (`0.747` beneficial vs `0.786` harmful).
  - Decision: TV gate and value+TV gate are worth validating; change gate is mostly an ablation.
- Formal Stage 3 gate sweep is running:
  - report target: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_state_selection_20260612_v52_formal/stage3_gate_sweep`
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_2_state_selection_20260612_v52_formal/stage3_gate_sweep/queue.log`
  - Do not run full 500-seed eval unless Stage 3 produces a mode where true-history beats base, old latest, and corrupted controls.
- Stage 4 best-full-eval entrypoint exists but should only be used after Stage 3 passes:
  - `BEST_MODE=<mode> BEST_THRESHOLD=<threshold> bash experiments/run_ttac_v5_2_best_full_eval.sh`

## 2026-06-12 QLearning non-ZSC partner bank
- User requested extra non-ZSC methods for 1-ZSC evaluation: IQL, VDN, and PQN-VDN. SAD/SAD+AUX are intentionally deferred.
- New active server for this line:
  - `ssh -p 26991 root@hz-4.matpool.com`
  - GPU: `NVIDIA RTX A6000`, 49140 MiB.
- Existing JaxMARL QLearning baselines were found:
  - `JaxMARL/baselines/QLearning/iql_cnn_overcooked.py`
  - `JaxMARL/baselines/QLearning/vdn_cnn_overcooked.py`
  - `JaxMARL/baselines/QLearning/pqn_vdn_cnn_overcooked.py`
- Important implementation note:
  - These scripts were originally old `overcooked` baselines.
  - They were patched lightly to support `overcooked_v2` by adding an `overcooked_v2` env branch and a `_manager_state` helper for `get_valid_actions`.
  - Algorithm logic was not changed.
- Smoke test passed for all three on OV2 `counter_circuit`:
  - IQL saved `iql_cnn_overcooked_v2_counter_circuit_seed42_vmap0.safetensors`.
  - VDN saved `vdn_cnn_overcooked_v2_counter_circuit_seed42_vmap0.safetensors`.
  - PQN-VDN saved `pqn_vdn_cnn_overcooked_v2_counter_circuit_seed42_vmap0.safetensors`.
- Formal queue script:
  - `/teams/ius_1663576043/hby/rl/ov2/experiments/run_qlearning_ov2_1zsc_partners.sh`
- Formal training queue started on the A6000:
  - run root: `/teams/ius_1663576043/hby/rl/ov2/runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M`
  - log dir: `/teams/ius_1663576043/hby/rl/ov2/logs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M`
  - order: IQL -> VDN -> PQN-VDN
  - config: `SEED=42`, `NUM_SEEDS=10`, `TOTAL_TIMESTEPS=10M`, `layout=counter_circuit`, `WANDB_MODE=offline`.
- These QLearning models are intended for the core non-ZSC partner bank in 1-ZSC evaluation, alongside PPO/MAPPO standard seeds. They should not be confused with population-based ZSC methods.

## 2026-06-13 QLearning non-ZSC partner bank completed
- A6000 formal QLearning queue completed at `2026-06-13 03:28:59 CST`.
- Run root:
  `/teams/ius_1663576043/hby/rl/ov2/runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M`
- Completed methods:
  - IQL CNN OV2, `NUM_SEEDS=10`, `SEED=42`, `TOTAL_TIMESTEPS=10M`.
  - VDN CNN OV2, `NUM_SEEDS=10`, `SEED=42`, `TOTAL_TIMESTEPS=10M`.
  - PQN-VDN CNN OV2, `NUM_SEEDS=10`, `SEED=42`, `TOTAL_TIMESTEPS=10M`.
- Each method saved 10 `.safetensors` checkpoints, `vmap0` through `vmap9`.
- These are training checkpoints only; SP/XP/1-ZSC evaluation still needs to be run.

## 2026-06-13 QLearning SP/XP evaluation
- Added a dedicated QLearning evaluator on the A6000 remote:
  `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning.py`
- Evaluation uses the trained Q-network with greedy argmax actions and the same all-actions-available convention used by `CTRolloutManager` for Overcooked/OV2 QLearning training.
- Formal evaluation queue:
  - script: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_qlearning_ov2_eval_1zsc_partners.sh`
  - output root: `/teams/ius_1663576043/hby/rl/ov2/reports/qlearning_ov2_1zsc_eval_20260613_qlearning_eval_500`
  - log dir: `/teams/ius_1663576043/hby/rl/ov2/logs/qlearning_ov2_1zsc_eval_20260613_qlearning_eval_500`
  - order: IQL -> VDN -> PQN-VDN
  - settings: `NUM_EVAL_SEEDS=500`, `SEED=42`, `PAIRING_BATCH_SIZE=10`, modes `sp,xp`.
- Smoke observations before full eval:
  - IQL tiny smoke returned `0`.
  - VDN tiny smoke returned SP `90`, XP `0`.
  - PQN-VDN tiny smoke returned SP `200`, XP `200`.
- Formal 500-seed SP/XP evaluation completed:
  - output root: `/teams/ius_1663576043/hby/rl/ov2/reports/qlearning_ov2_1zsc_eval_20260613_qlearning_eval_500`
  - combined summary: `/teams/ius_1663576043/hby/rl/ov2/reports/qlearning_ov2_1zsc_eval_20260613_qlearning_eval_500/combined_summary.md`
  - IQL: SP `106.0`, XP `7.111`.
  - VDN: SP `148.0`, XP `32.0`.
  - PQN-VDN: SP `200.0`, XP `152.0`.
- Note:
  - Initial PQN-VDN eval OOMed because parameters were repeated across eval seeds.
  - Evaluator was fixed to run pair-wise with a single checkpoint parameter copy per policy pair, then PQN-VDN eval completed successfully.

## 2026-06-14 Greedy core evaluation completed
- Motivation: QLearning/PQN-VDN policies are evaluated greedily, so core stochastic-policy baselines were re-evaluated with greedy argmax actions for fairer comparison.
- Output root:
  `/teams/ius_1663576043/hby/rl/ov2/reports/greedy_eval_20260614_greedy_core_500_tagged`
- Summary:
  `/teams/ius_1663576043/hby/rl/ov2/reports/greedy_eval_20260614_greedy_core_500_tagged/greedy_core_summary.md`
- Results with `500` eval seeds:
  - PPO CNN standard: SP `158.000`, XP `14.667`.
  - PPO CNN state-aug: SP `170.000`, XP `81.778`.
  - MAPPO CNN standard: SP `196.000`, XP `90.000`.
  - MAPPO CNN state-aug: SP `182.000`, XP `82.444`.
  - MEP ent=0.1 standard: SP `156.000`, XP `47.111`.
  - TrajeDi div=0.1 standard: SP `156.000`, XP `68.222`.
- Interpretation:
  - Greedy evaluation does not explain away PQN-VDN's high XP (`152.0`).
  - PQN-VDN remains much stronger than greedy PPO/MEP/TrajeDi and stronger than the greedy MAPPO CNN baselines in this batch.
- Follow-up stochastic-vs-greedy comparison on the same core runs:
  - PPO CNN standard: stochastic XP `61.407` vs greedy XP `14.667`.
  - PPO CNN state-aug: stochastic XP `156.831` vs greedy XP `81.778`.
  - MAPPO CNN standard: stochastic XP `101.300` vs greedy XP `90.000`.
  - MAPPO CNN state-aug: stochastic XP `104.871` vs greedy XP `82.444`.
  - MEP ent=0.1 standard: stochastic XP `90.458` vs greedy XP `47.111`.
  - TrajeDi div=0.1 standard: stochastic XP `97.305` vs greedy XP `68.222`.
- Updated interpretation:
  - Greedy should not replace the main evaluation for PPO-family/population methods, because many of these policies are trained and regularized as stochastic policies and their stochastic execution is materially better.
  - Greedy is useful as a deployment/ablation control and for comparing against Q-learning's natural greedy execution, but the main benchmark should report each method's natural execution mode plus a separate greedy-control column.

## 2026-06-14 PQN-VDN stochastic evaluation sweep completed
- Motivation: user asked whether PQN-VDN can be evaluated stochastically, or whether its strong XP is only a deterministic greedy artifact.
- Patched QLearning evaluator:
  `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning.py`
- Added action modes:
  - `greedy` remains the default.
  - `softmax` samples from masked Q-values with configurable temperature.
  - `epsilon_greedy` follows greedy except for random valid-action exploration with probability epsilon.
- Formal sweep output:
  `/teams/ius_1663576043/hby/rl/ov2/reports/pqn_vdn_stochastic_eval_20260614_pqn_vdn_stochastic_eval_500/pqn_vdn_stochastic_sweep_summary.md`
- Results with `500` eval seeds:
  - `softmax_t0p02`: SP `199.948`, XP `148.796`.
  - `softmax_t0p05`: SP `199.988`, XP `159.634`.
  - `softmax_t0p1`: SP `183.976`, XP `147.776`.
  - `softmax_t0p2`: SP `132.020`, XP `105.265`.
  - `softmax_t0p5`: SP `36.692`, XP `29.092`.
  - `softmax_t1p0`: SP `1.896`, XP `1.431`.
  - `eps0p01`: SP `200.000`, XP `153.773`.
  - `eps0p05`: SP `199.820`, XP `149.351`.
  - `eps0p1`: SP `186.892`, XP `141.267`.
  - `eps0p2`: SP `163.928`, XP `125.810`.
- Natural greedy reference: PQN-VDN SP `200.000`, XP `152.000`.
- Interpretation:
  - PQN-VDN is robust to near-greedy stochasticity. `eps0p01` is slightly above greedy XP, and `eps0p05` remains close.
  - Softmax is highly temperature-sensitive because Q-value scales are not calibrated probabilities. Very low temperatures remain strong; high temperatures collapse.
  - PQN-VDN's high XP is therefore not merely a brittle greedy artifact, but arbitrary high-entropy sampling is not a fair execution mode for value-based methods.

## 2026-06-12 TTAC v5.2 Stage 3 completed
- Formal Stage 3 gate sweep completed on the 3090 server:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_state_selection_20260612_v52_formal/stage3_gate_sweep/gate_sweep_summary.md`
  - GPU idle after completion.
- Key result:
  - base no-adapt: `160.840`
  - latest v5.2: `167.530`
  - best TV gate true-history: `167.890` at threshold `0.08`
  - matching corrupted controls around the best setting: delayed `167.450`, random `167.850`, wrong `167.810`
- Interpretation:
  - TV-gated latest update slightly improves over old latest, but the true-history advantage over corrupted histories is tiny and far below the planned `>=2 XP` or `>=0.75` pair win-rate success criterion.
  - value+TV gates are harmful, dropping to roughly `165-166`.
  - change+TV gates are worse than plain TV gates.
  - Do not run v5.2 Stage 4 full 500-seed eval unless explicitly requested; current Stage 3 did not pass the partner-specific criterion.

## 2026-06-14 Evaluation standard tightened
- User clarified that future comparable method results must be full evaluation, not pair20 screening:
  - full `90` off-diagonal cross-play pairings unless explicitly marked otherwise;
  - `500` eval seeds;
  - pair20/pair50 results are smoke/screening only and should not be compared directly to full baseline tables.
- Started full TTAC v5/v5.2 candidate evaluation on the A6000 remote:
  - server: `ssh -p 26991 root@hz-4.matpool.com`
  - wrapper log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_2_full_eval_20260614_latest_then_tv0p08.log`
  - first report dir: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_full_eval_20260614_latest_500`
  - second report dir: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_full_eval_20260614_tv0p08_500`
- Queue order:
  - `ttac_v5_2_latest`: base, true, wrong, random, delayed.
  - `ttac_v5_2_tv_gate` at threshold `0.08`: base, true, wrong, random, delayed.
- Settings:
  - `EVAL_NUM_SEEDS=500`
  - `EVAL_MAX_PAIRINGS=0` for all full pairings
  - `EVAL_BATCHES=10`
  - `SEED=42`

## 2026-06-15 TTAC v5.2 latest full eval completed
- Full `ttac_v5_2_latest` evaluation completed on A6000.
- Report dir:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_full_eval_20260614_latest_500`
- Full evaluation split:
  - SP: 10 diagonal self-pairings, 500 seeds each.
  - XP: 90 off-diagonal cross-pairings, 500 seeds each.
- Results:
  - `base_no_test_adapt`: SP `201.836`, XP `156.764`.
  - `ttac_v5_2_latest` true history: SP `201.436`, XP `164.209`.
  - `ttac_v5_2_latest_wrong_history`: SP `201.556`, XP `163.795`.
  - `ttac_v5_2_latest_random_history`: SP `201.456`, XP `163.920`.
  - `ttac_v5_2_latest_delayed_history`: SP `201.348`, XP `164.236`.
- Interpretation:
  - The full-eval XP gain is real: true-history improves over base by `+7.445` XP.
  - However, corrupted controls improve by nearly the same amount; delayed is even slightly higher than true.
  - Current evidence supports a strong generic/support-refinement online update, but does not yet prove partner-specific semantic adaptation.
- The next queued stage `ttac_v5_2_tv_gate@0.08` is running:
  - base completed.
  - true completed with XP `164.162`.
  - currently running wrong-history control at the time of this note.
- Runtime investigation after stopping the low-value TV-gate controls:
  - `latest` true/wrong/random/delayed each took about 3 hours, so wrong-history is not inherently slower.
  - `tv_gate@0.08 wrong-history` was the anomalous case; it emitted only startup/XLA warmup logs before being stopped.
  - Code inspection showed wrong-history only changes the stored history action from partner action to self action. It does not add a separate expensive model path.
  - Full online eval should be refactored to run in pair chunks with partial CSV/progress writes before any future overnight sweeps.

## 2026-06-15 TTAC v5.2 mechanism attribution
- Ran offline attribution script:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_mechanism_attribution_20260615/mechanism_attribution_summary.md`
- Inputs:
  - full-eval reward CSVs for base/true/wrong/random/delayed latest modes;
  - agreement dataset from `reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz`;
  - v5 agreement estimator checkpoint.
- Pair-level reward deltas:
  - true XP delta `+7.445`.
  - wrong XP delta `+7.031`, pair-delta correlation with true `0.988`.
  - random XP delta `+7.156`, pair-delta correlation with true `0.990`.
  - delayed XP delta `+7.472`, pair-delta correlation with true `0.995`.
- Target/update mechanism:
  - true-vs-wrong target TV mean `0.120`, target argmax same `0.885`, logit-gradient cosine `0.812`.
  - true-vs-random target TV mean `0.112`, target argmax same `0.897`, logit-gradient cosine `0.817`.
  - true-vs-delayed target TV mean `0.063`, target argmax same `0.942`, logit-gradient cosine `0.907`.
  - Despite raw action-history corruption, one-step proxy action-change overlap is about `0.988-0.993` across corrupted histories.
- Raw corruption strength:
  - true-vs-wrong history token match only `0.231`; last-action match `0.186`.
  - true-vs-random token match `0.214`; delayed token match `0.307`.
- Interpretation:
  - Corrupted histories are genuinely different, so this is not because controls are accidentally identical to true history.
  - The estimator/update path is only weakly partner-history-specific in decision-relevant ways: different histories produce targets and logit-gradient directions that remain highly aligned, and they change almost the same actions.
  - The v5.2 gain is best described as generic online support/convention refinement, not clean partner-semantic adaptation.

## 2026-06-15 TTAC v5.2 XP amplification sweep launched
- Goal: explore whether the v5.2 online agreement/support-refinement XP gain can be amplified by increasing online update strength.
- Server:
  `ssh -p 26991 root@hz-4.matpool.com`
- Remote script:
  `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_amplify_xp_sweep.sh`
- Local copy:
  `/Users/huibingyu/code/research/ov2tta/tools/run_ttac_v5_2_amplify_xp_sweep.sh`
- Report/log dirs:
  - `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_20260615_1502`
  - `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_2_xp_amplify_20260615_1502`
- Fixed evaluation subset:
  - `max_pairings=20`, all off-diagonal XP pairs, no SP diagonal.
  - `num_eval_seeds=100`, `eval_batches=10`, `seed=42`.
- Sweep:
  - base once: `base_no_test_adapt`.
  - `ttac_v5_2_latest` with `TTAC_V5_AGREEMENT_COEF = 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0`.
  - Other online settings unchanged: `lr=0.003`, `update_steps=3`, `history_len=50`, `ego/cur KL=0.01`, `hist KL=0.0`, `support_coef=1.0`.
- First observed result:
  - pair20 base XP `160.840`.
  - `coef=0.25` running at the time of note; GPU utilization about `93-94%`, memory about `2.3GB`.

## 2026-06-15 TTAC v5.2 XP amplification first sweep completed
- Report:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_20260615_1502/xp_amplify_sweep_summary.md`
- Pair20 off-diagonal XP results:
  - base: `160.840`
  - coef `0.25`: `166.500`
  - coef `0.5`: `167.010`
  - coef `1.0`: `167.530`
  - coef `2.0`: `168.020`
  - coef `5.0`: `168.140`
  - coef `10.0`: `168.740`
  - coef `20.0`: `168.960`
- Interpretation:
  - Increasing online agreement/update strength clearly amplifies the pair20 XP signal.
  - The old `coef=1.0` result is reproduced exactly at `167.530`, so the sweep is aligned with prior v5/v5.2 screening.
  - The best point is still at the sweep boundary (`20.0`), so the update does not yet show an over-update collapse in this pair20 setting.
- Launched a second high-strength sweep:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_high_20260615_1550`
  - coefs: `30.0, 50.0, 100.0`
  - same pair20/100-seed/off-diagonal setting.

## 2026-06-15 TTAC v5.2 XP amplification high sweep completed
- Report:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_high_20260615_1550/xp_amplify_decision.md`
- Additional pair20 XP results:
  - coef `30.0`: `168.530`
  - coef `50.0`: `168.030`
  - coef `100.0`: `167.300`
- Decision:
  - best observed coefficient is `20.0` with pair20 XP `168.960`.
  - The curve rises through `20.0` and falls at `30/50/100`, so `20.0` is a real observed local peak rather than just an unchecked boundary.
- Started full validation for `coef=20.0`:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_full_coef20_20260615_1615`
  - logs: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_2_xp_amplify_full_coef20_20260615_1615`
  - settings: `RUN_BASE=0`, `TAG_PREFIX=v5_2_amp_full_coef20`, `COEFS=20.0`, `EVAL_NUM_SEEDS=500`, `EVAL_MAX_PAIRINGS=0`, `EVAL_BATCHES=10`.
  - It runs true-history `ttac_v5_2_latest` only; compare against existing full base XP `156.764` and existing coef=1 full true XP `164.209`.
- Prepared but did not start full corrupted controls for `coef=20.0`:
  - local: `/Users/huibingyu/code/research/ov2tta/tools/run_ttac_v5_2_full_coef20_controls.sh`
  - remote: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_full_coef20_controls.sh`
  - modes: `ttac_v5_2_latest_wrong_history`, `ttac_v5_2_latest_random_history`, `ttac_v5_2_latest_delayed_history`
  - intended use: only launch if full true-history `coef=20` improves over the old full `coef=1` result enough to justify controls.
- Added and started watcher:
  - local: `/Users/huibingyu/code/research/ov2tta/tools/watch_ttac_v5_2_coef20_full_then_controls.sh`
  - remote: `/teams/ius_1663576043/hby/rl/ov2/experiments/watch_ttac_v5_2_coef20_full_then_controls.sh`
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/watch_ttac_v5_2_coef20_full_then_controls_20260615.log`
  - report-side log: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_full_coef20_20260615_1615/watch_coef20_full_then_controls.log`
  - behavior: poll every 300s for `reward_summary_cross_v5_2_amp_full_coef20_coef20p0.csv`, generate split SP/XP summary, and automatically launch full wrong/random/delayed controls only if true-history full XP exceeds old `coef=1` full XP `164.209`.

## 2026-06-15 TTAC v5.2 coef=20 full true-history completed
- Full true-history validation finished at about `2026-06-15 19:14 CST`.
- Summary:
  - `xp_mean`: `165.768`
  - `sp_mean`: `200.620`
  - `all_mean`: `169.253`
  - `xp_pairs`: `90`
  - `sp_pairs`: `10`
  - old full coef=1 XP: `164.209`
  - delta vs old coef=1: `+1.559` XP
- Source CSV:
  `/teams/ius_1663576043/hby/rl/ov2/runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606/reward_summary_cross_v5_2_amp_full_coef20_coef20p0.csv`
- Summary file:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_full_coef20_20260615_1615/coef20_full_true_split_summary.md`
- Watcher condition passed (`165.768 > 164.209`) and automatically launched full corrupted-history controls:
  - wrapper PID observed: `16646`
  - active mode at check: `ttac_v5_2_latest_wrong_history`
  - control report dir: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_xp_amplify_full_coef20_controls_20260615_auto`
  - control log: `/teams/ius_1663576043/hby/rl/ov2/logs/ttac_v5_2_xp_amplify_full_coef20_controls_20260615_auto.nohup.log`
- Interpretation:
  - Increasing agreement coefficient from `1` to `20` transfers from pair20 to full eval, but the full gain is smaller than pair20: old full true `164.209` -> coef20 full true `165.768`.
  - Need wrong/random/delayed controls before claiming partner-specificity; current result only proves stronger online refinement improves full XP.

## 2026-06-16 TTAC v5.2 coef=20 controls and local-state refinement plan
- Full corrupted-history controls for coef=20 completed:
  - true-history XP: `165.768`
  - wrong-history XP: `165.121`
  - random-history XP: `165.109`
  - delayed-history XP: `165.684`
- Interpretation:
  - coef=20 strengthens the full-eval online refinement signal, but true-history is only slightly above corrupted controls.
  - The more defensible current claim is not "precise partner-action semantic adaptation"; it is "test-time local convention/support refinement on states induced by the current interaction."
- Implemented minimal new policy modes in remote `experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/policy.py`:
  - `ttac_v5_2_prev_query`
  - `ttac_v5_2_oldest_query`
  - `ttac_v5_2_pseudorandom_query`
  - Existing `ttac_v5_2_latest` behavior is unchanged.
- Added local/remote script:
  - local: `/Users/huibingyu/code/research/ov2tta/tools/run_ttac_v5_2_local_state_refinement_pair20.sh`
  - remote: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_local_state_refinement_pair20.sh`
- Smoke test passed:
  - `2` pairings x `5` eval seeds.
  - base XP `154.000`, latest XP `162.000`, prev-query XP `158.000`.
- Launched formal pair20 local-state refinement experiment:
  - remote report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_local_state_refinement_20260616_1040_local_state_refinement`
  - remote log: `/teams/ius_1663576043/hby/rl/ov2/logs/run_ttac_v5_2_local_state_refinement_20260616_1040.nohup.log`
  - wrapper PID observed: `17602`; active eval child at launch: `17612`.
  - settings: `20` off-diagonal pairings, `100` eval seeds, `TTAC_V5_AGREEMENT_COEF=20.0`, `history_len=50`, `update_steps=3`, `lr=0.003`.
  - planned comparisons: base, latest, prev-query, oldest-query, pseudorandom-query, multiquery, plus TV/value/change gates with thresholds `0.03/0.05/0.08/0.12`.

## 2026-06-16 TTAC v5.2 local-state refinement pair20 completed
- Report:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_local_state_refinement_20260616_1040_local_state_refinement/local_state_refinement_summary.md`
- Pair20/100-seed XP:
  - base no-adapt: `160.840`
  - oldest query: `162.740`
  - pseudorandom query: `165.730`
  - multiquery: `166.330`
  - change-TV gate best: `168.120`
  - value-TV gate best: `166.560`
  - latest: `168.960`
  - prev query: `169.110`
  - TV gate threshold `0.05`: `169.190`
  - TV gate threshold `0.03`: `169.440`
- Interpretation:
  - Recent local states are the important update source: latest/prev clearly outperform oldest, pseudorandom, and multiquery.
  - A simple target-base TV gate provides the best pair20 result, while value and action-change gates underperform.
  - Best candidate for full eval is `ttac_v5_2_tv_gate`, threshold `0.03`, coef `20`.
- Launched true-only full eval for the best candidate:
  - script local: `/Users/huibingyu/code/research/ov2tta/tools/run_ttac_v5_2_tv003_coef20_full_true_only.sh`
  - script remote: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_tv003_coef20_full_true_only.sh`
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_tv003_coef20_full_true_20260616_1342`
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/run_ttac_v5_2_tv003_coef20_full_true_20260616_1342.nohup.log`
  - mode: `ttac_v5_2_tv_gate`
  - threshold: `0.03`
  - coef: `20.0`
  - setting: `500` eval seeds, full cross pairings, true-only; no repeated base or corrupted controls.

## 2026-06-16 Functional evaluator / git backup note
- Remote `v3` branch now has local commits:
  - `f669c80` backup of current TTAC experiment/eval tooling.
  - `e31f99a` functional TTAC v5.2 evaluator.
  - `03a9cfb` functional eval switch for the TTAC sharded runner.
- GitHub push from the remote server failed because the server lacks GitHub SSH/HTTPS credentials.
- A git bundle backup exists locally:
  `/Users/huibingyu/code/research/ov2tta/backups/ov2_v3_03a9cfb_20260616.bundle`
- Functional TTAC v5.2 evaluator is available only when `--functional_eval` is passed.
- Sharded full eval can use it with `FUNCTIONAL_EVAL=1`.
- Supported modes: base no-adapt, latest, prev-query, oldest-query, pseudorandom-query, and TV-gate.
- Smoke equivalence against the old object evaluator passed on CPU for base and TV-gate.
- Optimization commit:
  - `9afb62d` optimizes the functional TTAC v5.2 eval path.
  - Latest local bundle: `/Users/huibingyu/code/research/ov2tta/backups/ov2_v3_9afb62d_20260616.bundle`.
  - Key changes: selected-query-only agreement estimator, skip zero-coef KL forwards, base no-adapt skips updates, functional no-viz reward-only rollout.
  - CPU-only functional/sharded smoke passed after forcing `JAX_PLATFORMS=cpu`.
- Pair-batched fast runner:
  - `734075c` adds `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_full_eval_functional_fast.sh`.
  - It uses `FUNCTIONAL_EVAL=1`, `EVAL_BATCHES=1`, `SHARD_SIZE=10`, `PARALLEL_JOBS=1` by default so each shard batches all 10 pairings in one JAX batch.
  - Latest local bundle: `/Users/huibingyu/code/research/ov2tta/backups/ov2_v3_734075c_20260616.bundle`.

## 2026-06-16 TTAC v5.2 TV-gate full eval
- Full sharded eval completed and was manually merged:
  - mode `ttac_v5_2_tv_gate`
  - TV threshold `0.03`
  - agreement coef `20.0`
  - `100` pairings x `500` eval seeds = `50,000` rows
- Result:
  - XP `165.748`
  - SP `200.456`
  - all mean `169.218`
- Interpretation:
  - Similar to latest coef=20 full XP `165.768`; TV gate does not improve full eval despite pair20 best `169.440`.
  - Still substantially above base no-adapt full XP `156.764`.

## 2026-06-16 TTAC v5.2 attribution audit
- Added and committed remote attribution tooling:
  - commit `d40de35`: `Add TTAC v5.2 attribution audit`
  - runner: `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ttac_v5_2_attribution_audit.sh`
  - local bundle backup: `/Users/huibingyu/code/research/ov2tta/backups/ov2_v3_d40de35_20260616_170144.bundle`
- Attribution report:
  `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_attribution_tv003_full_vs_base_20260616_v2/paired_attribution_summary.md`
- Compared:
  - base no-adapt CSV: `reward_summary_cross_v5_2_full_base_base_no_test_adapt.csv`
  - TTAC CSV: `reward_summary_cross_v5_2_tv003_coef20_full_sharded_20260616_1413_merged.csv`
- Full paired reward delta:
  - paired episodes: `50,000`
  - mean delta: `+7.9472`
  - episode beneficial/harmful/neutral: `18,228 / 8,070 / 23,702`
  - policy pair beneficial/harmful/neutral: `85 / 15 / 0`
- Pair-level extremes:
  - strongest positive pairs include `cross-1_8 +36.32`, `cross-8_1 +33.12`, `cross-7_4 +25.88`.
  - strongest negative pairs include `cross-2_2 -5.04`, `cross-6_6 -3.44`, `cross-3_3 -3.04`, `cross-9_9 -2.64`.
- Fast state-selection audit on the agreement dataset found:
  - beneficial pairs have higher `latest_target_base_tv` (`0.2592` vs `0.2121`).
  - beneficial pairs have lower `base_value` (`39.09` vs `42.08`).
  - `partner_action_changed` is not a good positive trigger (`0.7306` beneficial vs `0.7712` harmful).
  - on the 20 pairs covered by the state-feature dataset, correlation with pair delta was strong for `latest_target_base_tv` (`Pearson 0.8477`, `Spearman 0.7111`) and negative for `base_value` (`Pearson -0.7302`, `Spearman -0.5079`).
- Interpretation:
  - The current TTAC signal is best described as local support/refinement: it helps when the learned agreement target disagrees with base policy and the base policy/value is weak.
  - This still does not prove precise partner-specific semantic adaptation; wrong/corrupted-history controls remain a critical open issue.
  - Next useful step is not more blind coefficient tuning, but rollout-level instrumentation: record actual base/adapted logits, action changes, gate decisions, and future reward-to-go on the same TTAC trajectory.

## 2026-06-16 TTAC v5.2 corrected mechanism readout
- Correction:
  - `+7.9472` is the all-pair TTAC-base delta over `100` pairs, not XP-only.
  - full XP delta for TV-gate `0.03`, coef `20` is `165.748 - 156.764 = +8.984`.
  - full SP delta is `200.456 - 201.836 = -1.380`.
- Added commit:
  - `d847345`: `Add TTAC v5.2 target-effect audit modes`
  - local bundle: `/Users/huibingyu/code/research/ov2tta/backups/ov2_v3_d847345_20260616_172318.bundle`
- Quick target-effect audit for current v5.2 modes:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/ttac_v5_2_current_target_effect_quick_coef20_tv003_20260616/target_effect_summary.csv`
  - setting: `8` pairs, `128` shared-state samples, `agreement_coef=20`, `tv_threshold=0.03`
- Target-effect changes:
  - `v5_2_latest`: agreement `-0.0115`, policy TV `+0.00866`, joint optimal `+0.00391`, joint regret `+0.00386`, action-change rate `0.0819`.
  - `v5_2_tv_gate`: agreement `-0.0127`, policy TV `+0.00879`, joint optimal `+0.00293`, joint regret `+0.00489`, action-change rate `0.0816`.
- Interpretation update:
  - Current reward gain is not explained by a measured Agreement increase; Agreement actually drops in this quick audit.
  - The update appears to move policy away from the base policy in a way that can increase XP, but it may also increase policy mismatch and joint regret on sampled shared states.
  - The next real mechanism question is what local support/refinement is being improved if not Agreement: likely escaping weak base-policy local conventions, not estimating partner policy distribution.

## 2026-07-05 v5.2 latest pair20 1-ZSC check on new A40 server
- New server requested by user:
  - `ssh -p 27775 root@hz-4.matpool.com`
  - remote workspace: `/teams/ius_1663576043/hby/rl/ov2`
  - GPU visible to `nvidia-smi`: `NVIDIA A40`, 49140 MiB.
- Python CUDA initialization is broken on that server:
  - JAX sees only CPU and prints `cuInit(0) failed: CUDA_ERROR_NOT_INITIALIZED`.
  - PyTorch also reports CUDA unavailable.
  - Direct `ctypes` call loads `libcuda.so.1`, `cuDriverGetVersion=13000`, but `cuInit(0)=3`.
  - Therefore v5.2 online-update eval ran CPU-only and 100-seed pair20 was not practical in this session.
- Correct historical 1-ZSC pair20 slice is:
  - `max_ego_policies=2`, `max_partner_policies=5`, both roles, giving `20` role-aware pairings per Q family.
  - Earlier diagnostic slice `10 ego x 1 partner x 2 roles` is not comparable because it overweights partner `run_0`.
- Completed aligned seed10 quick 1-ZSC pair20 for `ttac_v5_2_latest`:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/v52_latest_pg2q_pair20_1zsc_seed10_2x5_20260705`
  - eval seeds: `10` per pairing/role, not formal `100`.
  - same-run base aggregate over `IQL/VDN/PQN-VDN`: `81.133`; IQL `14.800`, VDN `50.700`, PQN-VDN `177.900`.
  - `ttac_v5_2_latest` aggregate: `84.233`; IQL `13.000`, VDN `55.300`, PQN-VDN `184.400`.
  - delta vs same-run seed10 base: aggregate `+3.100`; IQL `-1.800`, VDN `+4.600`, PQN-VDN `+6.500`.
- Interpretation:
  - The seed10 quick result suggests v5.2_latest helps on VDN/PQN-VDN but hurts IQL in this pair20 slice.
  - It remains below historical 100-seed pair20 old v5.8 logit_bias aggregate `88.243` and query-attn qexp4 VAE direct aggregate `89.017`.
  - A formal 100-seed v5.2_latest pair20 1-ZSC should be run only after Python CUDA works on the new A40 server or on a GPU-ready server.

## 2026-07-05 replacement A40 CUDA check and formal v5.2 pair20 1-ZSC
- Replacement server from user:
  - `ssh -p 29898 root@hz-t3.matpool.com`
  - hostname observed: `Pbr2nm`
  - workspace: `/teams/ius_1663576043/hby/rl/ov2`
  - GPU: `NVIDIA A40`, 49140 MiB.
- Python CUDA is healthy on this replacement server:
  - direct `cuInit(0)=0`
  - PyTorch `torch.cuda.is_available() == True`, device `NVIDIA A40`
  - JAX devices include `cuda(id=0)` and default backend is `gpu`.
- Completed formal 100-seed v5.2_latest pair20 1-ZSC:
  - report: `/teams/ius_1663576043/hby/rl/ov2/reports/v52_latest_pg2q_pair20_1zsc_100seed_2x5_20260705`
  - correct pair20 slice: `2 ego checkpoints x 5 Q partners x 2 roles = 20` role-aware pairings per Q family.
  - eval seeds: `100` per pairing/role.
  - same-run base aggregate over `IQL/VDN/PQN-VDN`: `81.793`; IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - `ttac_v5_2_latest` aggregate: `86.237`; IQL `16.360`, VDN `55.020`, PQN-VDN `187.330`.
  - delta vs same-run base: aggregate `+4.443`; IQL `+0.540`, VDN `+3.330`, PQN-VDN `+9.460`.
- Interpretation:
  - v5.2_latest does improve strict pair20 1-ZSC over base, mostly through PQN-VDN and VDN.
  - It remains below historical pair20 old v5.8 logit_bias aggregate `88.243` and query-attn qexp4 VAE direct aggregate `89.017`.

## 2026-07-05 oracle same-view policy matching diagnostic
- User clarified the intended paper direction:
  - do not imitate the partner's current executed action;
  - infer the partner policy from history;
  - compare/align policies under the same role and same input observation, i.e. use `pi_partner(. | ego/query obs)` as a counterfactual same-view adaptation target.
- Implemented diagnostic eval mode `ttac_oracle_partner_direct_blend` on replacement A40 server `ssh -p 29898 root@hz-t3.matpool.com`.
  - Code touched remotely:
    - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/policy.py`
    - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_mixed_1zsc.py`
  - Mode uses the real held-out Q partner checkpoint to compute target action probabilities on the ego/query observation, then direct-blends ego logits toward that target.
  - Default target in this diagnostic was greedy one-hot from the partner Q values, with blend clip `2.0`.
  - This is an oracle upper bound, not a deployable 1-ZSC estimator, because it accesses the test partner policy.
- Pair20 / 100-eval-seed strict 1-ZSC results against `IQL`, `VDN`, `PQN-VDN`:

| mode | aggregate | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base no test adapt | 81.793 | 15.820 | 51.690 | 177.870 |
| v5.2_latest | 86.237 | 16.360 | 55.020 | 187.330 |
| oracle same-view alpha=0.25 | 90.497 | 17.500 | 59.060 | 194.930 |
| oracle same-view alpha=0.50 | 95.050 | 20.450 | 64.460 | 200.240 |
| oracle same-view alpha=0.75 | 99.767 | 25.930 | 71.760 | 201.610 |
| oracle same-view alpha=1.00 | 103.703 | 30.700 | 81.710 | 198.700 |

- Interpretation:
  - The same-view policy-matching target is reward-useful when it is accurate: best oracle aggregate improves `+21.910` over same-run base and `+17.466` over v5.2_latest.
  - The alpha sweep is mostly monotonic on aggregate, suggesting stronger movement toward the partner same-view policy helps in this diagnostic.
  - PQN-VDN peaks at alpha `0.75`, while aggregate peaks at alpha `1.00`; future learned methods likely need confidence/gating rather than always max-strength blending.
  - The large gap between oracle same-view and learned estimator variants supports the current thesis that estimator/target reconstruction is the central bottleneck.
  - This result strengthens the paper narrative around "counterfactual same-view partner-policy projection" but must be paired with learned-estimator experiments before it can be claimed as a practical 1-ZSC method.

## 2026-07-06 classic MARL partner expansion: COMA / QPLEX / WQMIX
- User asked to go beyond methods already in the repo and try classic MARL non-ZSC networks.
- Active server used:
  - `ssh -p 26991 root@hz-4.matpool.com`
  - workspace: `/teams/ius_1663576043/hby/rl/ov2`
  - GPU check before formal run: `NVIDIA RTX A6000`, idle, 49140 MiB.
- Implemented remote files:
  - `JaxMARL/baselines/QLearning/qplex_cnn_overcooked.py`
  - `JaxMARL/baselines/QLearning/wqmix_cnn_overcooked.py`
  - `JaxMARL/baselines/QLearning/coma_cnn_overcooked.py`
  - `experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py`
  - `experiments/run_classic_marl_ov2_partners.sh`
  - Updated loaders:
    - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_mixed_1zsc.py`
    - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning_object_sp_xp.py`
- Implementation notes:
  - `qplex` and `wqmix` reuse the CNN per-agent Q policy and centralized mixer training path, so downstream partner execution remains greedy per-agent Q.
  - `qplex` uses a QPLEX-style duplex-dueling mixer: centralized `V_tot(s)` over per-agent max-Q plus positive state-dependent weights on individual advantages.
  - `wqmix` uses a WQMIX-style weighted TD objective: full weight when the bootstrapped target exceeds current joint Q, otherwise `WQMIX_ALPHA=0.1`.
  - `coma` uses a decentralized CNN actor plus centralized counterfactual critic taking concatenated observations, other-agent one-hot actions, and focal-agent id. The actor advantage is `Q_i(a_i)-E_pi Q_i`.
- Smoke checks completed on A6000:
  - QPLEX / WQMIX / COMA each completed a `1 seed`, tiny `4096` timestep run and saved checkpoints.
  - Minimal SP loader checks passed for all three with `max_policies=1`, `num_eval_seeds=2`, greedy action mode. Rewards were `0.0`, expected for tiny smoke training.
  - Smoke roots:
    - QPLEX/WQMIX: `runs/classic_marl_ov2_partners_classic_smoke_20260706_005808`
    - COMA: `runs/classic_marl_ov2_partners_classic_smoke_coma_20260706_010014`
    - Loader reports: `reports/classic_smoke_qplex_loader_check`, `reports/classic_smoke_wqmix_loader_check`, `reports/classic_smoke_coma_loader_check`.
- Formal 10seed/10M queue started:
  - run root: `/teams/ius_1663576043/hby/rl/ov2/runs/classic_marl_ov2_partners_classic_marl_10seed_20260706_010208`
  - log dir: `/teams/ius_1663576043/hby/rl/ov2/logs/classic_marl_ov2_partners_classic_marl_10seed_20260706_010208`
  - queue process: `bash experiments/run_classic_marl_ov2_partners.sh` PID observed as `77268`; current child at start was QPLEX split `0/10`.
  - formal settings: `METHODS=qplex,wqmix,coma`, `NUM_SEEDS=10`, `TOTAL_TIMESTEPS=10000000`, `NUM_ENVS=64`, `NUM_STEPS=16`, `COMA_NUM_STEPS=128`, `RUN_SPXP=1`, `SPXP_EVAL_SEEDS=500`, `WANDB_MODE=disabled`.
  - Queue order: QPLEX -> WQMIX -> COMA; after each method, SP/XP500 sanity is run via `evaluate_qlearning_object_sp_xp.py`.
- Next checks:
  - First verify QPLEX/WQMIX/COMA SP is nonzero before using them as healthy partner pools.
  - If at least one family has reasonable SP/XP, run base vs query-attn qexp4 VAE direct blend 1-ZSC on the healthy subset.

## 2026-07-06 fixed-blend estimator improvement: LPR contrast and prototype posterior
- User requested implementing the fixed direct-blend estimator plan:
  - keep adapter fixed as output-space direct logit blend with `alpha=0.5`, `clip=2.0`;
  - train estimator from PG-only data;
  - keep `IQL/VDN/PQN-VDN` strictly for 1-ZSC testing.
- Implemented remotely on replacement A40:
  - server: `ssh -p 29898 root@hz-t3.matpool.com`
  - workspace: `/teams/ius_1663576043/hby/rl/ov2`
  - touched files:
    - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/collect_pg_mixed_latent_decoder_dataset.py`
    - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/train_latent_partner_decoder.py`
    - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/policy.py`
    - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_mixed_1zsc.py`
- Code changes:
  - collector now writes `ego_policy_id`, `partner_policy_id`, `ego_family`, `partner_family`, and supports `--ppo_policy_indices` / `--mappo_policy_indices`.
  - latent decoder trainer now supports:
    - `--query_expansion_scope partner_policy`
    - `--cross_context_fraction`
    - `--contrastive_coef`
    - `--contrastive_temperature`
    - `--partner_balanced_batch`
    - embedding retrieval diagnostic by `partner_policy_id`.
  - added eval mode `ttac_policy_bank_posterior_blend` plus wrong/random/delayed controls.
  - policy-bank posterior is the existing bank likelihood posterior/mix behavior exposed under a non-cheating diagnostic name; first implementation uses PPO-compatible bank params only.
- Smoke checks passed:
  - collector smoke: `reports/lpr_contrast_collector_smoke_20260706`, 800 transitions and new policy-id fields verified.
  - trainer smoke: `reports/lpr_contrast_trainer_smoke_20260706`, contrast loss active and embedding retrieval runs.
  - prototype posterior eval smoke: `reports/policy_bank_posterior_smoke_20260706`.
- Formal light PG-only dataset:
  - report: `reports/lpr_pgonly_train_dataset_light_20260706`
  - train policies: PPO/MAPPO `run_0..7`.
  - pairings: 20 total, 50 eval seeds, 120,000 transitions.
  - An initial 80-pair collection was stopped because MAPPO collection was too slow for this iteration.
- LPR contrastive estimator training:
  - report: `reports/lpr_contrast_qattn_vae_pgonly_light_train_20260706`
  - config: query-attn VAE, `query_expansion_k=4`, `query_expansion_scope=partner_policy`, `cross_context_fraction=0.25`, `contrastive_coef=0.1`, `contrastive_temperature=0.1`, partner-balanced batch, 3000 steps.
  - final offline metrics:
    - true KL `0.5402`, TV `0.3059`, argmax acc `0.6674`.
    - wrong/delayed/random KL gaps were effectively zero: `+0.00003`, `+0.00000`, `-0.00009`.
    - embedding retrieval acc `0.7415`.
  - Interpretation: contrastive latent learned some partner identity, but decoder output still does not depend meaningfully on true vs corrupted history.
- Pair20 strict 1-ZSC gate:
  - report: `reports/lpr_and_prototype_true_pair20_1zsc_20260706`
  - setting: IQL/VDN/PQN-VDN, `max_ego_policies=2`, `max_partner_policies=5`, both roles, 100 eval seeds.
  - same-run base: aggregate `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - LPR contrast q-attn VAE direct: aggregate `86.183`, IQL `16.740`, VDN `49.910`, PQN-VDN `191.900`.
  - PPO prototype posterior bank: aggregate `86.730`, IQL `16.390`, VDN `56.460`, PQN-VDN `187.340`.
- Decision:
  - Neither LPR contrast nor prototype posterior passed the pair20 gate against current learned VAE reference `89.017`.
  - Controls and full eval were not run.
  - Result is useful but negative: policy-id representation alone is not enough; the decoder still collapses toward query/family prior.
  - Next estimator direction should explicitly make the predicted same-view target history-sensitive, for example by adding a history-swap/ranking loss on output KL/TV, gating direct blend by target-history sensitivity, or training on paired histories for the same query where different partner ids produce measurably different targets.

## 2026-07-06 counterfactual/disagreement estimator dataset and z-normalized VAE query-attn

- Motivation:
  - User suspected `query_obs` is too strong and history too weak in current estimator training.
  - We reconstructed the estimator dataset so that the same `query_obs` is paired with different partner histories and different same-view teacher targets.
  - This forces the estimator to use history; otherwise it cannot fit conflicting targets for the same query.
- Implementation on replacement A40:
  - server: `ssh -p 29898 root@hz-t3.matpool.com`
  - workspace: `/teams/ius_1663576043/hby/rl/ov2`
  - new script:
    - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/build_counterfactual_latent_decoder_dataset.py`
  - trainer changes:
    - added same-query output sensitivity diagnostics:
      - `same_query_output_tv`
      - `same_query_target_tv`
      - `same_query_output_target_tv_corr`
      - `same_query_output_target_tv_ratio`
    - added VAE/query-attn and decoder latent flags:
      - `--vae_use_query_attn`
      - `--decoder_z_norm`
      - `--decoder_z_scale`
  - latent decoder changes:
    - VAE query-attn flag lets query embedding attend history tokens before decoding.
    - decoder z normalization prevents tiny VAE means from being ignored.
    - flags are saved in new estimator `.npz` files and are backward compatible with old checkpoints.
- Dataset:
  - source: `reports/lpr_pgonly_train_dataset_light_20260706/pg_mixed_latent_decoder_dataset.npz`
  - output: `reports/cf_disagreement_pgonly_dataset_20260706/counterfactual_latent_decoder_dataset.npz`
  - train policies: PPO/MAPPO `run_0..7`.
  - no IQL/VDN/PQN-VDN trajectory, checkpoint logits, or action labels used for estimator training.
  - rows: `80,000`, split into `40,000` counterfactual rows and `40,000` original rows.
  - candidate queries: `20,000`.
  - selected high-disagreement queries: `2,500`.
  - policies per query: `16`.
  - selected mean pairwise teacher TV: `0.5916`.
  - selected argmax disagreement mean: `0.7034`.
- Negative control without z-norm/query-attn flags:
  - report: `reports/cf_disagreement_qattn_vae_pgonly_train_20260706`
  - offline:
    - true KL `0.9741`, TV `0.4337`, argmax acc `0.5056`.
    - wrong/delayed/random gaps were effectively zero.
    - same-query output TV `0.00076`, target TV `0.5934`, ratio `0.0013`.
    - embedding retrieval acc `0.8135`.
  - Interpretation:
    - latent identity was learned, but decoder output almost completely ignored the history latent.
- Fixed estimator with `--vae_use_query_attn --decoder_z_norm --decoder_z_scale 1.0`:
  - report: `reports/cf_disagreement_qattn_vae_znorm_pgonly_train_20260706`
  - offline:
    - true KL `0.4156`, TV `0.2905`, argmax acc `0.6534`.
    - wrong KL `0.5039`, delayed KL `0.4309`, random KL `0.5100`.
    - wrong/random KL gaps: `+0.0882` / `+0.0944`; delayed gap `+0.0152`.
    - same-query output TV `0.4416`, target TV `0.5934`, ratio `0.744`, correlation `0.400`.
    - embedding retrieval acc `0.6712`.
  - Interpretation:
    - This is the first estimator variant in this branch that clearly uses partner history to change same-view target output distribution.
- Pair20 strict 1-ZSC:
  - report: `reports/cf_disagreement_znorm_pair20_1zsc_20260706`
  - setting:
    - partners: `IQL`, `VDN`, `PQN-VDN`
    - `max_ego_policies=2`
    - `max_partner_policies=5`
    - both roles
    - `num_eval_seeds=100`
    - direct blend `alpha=0.5`, `clip=2.0`
  - results:
    - base aggregate `81.793`; IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
    - true-history estimator aggregate `86.187`; IQL `14.470`, VDN `52.690`, PQN-VDN `191.400`.
    - wrong-history aggregate `84.660`; IQL `14.510`, VDN `50.740`, PQN-VDN `188.730`.
    - random-history aggregate `84.363`; IQL `15.350`, VDN `48.810`, PQN-VDN `188.930`.
    - delayed-history aggregate `85.457`; IQL `13.890`, VDN `51.840`, PQN-VDN `190.640`.
- Decision:
  - The estimator-history-use bottleneck is partially fixed: true history is now measurably better than wrong/random/delayed in offline diagnostics and pair20 reward.
  - But pair20 reward still does not beat current learned VAE reference `89.017`.
  - Gains mainly come from PQN-VDN; IQL is worse than base, so reward alignment is still partner-family dependent.
  - Next work should not just increase history sensitivity; it should improve reward alignment or gate/regularize direct blend so bad targets do not hurt IQL.

### Direct estimator replacement diagnostic

- User asked whether we can directly use estimator predictions to cooperate with the partner.
- Instead of adding a new mode first, we ran the near-equivalent existing path:
  - mode: `ttac_v5_8_latent_decoder_blend`
  - estimator: `reports/cf_disagreement_qattn_vae_znorm_pgonly_train_20260706/latent_partner_decoder.npz`
  - `alpha=1.0`
  - `clip=10.0`
  - report: `reports/cf_disagreement_znorm_pair20_1zsc_alpha1_clip10_20260706`
  - setting: pair20 strict 1-ZSC, IQL/VDN/PQN-VDN, 100 eval seeds, both roles.
- Results:
  - base aggregate `81.793`; IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - true-history alpha1 aggregate `55.103`; IQL `7.910`, VDN `26.240`, PQN-VDN `131.160`.
  - wrong-history aggregate `49.000`; IQL `5.080`, VDN `22.100`, PQN-VDN `119.820`.
  - random-history aggregate `44.900`; IQL `4.490`, VDN `20.350`, PQN-VDN `109.860`.
  - delayed-history aggregate `53.100`; IQL `6.450`, VDN `23.720`, PQN-VDN `129.130`.
- Interpretation:
  - True history remains better than corrupted histories, so the estimator signal is real.
  - But overriding base is catastrophic because estimator errors and one-step same-view imitation are not reward-safe enough.
  - Current evidence favors keeping a conservative base-policy anchor; pure `replace_policy` is unlikely to help unless estimator accuracy/reward alignment improves substantially.

### Local decision-point benchmark

- User proposed first testing on discrete/local scenarios where cooperation is affected and partner policies disagree, before worrying about ordinary non-disagreement states.
- Implemented diagnostic-only benchmark:
  - collector output: `reports/local_decision_qdiag_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`
  - benchmark report: `reports/local_decision_benchmark_old_vs_new_20260706/local_decision_summary.md`
  - script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/evaluate_local_decision_points.py`
  - selected rows: `1834` PPO-ego / Q-partner local decision points.
  - Q partner data is used only for diagnostic evaluation, not estimator training.
- Buckets:
  - `high_partner_disagreement`: top 20% mean pairwise TV across IQL/VDN/PQN-VDN policies.
  - `high_oracle_base_tv`: top 20% TV between oracle Q partner target and base PPO policy.
  - `high_return_to_go`: top 20% logged return-to-go.
  - `high_joint_disagreement_base_tv`: top 20% product of partner disagreement and oracle-base TV.
  - `high_reward_joint`: top 20% product of disagreement, oracle-base TV, and positive return-to-go.
  - `low_oracle_base_tv`: bottom 20% ordinary states where base already matches oracle.
- All-partner KL to oracle target:
  - all: base `2.9062`, old VAE `1.6247`, new z-norm VAE `1.6726`.
  - high partner disagreement: base `3.3158`, old VAE `1.8575`, new z-norm VAE `1.7731`.
  - high oracle-base TV: base `6.9741`, old VAE `2.5701`, new z-norm VAE `2.1011`.
  - high reward joint: base `2.5501`, old VAE `1.7041`, new z-norm VAE `1.7004`.
  - low oracle-base TV: base `0.0376`, old VAE `0.4017`, new z-norm VAE `0.5461`.
- Key interpretation:
  - New z-norm estimator is indeed better on the intended local buckets where partner modeling should matter, especially high partner disagreement and high oracle-base-TV states.
  - Old VAE remains slightly better globally and in ordinary low oracle-base-TV states.
  - This supports a gated/local adaptation direction: use stronger history-sensitive estimator only when local disagreement/target-base mismatch suggests partner modeling matters; otherwise preserve base/old prior.
- Family split:
  - In high partner-disagreement states, new z-norm improves IQL/VDN strongly but old VAE remains better for PQN-VDN.
  - In high oracle-base-TV states, new z-norm improves IQL/VDN, while old VAE remains better on PQN-VDN.
  - In low oracle-base-TV states, base is nearly perfect; new z-norm especially harms IQL ordinary states.

### Sensitivity gate test for z-normalized counterfactual estimator

- Motivation:
  - Local decision diagnostics showed the new z-norm VAE is useful in high-disagreement/high oracle-base-TV states but harms ordinary states where base already matches the oracle target.
  - User suggested first trying a gate before redesigning estimator or adapter.
- Implementation:
  - Added latent decoder gate args to `evaluate_mixed_1zsc.py`.
  - Added `TTAC_LATENT_DECODER_GATE_MODE=sensitivity_no_history` in `ttac_v5_8_fast_online/policy.py`.
  - Gate signal is the TV distance between true-history estimator target and no-history estimator target:
    - `sensitivity = 0.5 * sum(abs(target_probs - no_hist_probs))`
    - `gate = sigmoid(scale * (sensitivity - threshold))`
    - effective alpha is `alpha * (min_gate + (1 - min_gate) * gate)`.
- Pair20 strict 1-ZSC report:
  - report: `reports/cf_znorm_gated_sens_t010_pair20_1zsc_20260706`
  - estimator: `reports/cf_disagreement_qattn_vae_znorm_pgonly_train_20260706/latent_partner_decoder.npz`
  - setting: `alpha=0.5`, `clip=2.0`, `history_len=50`, gate threshold `0.1`, scale `12.0`, min `0.05`.
- Results:
  - base aggregate `81.793`; IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - gated true-history aggregate `87.113`; IQL `14.650`, VDN `54.700`, PQN-VDN `191.990`.
  - gated wrong-history aggregate `85.900`; IQL `14.710`, VDN `52.840`, PQN-VDN `190.150`.
  - gated random-history aggregate `85.990`; IQL `15.250`, VDN `52.070`, PQN-VDN `190.650`.
  - gated delayed-history aggregate `85.927`; IQL `14.090`, VDN `53.160`, PQN-VDN `190.530`.
- Interpretation:
  - Gate improves over ungated z-norm VAE pair20 `86.187 -> 87.113`, so reducing ordinary-state perturbation helps.
  - It still does not beat the older qexp4 VAE direct reference `89.017`.
  - True-history beats corrupted controls by only about `1.1` reward, below the planned `+2` pair20 gate.
  - IQL remains below base, while VDN/PQN-VDN improve. The current gate is a useful direction, not a solved estimator/adaptation mechanism.

### 2026-07-06 Official EPyMARL migration sanity outcome

- Vendored official EPyMARL and official QPLEX/WQMIX modules were connected to OV2 through `third_party/epymarl/src/envs/ov2_wrapper.py`.
- Smoke tests proved the wrapper can execute, but the 500k sanity training run is not a valid partner source:
  - run root: `runs/epymarl_official_ov2_sanity_20260706_094300`
  - completed with status 0 but sparse/test stayed zero: `vdn`, `qmix`, `qplex`, `ow_qmix`
  - `coma` failed at launch with CUDA context initialization errors and was terminated with `status=143`
  - `cw_qmix` was terminated around `479k/500k` after staying zero throughout
- Treat this EPyMARL-OV2 line as failed/inconclusive, not as evidence that the algorithms themselves are unusable.
- Likely training-line causes:
  - EPyMARL uses one episode runner rather than the native JaxMARL `64 env / 16 step` parallel data regime.
  - OV2 grid observations were flattened into MLP/RNN-style inputs instead of using the CNN observation encoder used by healthy OV2 Q baselines.
  - Reward shaping was constant in the wrapper, while successful native PPO/Q-learning lines anneal shaping.
  - Common-reward handling summed both agents' shaped rewards, changing the reward scale relative to native OV2 training.
- Current healthy non-ZSC Q partner pools remain the native JaxMARL CNN lines:
  - IQL/VDN/PQN-VDN from `runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M`
  - `iql_double`, `iql_dueling`, `vdn_dueling` from `runs/partner_variants_10seed_10m_seed42_20260705`
- Current failed classic MARL additions:
  - earlier JaxMARL-style COMA/QPLEX/WQMIX training produced SP `0.000`, XP `0.000`
  - official EPyMARL VDN/QMIX/QPLEX/OW-QMIX/CW-QMIX/COMA migration also produced zero or failed
- Guardrail:
  - Do not run 1-ZSC/XP tables for any new partner method unless its SP is first nonzero and preferably comparable to existing Q baselines.
  - If COMA/QPLEX/WQMIX are still needed, re-implement them inside the native JaxMARL CNN parallel training framework and validate SP before any TTAC/ZSC evaluation.

### 2026-07-06 Native Q-learning algorithm migration

- User requested migrating Q-learning method ideas onto the effective native Q-learning base, rather than using external frameworks.
- Implemented two OV2-native PQN-based migration variants directly on the A6000 remote workspace:
  - `JaxMARL/baselines/QLearning/pqn_wqmix_cnn_overcooked.py`
    - base: healthy `pqn_vdn_cnn_overcooked.py`
    - change: WQMIX-style optimistic weighted TD loss
    - method name/subdir/prefix: `pqn_wqmix`, `pqn_wqmix`, `pqn_wqmix_cnn`
  - `JaxMARL/baselines/QLearning/pqn_soft_cnn_overcooked.py`
    - base: healthy `pqn_vdn_cnn_overcooked.py`
    - change: SHAQ/soft-Q-style soft bootstrap target
    - method name/subdir/prefix: `pqn_soft`, `pqn_soft`, `pqn_soft_cnn`
- Updated:
  - `experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py`
  - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_mixed_1zsc.py`
  - `experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning_object_sp_xp.py`
  - `experiments/run_qlearning_native_migrations.sh`
- Checks:
  - `py_compile` passed for new modules and touched evaluators.
  - import check passed; `METHOD_SPECS` contains `pqn_wqmix` and `pqn_soft`.
- Smoke run:
  - run root: `runs/qlearning_native_migrations_native_migration_smoke_20260706_134651`
  - reports:
    - `reports/qlearning_native_migrations_pqn_wqmix_spxp_native_migration_smoke_20260706_134651`
    - `reports/qlearning_native_migrations_pqn_soft_spxp_native_migration_smoke_20260706_134651`
  - purpose: training/save/load/eval path only; 65k steps, SP expectedly `0`.
- 1M single-seed sanity:
  - run root: `runs/qlearning_native_migrations_native_migration_sp1m_20260706_134944`
  - reports:
    - `reports/qlearning_native_migrations_pqn_wqmix_spxp_native_migration_sp1m_20260706_134944`
    - `reports/qlearning_native_migrations_pqn_soft_spxp_native_migration_sp1m_20260706_134944`
  - results:
    - `pqn_wqmix`: SP50 `240.000`
    - `pqn_soft`: SP50 `180.000`
  - Interpretation: both native migrations are learning and pass the nonzero-SP gate; `pqn_wqmix` is especially promising.
- Formal queue started:
  - run id: `native_migration_10seed_20260706_135346`
  - run root: `runs/qlearning_native_migrations_native_migration_10seed_20260706_135346`
  - logs: `logs/qlearning_native_migrations_native_migration_10seed_20260706_135346`
  - methods: `pqn_wqmix,pqn_soft`
  - config: `NUM_SEEDS=10 TOTAL_TIMESTEPS=10000000 NUM_ENVS=64 NUM_STEPS=16 RUN_SPXP=1 SPXP_EVAL_SEEDS=500`
  - completed at `2026-07-06 16:47:39`.
  - checkpoint count: 20 safetensors total, 10 for `pqn_wqmix` and 10 for `pqn_soft`.
  - SP/XP500 results:
    - `pqn_wqmix`: SP `182.000`, XP `115.778`.
    - `pqn_soft`: SP `216.000`, XP `169.556`.
  - Interpretation: both pass the nonzero-SP formal gate; in the full 10seed/XP500 run, `pqn_soft` is stronger than `pqn_wqmix` on both SP and XP.

### 2026-07-06 Native actor-critic migration

- User clarified the desired actor-critic line: migrate useful algorithm ideas onto a stable native OV2 AC bottom, not use failed standalone COMA/A2C reproductions.
- Implemented a first native PPO-COMA hybrid on the A6000 remote workspace:
  - new module: `JaxMARL/baselines/QLearning/ppo_coma_cnn_overcooked.py`
  - method/subdir/prefix: `ppo_coma`, `ppo_coma`, `ppo_coma_cnn`
  - base: native OV2 CNN actor-critic data path and safetensors actor checkpoint contract.
  - idea: COMA counterfactual critic plus PPO-style old log-prob tracking, clipped ratio objective, update epochs, and minibatches.
- Updated support code:
  - `experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py` now supports `ppo_coma`.
  - `evaluate_qlearning_object_sp_xp.py` can evaluate `ppo_coma`.
  - `evaluate_mixed_1zsc.py` can load `ppo_coma`, and was also fixed to include `pqn_wqmix` and `pqn_soft` for later 1-ZSC evaluation.
  - added runner: `experiments/run_ac_native_migrations.sh`.
- Validation:
  - `py_compile` passed for the new module and touched trainer/evaluators.
  - import/registry check passed for `ppo_coma`, `pqn_wqmix`, and `pqn_soft`.
  - smoke run completed:
    - run root: `runs/ac_native_migrations_ppo_coma_smoke_20260706_141933`
    - report: `reports/ac_native_migrations_ppo_coma_spxp_ppo_coma_smoke_20260706_141933`
    - purpose: train/save/load/eval path only; 512 timesteps, SP expectedly `0`.
- Queued follow-up:
  - the queued waiter was manually bypassed after Q-learning completed and the GPU became idle.
  - 1M single-seed `ppo_coma` sanity completed:
    - run root/log id: `ppo_coma_sp1m_after_q_20260706_142214`
    - config: `NUM_SEEDS=1 TOTAL_TIMESTEPS=1000000 NUM_ENVS=64 COMA_NUM_STEPS=128 NUM_EPOCHS=4 NUM_MINIBATCHES=16 RUN_SPXP=1 SPXP_EVAL_SEEDS=50`
    - report: `reports/ac_native_migrations_ppo_coma_spxp_ppo_coma_sp1m_after_q_20260706_142214`
    - result: SP50 `0.000`; XP is `nan` because only one policy was trained.
  - Interpretation: this first PPO-COMA migration did not pass the nonzero-SP gate at 1M and should not enter XP/ZSC partner evaluations yet.

### 2026-07-06 AC diversity sanity and formal queue

- User requested more diverse reinforcement-learning methods because the actor-critic family was too sparse.
- Added remote runner:
  - `experiments/run_ac_diversity_sanity.sh`
  - trains PPO/IPPO and MAPPO variants from the stable native `ppo/` and `mappo/` code paths, then runs cross-play SP/XP and writes a report.
- Smoke check:
  - `TS=ac_diversity_smoke_20260706_165937`
  - method: `ippo_cnn_standard`
  - 2 seeds, 512 timesteps, eval seeds 2
  - train/save/eval/report path passed.
- 1M/5seed/XP50 sanity:
  - report: `reports/ac_diversity_sanity_20260706_170036/summary.md`
  - results:
    - `ippo_cnn_gae090`: SP `35.200`, XP `22.080`
    - `ippo_cnn_standard`: SP `21.920`, XP `9.140`
    - `ippo_rnn_standard`: SP `21.200`, XP `1.440`
    - `ippo_cnn_clip010`: SP `4.320`, XP `3.840`
    - `ippo_cnn_ent001`: SP `4.080`, XP `3.920`
    - `mappo_cnn_world_state`: SP `0.000`, XP `0.000`
    - `mappo_cnn_local_obs`: SP `0.000`, XP `0.000`
    - `mappo_cnn_zero_world_state`: SP `0.000`, XP `0.040`
  - Interpretation:
    - The IPPO family passes the short sanity gate; `ippo_cnn_gae090` is the most promising AC candidate at 1M.
    - The MAPPO CNN critic-input ablations do not learn at this short scale, so they were not placed into the immediate formal queue.
- Formal AC diversity queue launched, then stopped:
  - run id: `ac_diversity_formal_20260706_172044`
  - PID: `25997`
  - queue log: `logs/ac_diversity_formal_20260706_172044/queue.log`
  - report target: `reports/ac_diversity_formal_20260706_172044/summary.md`
  - methods: `ippo_cnn_gae090`, `ippo_cnn_standard`, `ippo_rnn_standard`, `ippo_cnn_ent001`, `ippo_cnn_clip010`
  - config: `NUM_SEEDS=10 TOTAL_TIMESTEPS=10000000 REW_SHAPING_HORIZON=5000000 NUM_ENVS=64 NUM_STEPS=256 UPDATE_EPOCHS=4 NUM_MINIBATCHES=16 EVAL_SEEDS=500`
  - stopped at `2026-07-06 17:25:28` after user correctly noted these are not method-level AC diversity; they are PPO/IPPO family hyperparameter/architecture variants.
  - GPU was released afterward (`3 MiB`, `0%` utilization).
  - Do not treat this formal queue as a valid new-method experiment. The 1M/5seed sanity table can remain as a PPO-family ablation, but it should not be used to claim broader RL-method diversity.

### 2026-07-06 Method-level AC migrations on stable MAPPO bottom

- User requested genuinely distinct AC methods rather than more PPO/IPPO variants:
  - COMA-on-stable-PPO-bottom
  - MAAC / attention critic
  - HAPPO / HATRPO-style sequential agent update
  - IMPALA / V-trace
- Implemented a new remote trainer that reuses the stable OV2 MAPPO actor/CNN/checkpoint/evaluator contract:
  - `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ac_method_migrations/train.py`
  - saved checkpoints are MAPPO-compatible `{"actor": actor_params, "critic": critic_params}` so `mappo/utils/visualize.py` can evaluate them.
- Implemented methods:
  - `coma_ppo`: PPO actor update with COMA-style counterfactual Q critic/advantage.
  - `maac`: attention-style centralized critic over joint agent observations.
  - `happo`: HAPPO-lite sequential per-agent masked PPO updates with shared actor and centralized critic; not full HATRPO/TRPO.
  - `vtrace`: IMPALA/V-trace-style actor-critic target in the vectorized trainer; not a distributed actor-learner IMPALA system.
- Debug/validation:
  - `py_compile` and import checks passed.
  - Smoke runs passed for all four methods with `2 seeds, 512 timesteps`.
  - MAAC initially hit `LLVM ERROR: mma16816 data type not supported`; root cause was non-float joint obs entering attention critic Dense/matmul. Fixed by explicit `float32` casting and safer attention tensor shapes.
  - Existing `mappo/utils/visualize.py` successfully loaded/evaluated the MAAC smoke checkpoint.
- Added gate runner:
  - `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ac_method_migrations_gate.sh`
  - default gate: `10 seeds, 1M timesteps, NUM_ENVS=64, NUM_STEPS=256, UPDATE_EPOCHS=4, NUM_MINIBATCHES=16, eval50 SP/XP`.
- 1M gate run:
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/ac_method_migrations_gate/ac_method_migrations_gate_20260706_180127.log`
  - run root: `/teams/ius_1663576043/hby/rl/ov2/runs/ac_method_migrations_gate`
  - completed at `2026-07-06 18:14:54`.
  - results:
    - `coma_ppo`: SP `0.000`, XP `0.000`.
    - `maac`: SP `0.000`, XP `0.000`.
    - `happo`: SP `0.000`, XP `0.009`.
    - `vtrace`: SP `0.000`, XP `0.000`.
- Interpretation:
  - The method-level AC migrations now run end-to-end on the stable OV2 MAPPO bottom and use GPU efficiently.
  - None pass the nonzero-SP gate at 1M. `happo` XP `0.009` is random-level leakage, not a useful partner result.
  - Do not spend 10M/30M or 1-ZSC evaluation on these exact AC migrations yet.
  - Current positive new-method evidence remains the native Q-learning migration line, especially `pqn_soft` (10seed SP `216.000`, XP `169.556`) and `pqn_wqmix` (SP `182.000`, XP `115.778`).

### 2026-07-06 Method-level AC full-mask long training

- Follow-up after user asked whether training longer might help.
- Identified and patched a pure self-play mismatch in the AC migration trainer:
  - Added `--full_train_mask` to `/teams/ius_1663576043/hby/rl/ov2/experiments/overcooked_v2_experiments/ac_method_migrations/train.py`.
  - Added `FULL_TRAIN_MASK=1` support to `/teams/ius_1663576043/hby/rl/ov2/experiments/run_ac_method_migrations_gate.sh`.
  - Rationale: the prior 1M gate used FCP/population-style `train_mask_flat`, so pure self-play actor updates used only half of agent samples. Stable MAPPO pure SP effectively trains all agents. The new long run is therefore a "fullmask_long" condition, not the exact same condition as the earlier 1M gate.
- Full-mask smoke:
  - `happo_fullmask_smoke_20260706`, `2 seeds`, `512 timesteps`; passed.
- Long run:
  - run root: `/teams/ius_1663576043/hby/rl/ov2/runs/ac_method_migrations_fullmask_long`
  - log: `/teams/ius_1663576043/hby/rl/ov2/logs/ac_method_migrations_fullmask_long/ac_method_migrations_gate_20260706_183732.log`
  - config: `NUM_SEEDS=10 TOTAL_TIMESTEPS=10000000 REW_SHAPING_HORIZON=5000000 NUM_ENVS=64 NUM_STEPS=256 UPDATE_EPOCHS=4 NUM_MINIBATCHES=16 FULL_TRAIN_MASK=1 EVAL_SEEDS=100`.
  - completed at `2026-07-06 19:59:05`; A6000 returned idle.
- eval100 results:
  - `coma_ppo`: SP `0.000`, XP `0.000`.
  - `maac`: SP `123.840`, XP `26.202`.
  - `happo`: SP `117.700`, XP `38.820`.
  - `vtrace`: SP `0.000`, XP `0.000`.
- eval500 follow-up for positive candidates:
  - MAAC run: `/teams/ius_1663576043/hby/rl/ov2/runs/ac_method_migrations_fullmask_long/ac_method_migration_fullmask_long_maac_20260706_183732`
    - SP `123.772`, XP `26.310`, cross-self `123.796`.
  - HAPPO-lite run: `/teams/ius_1663576043/hby/rl/ov2/runs/ac_method_migrations_fullmask_long/ac_method_migration_fullmask_long_happo_20260706_183732`
    - SP `117.688`, XP `39.239`, cross-self `117.736`.
- Greedy action-selection check for AC eval500:
  - Q-learning SP/XP evaluator defaults to greedy argmax.
  - MAPPO/AC evaluator defaults to stochastic sampling unless `--greedy` is passed.
  - MAAC greedy eval500: SP `124.000`, XP `4.889`, cross-self `124.000`.
  - HAPPO-lite greedy eval500: SP `118.000`, XP `14.000`, cross-self `118.000`.
  - Interpretation: for these AC policies, greedy preserves SP but hurts XP badly; stochastic sampling is helping cross-play rather than artificially depressing it.
- Interpretation:
  - Training longer plus full self-play mask does rescue method-level AC partners for MAAC and HAPPO-lite.
  - COMA and V-trace remain zero after the same 10M/full-mask setting; do not prioritize them.
  - MAAC has stronger SP; HAPPO-lite has stronger XP and may be the better diversity partner candidate.
  - These AC partners are weaker than native Q migrations on XP (`pqn_soft` XP `169.556`, `pqn_wqmix` XP `115.778`) but add genuine actor-critic diversity and are now credible for downstream 1-ZSC/TTAC partner-pool tests.

### 2026-07-06 Accuracy-first estimator heldout experiment

- User reframed estimator progress around partner policy reconstruction accuracy, not history-sensitive gap.
- Implemented the accuracy-first tooling on the A40 server `/teams/ius_1663576043/hby/rl/ov2`:
  - unified offline evaluator: `evaluate_latent_decoder_accuracy.py`
  - hard-query dataset builder: `build_accuracy_hard_query_dataset.py`
  - prototype-prior dataset builder: `build_prototype_prior_dataset.py`
  - trainer support for `top2_acc` and optional prototype prior
  - latent-decoder ensemble loading and uncertainty-gate support in `policy.py` / `evaluate_mixed_1zsc.py`
- Strict heldout-PG dataset:
  - `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`
  - PPO/MAPPO run8-9 only, 60k rows.
- Unified heldout-PG anchor:
  - old qexp4 VAE: KL `0.4484`, TV `0.3130`, argmax acc `0.7267`, top2 `0.8508`, high-baseline-error KL `1.4184`.
  - This stricter run8/9 number should not be mixed with the older internal validation anchor KL `0.1566`, acc `0.7958`.
- Single-model candidates:
  - A qexp4 VAE rerun b512: KL `0.4672`, acc `0.6438`, top2 `0.8721`, high-error KL `1.1076`.
  - B multi-query/cross-context VAE: KL `0.5722`, acc `0.7017`, top2 `0.7604`, high-error KL `1.2710`.
  - C hard-query VAE: KL `0.5268`, acc `0.6700`, top2 `0.8087`, high-error KL `1.1912`.
  - None beat old qexp4 VAE on the single-model global gate.
- Diagnostic ensemble probe:
  - old+A: KL `0.3786`, acc `0.7654`, top2 `0.8788`, high-error KL `1.0282`.
  - old+C: KL `0.3979`, acc `0.7675`, top2 `0.8800`, high-error KL `1.0254`.
  - Interpretation: ensemble is not the intended method; it only shows old global prior and new hard-state behavior are complementary.
  - User correctly objected that ensemble is a technique/diagnostic, not a clean method for the paper goal of training one estimator.
  - The ensemble pair20 1-ZSC run was stopped and should not be cited as a formal method result.
- Current methodological implication:
  - Do not optimize for history gap alone.
  - Do not present ensemble as the method.
  - Next estimator direction should be a single-model design/training objective that preserves old VAE global accuracy while improving hard-state reconstruction; do not use ensemble or distillation as the method narrative.

### 2026-07-06 PG-only strict QA-PPNP estimator attempt

- User clarified the next estimator line should remain PG-only strict:
  - train estimator only on PPO/MAPPO;
  - keep IQL/VDN/PQN-VDN only for strict 1-ZSC eval;
  - do not use ensemble or distillation as the method narrative.
- Implemented a single-model query-conditioned VAE / ANP-style path:
  - code changed in `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/latent_partner_decoder.py`
  - trainer flag: `--query_conditioned_vae`
  - behavior: query observation cross-attends to partner history tokens, then the query-conditioned representation produces the VAE latent instead of using a mean history summary.
  - Old checkpoints are backward compatible because the new path only activates when `latent_query_conditioned_vae` is present in the saved `.npz`.
- Smoke passed:
  - report: `reports/qa_ppnp_smoke_20260706`
- Formal PG-only strict train:
  - report: `reports/qa_ppnp_qcond_vae_pgonly_train_20260706`
  - config: query-attn VAE, `query_conditioned_vae=True`, `decoder_z_norm=True`, `query_expansion_k=4`, `query_expansion_scope=partner_policy`, `cross_context_fraction=0.25`, 3000 steps, batch 512.
  - internal val: true KL `0.1978`, TV `0.1941`, argmax acc `0.7462`, top2 `0.8617`; wrong/random/delayed gaps were positive.
- Unified heldout-PG leaderboard:
  - report: `reports/qa_ppnp_qcond_vae_heldout_leaderboard_20260706`
  - old qexp4 VAE anchor: KL `0.4484`, acc `0.7267`, top2 `0.8508`.
  - QA-PPNP q-conditioned VAE: KL `0.6321`, acc `0.5704`, top2 `0.6842`, wrong gap `+0.2409`, random gap `+0.0547`, delayed gap `+0.0165`.
  - It did not pass offline gate; no pair20 1-ZSC was run.
- Interpretation:
  - Query-conditioned posterior increases true-vs-corrupted separation but hurts heldout policy reconstruction accuracy.
  - This reinforces the current lesson: history sensitivity is evidence of using context, not evidence of accurate partner policy reconstruction.
  - Next single-model estimator should focus on heldout accuracy, likely via better PG-only data construction and objectives that preserve global query-policy accuracy while extracting only reliable partner-specific corrections.

### 2026-07-07 Identifiable local policy estimator

- Implemented the planned PG-only strict identifiable estimator branch on the A40 server:
  - grouped dataset builder: `build_identifiable_grouped_dataset.py`
  - identifiability diagnostic: `diagnose_identifiable_dataset.py`
  - residual-gated single-model estimator in `latent_partner_decoder.py`
  - trainer flags: `--residual_gated`, `--query_group_balanced_batch`, prior KL, same-query pairwise TV loss, low-disagreement residual penalty
  - evaluator low/high `query_disagreement_tv` slices
- Stage 0 reproduced the old heldout-PG anchor exactly:
  - old qexp4 VAE on `reports/accuracy_first_pg_heldout_dataset_20260706`: KL `0.4484`, TV `0.3130`, acc `0.7267`, top2 `0.8508`.
- Stage 1 grouped datasets:
  - train run0-7: `reports/identifiable_grouped_train_pgonly_20260707`, 160k rows, 16 policies, mean query TV `0.4618`, p50 `0.4738`, p90 `0.5721`, mean residual TV `0.3424`.
  - heldout run8-9: `reports/identifiable_grouped_heldout_pg_20260707`, 60k rows, 4 policies, mean query TV `0.5569`, p50 `0.5474`, p90 `0.7778`, mean residual TV `0.3629`.
- Identifiability diagnostic with PG prototype history likelihood:
  - train grouped: retrieval acc `0.9589`, true posterior mean `0.2609`.
  - heldout grouped: retrieval acc `0.9499`, true posterior mean `0.7614`.
  - Interpretation: the small history window is not information-free; the current bottleneck is not simply impossibility or no data signal.
- Original heldout-PG accuracy gate:
  - old qexp4 VAE: KL `0.4484`, acc `0.7267`, top2 `0.8508`.
  - residual-gated: KL `0.6062`, TV `0.4063`, acc `0.5054`, top2 `0.6762`.
  - no pairwise loss: KL `0.7223`, acc `0.4996`, top2 `0.6321`.
  - no low-disagreement penalty: KL `0.6273`, acc `0.5042`, top2 `0.6504`.
  - None passed the accuracy gate; no pair20 1-ZSC was run.
- Grouped heldout-PG:
  - old qexp4 VAE: KL `0.6753`, high-disagreement KL `0.9546`, low-disagreement KL `0.0894`.
  - residual-gated: KL `0.5391`, high-disagreement KL `0.7655`, low-disagreement KL `0.1437`.
  - no pairwise loss: KL `0.6115`, high-disagreement KL `0.9657`, low-disagreement KL `0.0940`.
  - no low-disagreement penalty: KL `0.5437`, high-disagreement KL `0.7929`, low-disagreement KL `0.1359`.
- Interpretation:
  - Grouped training improves the intended same-query/high-disagreement reconstruction objective.
  - It also sacrifices natural rollout/global heldout accuracy and ordinary low-disagreement states, which likely explains why it is not ready for reward eval.
  - Pairwise TV loss helps the grouped objective; low-disagreement penalty has only a small effect at the tested coefficient.
  - The next estimator design should preserve a stronger old-VAE-like global/query prior and add partner-specific residual only where identifiable, rather than replacing the prior with a grouped-specialized estimator.
- User-requested pair20 follow-up despite the offline gate failure:
  - same-run base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - residual-gated with `sensitivity_no_history` gate: all_q `85.800`, IQL `14.840`, VDN `54.110`, PQN-VDN `188.450`.
  - residual-gated ungated direct blend: all_q `85.593`, IQL `14.240`, VDN `52.940`, PQN-VDN `189.600`.
  - gated controls: wrong `85.217`, random `84.360`, delayed `85.553`.
  - Interpretation: residual estimator has local reward signal (`+4.007` over same-run base), but the simple sensitivity gate adds only `+0.207` over ungated and true-history control separation is weak (`+0.583` over wrong, `+1.440` over random, `+0.247` over delayed). It remains below old qexp4 VAE pair20 anchor `89.017`, so this is useful evidence, not a solved method.

### 2026-07-07 Prior-anchor local residual estimator

- User accepted narrowing the method claim to identifiable local partner-specific same-view policy correction.
- Implemented a single-checkpoint prior-anchor residual estimator on A40:
  - `latent_partner_decoder.py` now supports embedded frozen `prior_anchor__*` decoder params.
  - `train_latent_partner_decoder.py` adds `--prior_decoder_path`, matching-param initialization from old qexp4 VAE, frozen prior-anchor gradients, `--gate_target_coef`, and `--residual_tv_coef`.
  - Prior anchor: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
  - Formal train: `reports/prior_anchor_residual_pgonly_train_20260707`.
- Train setup:
  - grouped PG-only train dataset: `reports/identifiable_grouped_train_pgonly_20260707/identifiable_grouped_latent_decoder_dataset.npz`.
  - `steps=3000`, `lr=2e-4`, `pairwise_tv_coef=0.2`, `low_disagreement_residual_coef=0.2`, `residual_tv_coef=0.02`, `gate_target_coef=0.05`, `residual_clip=2.0`.
- Offline result:
  - original heldout global worsens vs old VAE: KL `0.5195` vs `0.4484`, acc `0.6308` vs `0.7267`.
  - original high-baseline-error improves: KL `1.1757` vs `1.4185`.
  - grouped heldout global improves: KL `0.5675` vs `0.6753`.
  - grouped high-disagreement improves: KL `0.7678` vs `0.9546`.
  - grouped low-disagreement worsens: KL `0.1477` vs `0.0894`.
  - Interpretation: local correction works where old VAE is wrong / partner disagreement is high, but it still damages ordinary states.
- Pair20 strict 1-ZSC with `sensitivity_no_history` gate:
  - base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - true history: all_q `88.257`, IQL `16.710`, VDN `56.420`, PQN-VDN `191.640`.
  - wrong history: all_q `88.467`, IQL `17.950`, VDN `56.480`, PQN-VDN `190.970`.
  - random history: all_q `86.740`, IQL `16.580`, VDN `53.360`, PQN-VDN `190.280`.
  - delayed history: all_q `87.923`, IQL `16.480`, VDN `56.500`, PQN-VDN `190.790`.
  - Interpretation: this is close to old qexp4 VAE pair20 `89.017` and better than base by `+6.463`, with improved IQL/VDN relative to old VAE, but PQN is lower and wrong-history is slightly higher than true. It supports the local-correction direction, but does not yet prove reliable partner-specific history use. No full eval for this version.

### 2026-07-07 Q-family estimator training diagnostic

- User proposed directly training the estimator on the 1-ZSC target Q-family methods (`IQL`, `VDN`, `PQN-VDN`) to test whether the current bottleneck is estimator training-data coverage rather than model capacity alone.
- This is intentionally contaminated and should not be presented as a clean strict 1-ZSC method.
- Dataset:
  - raw: `reports/qfamily_diagnostic_raw_dataset_20260707/pg_mixed_latent_decoder_dataset.npz`, 144k rows.
  - Q-target filtered: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, 96k rows.
  - partner-family rows: `PQN-VDN 38k`, `VDN 36k`, `IQL 22k`.
- Estimator:
  - checkpoint: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
  - query-attn qexp4 VAE, `vae_beta=0.001`, `decoder_z_norm=true`, `query_expansion_k=4`, `cross_context_fraction=0.25`, `steps=3000`, `batch_size=512`.
- Offline validation:
  - train step-3000 KL/CE `0.0927`, TV `0.0543`, argmax acc `0.9644`, top2 `0.9941`.
  - val true KL `0.3581`, TV `0.1021`, argmax acc `0.9160`, top2 `0.9698`.
  - history gaps remain weak: wrong `+0.0041`, delayed `+0.0009`, random `+0.0140`; embedding retrieval acc `0.6362`.
- Pair20 1-ZSC diagnostic:
  - base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - Q-trained true: all_q `89.120`, IQL `16.780`, VDN `56.510`, PQN-VDN `194.070`.
  - Q-trained wrong: all_q `88.573`, IQL `15.740`, VDN `56.330`, PQN-VDN `193.650`.
  - Q-trained random: all_q `89.287`, IQL `16.710`, VDN `56.780`, PQN-VDN `194.370`.
  - Q-trained delayed: all_q `88.777`, IQL `16.470`, VDN `56.170`, PQN-VDN `193.690`.
  - old clean qexp4 VAE anchor: all_q `89.017`, IQL `15.990`, VDN `55.190`, PQN-VDN `195.870`.
- Interpretation:
  - Direct Q-family training brings aggregate reward to roughly the old clean VAE level and improves IQL/VDN over old VAE.
  - However random history slightly beats true history (`89.287` vs `89.120`), so the improvement is mostly a learned Q-family same-view prior rather than reliable partner-specific history use.
  - Conclusion: data distribution is a real bottleneck, but not the whole bottleneck. Even with target-family data, short-history binding/local partner identification remains unresolved. Do not run full eval for this diagnostic.
- Follow-up fair offline comparison on the same Q-family val split:
  - report: `reports/qfamily_diagnostic_old_vs_qtrained_offline_20260707/accuracy_summary.md`.
  - dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val, 19,400 rows.
  - old clean qexp4 VAE: KL `1.5416`, TV `0.5824`, acc `0.5048`, top2 `0.6327`.
  - Q-trained qexp4 VAE: KL `0.3539`, TV `0.1021`, acc `0.9160`, top2 `0.9698`.
  - This makes the puzzle sharper: Q-trained VAE is much better at Q-family policy reconstruction, but pair20 reward barely improves over old clean VAE (`89.120` vs `89.017`).
  - Interpretation: old clean VAE reward likely comes from a reward-useful cooperative prior, not accurate Q-policy reconstruction. Q-trained VAE improves many offline states that are not reward-critical and still lacks reliable true-history conditioning.
- Query sensitivity diagnostic on the same Q-family val split:
  - old VAE is not a constant output prior. Replacing real query with shuffled/zero/gaussian query changes its prediction substantially:
    - old VAE TV vs real-query output: shuffled `0.5360`, zero `0.6926`, gaussian `0.5819`.
    - old VAE KL to target worsens from real `1.5416` to shuffled `2.5006`, zero `2.9865`, gaussian `2.7666`.
  - Q-trained VAE is also query-sensitive:
    - TV vs real-query output: shuffled `0.7489`, zero `0.8764`, gaussian `0.7438`.
  - Interpretation: old VAE should be described as a query-conditioned cooperative prior with weak/poor partner-history binding, not as an observation-independent fixed action distribution.
- Pair20 1-ZSC query-randomization diagnostic:
  - code path: `TTAC_LATENT_DECODER_QUERY_MODE=gaussian`, which randomizes only the latent decoder's `query_obs`; ego base policy still sees true obs and partner history remains true.
  - report: `reports/old_vae_gaussian_query_pair20_1zsc_20260707/mixed_1zsc_summary.md`.
  - base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - old qexp4 VAE real query: all_q `89.017`, IQL `15.990`, VDN `55.190`, PQN-VDN `195.870`.
  - old qexp4 VAE gaussian query: all_q `77.447`, IQL `15.370`, VDN `48.010`, PQN-VDN `168.960`.
  - Interpretation: randomizing the estimator query drops old VAE by `-11.570` vs real-query VAE and `-4.347` below base. Its 1-ZSC gain strongly depends on real current query observation, so it is not a constant prior; it is a query-conditioned cooperative prior with weak partner-history binding.
- Q-trained VAE direct-policy diagnostic:
  - added eval mode `ttac_v5_8_latent_decoder_policy` in `ttac_v5_8_fast_online/policy.py`.
  - behavior: when history exists, ego action distribution is replaced by the latent decoder's `target_probs`; no base-policy blend is used.
  - report: `reports/qtrained_vae_policy_replace_pair20_1zsc_20260707/mixed_1zsc_summary.md`.
  - base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - Q-trained VAE direct blend: all_q `89.120`, IQL `16.780`, VDN `56.510`, PQN-VDN `194.070`.
  - Q-trained VAE direct policy replacement: all_q `42.923`, IQL `10.000`, VDN `21.140`, PQN-VDN `97.630`.
  - Interpretation: directly executing the reconstructed same-view partner policy collapses reward. Accurate partner-policy reconstruction is not itself a good ego controller; the useful signal comes from a bounded correction target on top of a competent base policy.
- Old-vs-Q-trained VAE output diagnostic on Q-family val:
  - dataset: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val, 19,400 rows.
  - Q-family target probs are effectively one-hot because data collection used greedy Q action targets; target entropy `0.0`.
  - old VAE true history: KL `1.5416`, TV `0.5824`, acc `0.5048`, entropy `1.0754`, mean max prob `0.6210`.
  - old VAE no history: KL `1.4154`, TV `0.5948`, acc `0.5158`, entropy `1.1453`, mean max prob `0.6024`.
  - Q-trained VAE true history: KL `0.3539`, TV `0.1021`, acc `0.9160`, entropy `0.1275`, mean max prob `0.9530`.
  - Q-trained VAE no history: KL `0.4379`, TV `0.1422`, acc `0.8811`, entropy `0.1900`, mean max prob `0.9341`.
  - old true vs Q-trained true: mean output TV `0.5533`, top-action same `0.5080`.
  - history effect is weak for both: old true-vs-no-history TV `0.1299`, top-action same `0.9246`; Q-trained true-vs-no-history TV `0.0763`, top-action same `0.9376`.
  - by family, Q-trained is much closer to Q targets: IQL KL `0.2382` vs old `2.2420`, VDN KL `0.4130` vs old `1.7726`, PQN-VDN KL `0.3648` vs old `0.9801`.
  - Interpretation: old VAE and Q-trained VAE have fundamentally different query-conditioned outputs. Q-trained learns a sharp Q-greedy predictor; old VAE learns a softer cooperative prior. But neither solves strong partner-specific history-conditioned modeling.

### 2026-07-07 Q-family same-query counterfactual VAE

- User asked to test method A: keep the Q-family diagnostic setup (`IQL`, `VDN`, `PQN-VDN`) but reconstruct the training data so the same `query_obs` is paired with multiple concrete Q partner histories and targets.
- Added builder: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/build_qfamily_grouped_dataset.py`.
- Dataset:
  - source: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`.
  - output: `reports/qfamily_grouped_qcounter_dataset_20260707/qfamily_grouped_latent_decoder_dataset.npz`.
  - rows `159,992`, query groups `11,428`, policies per query `14`.
  - loaded policies: 14 concrete Q policies from IQL/VDN/PQN-VDN; `iql:run_4` skipped due no support rows.
  - mean query disagreement TV `0.6936`, p50 `0.7033`, p90 `0.8022`, mean residual target TV `0.6441`, target entropy `0.0`.
- Training:
  - checkpoint: `reports/qfamily_grouped_qcounter_vae_k1_train_20260707/latent_partner_decoder.npz`.
  - structure: query-attn VAE, `vae_beta=0.001`, `decoder_z_norm=true`.
  - key difference from ordinary Q-trained VAE: `query_expansion_k=1`, `query_group_balanced_batch=true`, `query_groups_per_batch=64`, `rows_per_query_group=14`.
  - rationale: keep same-query counterfactual pressure clean; qexp4 would dilute it by sampling other queries.
- Internal grouped validation:
  - true KL `0.3757`, TV `0.1588`, acc `0.8726`, top2 `0.9544`.
  - wrong gap `+0.0592`, random gap `+0.0213`, delayed gap `+0.0008`.
  - same-query output TV `0.6715` vs target TV `0.7011`, corr `0.7964`, ratio `0.9579`.
  - embedding retrieval acc `0.8991`.
- Same Q-target val comparison:
  - report: `reports/qfamily_qtrained_vs_qcounter_offline_20260707/accuracy_summary.md`.
  - ordinary Q-trained qexp4 VAE: KL `0.3539`, TV `0.1021`, acc `0.9160`, top2 `0.9698`, wrong gap `+0.0039`, no-history gap `+0.0840`.
  - Q same-query counterfactual VAE: KL `0.3561`, TV `0.1621`, acc `0.8771`, top2 `0.9552`, wrong gap `+0.0713`, no-history gap `+1.4431`.
  - Interpretation: same-query training greatly increases history dependence without much worsening KL, but lowers argmax accuracy.
- Pair20 1-ZSC:
  - report: `reports/qfamily_grouped_qcounter_vae_k1_pair20_1zsc_20260707/mixed_1zsc_summary.md`.
  - base: all_q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`.
  - true: all_q `84.700`, IQL `13.720`, VDN `52.680`, PQN-VDN `187.700`.
  - wrong: all_q `84.997`, IQL `13.910`, VDN `53.280`, PQN-VDN `187.800`.
  - random: all_q `84.360`, IQL `13.270`, VDN `52.310`, PQN-VDN `187.500`.
  - delayed: all_q `85.037`, IQL `13.760`, VDN `52.740`, PQN-VDN `188.610`.
  - ordinary Q-trained anchor: all_q `89.120`; old clean qexp4 VAE anchor: all_q `89.017`.
- Interpretation:
  - The data construction did what it was supposed to do offline: stronger same-query policy variation and stronger history/no-history separation.
  - It still does not solve reward: true-history is not higher than wrong/delayed, and aggregate is well below ordinary Q-trained/old VAE.
  - Same-query counterfactual alone is not enough under fixed direct blend. It likely needs a natural-query/qexp mixture, gate, or online latent fitting before it can become a reward-useful estimator.

### 2026-07-07 Estimator policy-geometry diagnostic

- User asked whether the three trained estimators shrink or enlarge the policy-distribution gap to the 1-ZSC partner models when used as estimators.
- Added script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_estimator_policy_geometry.py`.
- Report: `reports/estimator_policy_geometry_ppoego_qval_20260707/policy_geometry_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val.
- Filter: only `ppo:run_0` and `ppo:run_1` ego rows, 4,700 rows, matching pair20 1-ZSC ego policies.
- Metric: compare true Q partner policy `pi_partner(.|ego_obs)` with base PPO and with direct-blended PPO (`alpha=0.5`, `clip=2.0`). Negative delta means the blend moves ego closer to the partner.
- All-Q geometry:
  - old clean VAE: base TV `0.5238`, estimator TV `0.5860`, blend TV `0.5246`, delta TV `+0.0008`, shrink rows `55.7%`; base KL `1.8360`, estimator KL `1.3882`, blend KL `1.5359`, delta KL `-0.3001`.
  - Q-trained VAE: base TV `0.5238`, estimator TV `0.1202`, blend TV `0.4135`, delta TV `-0.1102`, shrink rows `92.7%`; base KL `1.8360`, estimator KL `0.4432`, blend KL `1.0994`, delta KL `-0.7365`.
  - Q same-query counterfactual VAE: base TV `0.5238`, estimator TV `0.1963`, blend TV `0.4396`, delta TV `-0.0841`, shrink rows `82.2%`; base KL `1.8360`, estimator KL `0.4610`, blend KL `1.1560`, delta KL `-0.6800`.
- Family delta TV / shrink rows:
  - old clean VAE: IQL `+0.0958` / `10.0%`, VDN `-0.0308` / `79.2%`, PQN-VDN `-0.0138` / `47.5%`.
  - Q-trained VAE: IQL `-0.0449` / `97.5%`, VDN `-0.1382` / `93.0%`, PQN-VDN `-0.1088` / `88.5%`.
  - Q same-query counterfactual VAE: IQL `-0.0397` / `89.9%`, VDN `-0.1145` / `86.9%`, PQN-VDN `-0.0622` / `67.5%`.
- Estimator argmax acc against true Q partner policy, before blend:
  - old clean VAE: all-Q `0.5877`, IQL `0.9400`, VDN `0.3892`, PQN-VDN `0.6831`.
  - Q-trained VAE: all-Q `0.8957`, IQL `0.9640`, VDN `0.8971`, PQN-VDN `0.8408`.
  - Q same-query counterfactual VAE: all-Q `0.8481`, IQL `0.9410`, VDN `0.8667`, PQN-VDN `0.7423`.
- Interpretation:
  - Q-trained VAE clearly shrinks the policy gap to true Q partners; qcounter also shrinks it, but less strongly.
  - Old clean VAE does not consistently shrink TV overall and moves away on IQL, although it reduces KL by assigning more mass to target actions on high-KL rows.
  - Therefore reward behavior cannot be explained by "moving closer to partner distribution" alone. Partner-policy closeness is achievable and measurable, but fixed direct blend reward depends on which states/families are moved and whether the correction remains compatible with the base ego policy.

### 2026-07-07 Q-trained VAE noise-history distance diagnostic

- User asked whether Q-trained VAE with correct history is clearly closer to true partner policy than with random-noise history.
- Added script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_noise_history_distance.py`.
- Report: `reports/qtrained_vae_noise_history_distance_20260707/noise_history_distance_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val.
- Decoder: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
- On all Q-family val rows:
  - true history: KL `0.3539`, TV `0.1021`, acc `0.9160`, entropy `0.1275`.
  - shuffled real row history: KL `0.8480`, TV `0.1870`, acc `0.8281`.
  - Gaussian std1 obs + uniform actions: KL `0.8057`, TV `0.1744`, acc `0.8387`.
  - Gaussian matched obs + empirical actions: KL `1.1457`, TV `0.1983`, acc `0.8120`.
  - zero obs + uniform actions: KL `0.4518`, TV `0.1418`, acc `0.8747`.
- On pair20-relevant PPO ego rows (`ppo:run_0/run_1`, 4,700 rows):
  - true history: KL `0.4432`, TV `0.1202`, acc `0.8957`.
  - shuffled real row history: KL `0.5788`, TV `0.1553`, acc `0.8634`.
  - Gaussian std1 obs + uniform actions: KL `0.5685`, TV `0.1503`, acc `0.8691`.
  - Gaussian matched obs + empirical actions: KL `0.5863`, TV `0.1507`, acc `0.8677`.
  - zero obs + uniform actions: KL `0.5347`, TV `0.1465`, acc `0.8728`.
- Interpretation:
  - Correct history is consistently closer than noise history, so history is not ignored.
  - The gap is moderate rather than decisive on pair20-relevant rows: true TV `0.1202` vs noise TV about `0.1465`-`0.1553`, and true acc `0.8957` vs noise acc about `0.8634`-`0.8728`.
  - This supports the current diagnosis: Q-trained VAE uses history, but a large part of its accuracy still comes from query-conditioned Q-family prior rather than precise per-partner identification.

### 2026-07-07 Q-trained VAE direct on-policy accuracy diagnostic

- User asked why Q-trained VAE can look close to the real partner policy offline but collapses when directly used as a controller.
- Added script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_direct_onpolicy_accuracy.py`.
- Report: `reports/qtrained_direct_onpolicy_accuracy_pair20s20_20260707/direct_onpolicy_accuracy_summary.md`.
- Setup:
  - ego mode: `ttac_v5_8_latent_decoder_policy`, direct estimator policy replacement, no blend/base after warmup.
  - estimator: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
  - pair20-style: `max_ego_policies=2`, `max_partner_policies=5`, partners `IQL/VDN/PQN-VDN`, both roles, `20` eval seeds.
  - metrics compare estimator output `pi_hat(.|ego_obs, partner_history)` to true Q partner policy on the same `ego_obs`; no blend/base is used in the metric.
- Offline reference on PPO-ego Q val rows from policy geometry: Q-trained VAE `KL=0.4432`, `TV=0.1202`, `argmax acc=0.8957`.
- Direct on-policy states:
  - all-Q all steps: reward `41.500`, KL `3.2600`, TV `0.5184`, acc `0.4909`, top2 `0.6600`, entropy `0.2624`, maxprob `0.9042`.
  - all-Q after history warmup (`t>=50`): reward `41.500`, KL `3.3645`, TV `0.5275`, acc `0.4813`, top2 `0.6506`, entropy `0.2527`, maxprob `0.9079`.
  - IQL `t>=50`: reward `10.300`, KL `3.8653`, TV `0.6024`, acc `0.4159`.
  - VDN `t>=50`: reward `20.750`, KL `3.9734`, TV `0.6124`, acc `0.3874`.
  - PQN-VDN `t>=50`: reward `93.450`, KL `2.2548`, TV `0.3677`, acc `0.6407`.
- Interpretation:
  - Direct estimator rollout causes severe closed-loop distribution shift. The estimator is accurate on the offline/base-like PPO ego state distribution, but not on the states it creates when it controls the ego by itself.
  - The collapse persists after history warmup, so it is not just the first 50 steps lacking history.
  - The estimator remains confident on direct on-policy states (`maxprob ~=0.91`, entropy `~0.25`), so errors are high-confidence and compound through the trajectory.
  - This explains why direct `pi_hat` cooperation is much worse than blend: blend preserves base closed-loop stability; direct replacement leaves the estimator operating far outside the distribution where its offline accuracy was measured.

### 2026-07-07 New non-ZSC partner 1-ZSC gap test

- Goal: test whether adding the newly trained effective non-ZSC partner methods changes the old-vs-new VAE 1-ZSC improvement gap.
- Evaluator patch:
  - `evaluate_mixed_1zsc.py` now supports Q partners `pqn_soft`, `pqn_wqmix` and MAPPO-compatible AC partner labels `maac`, `happo`.
  - This only adds loading/summarization support; evaluation logic is unchanged.
- Formal eval:
  - eval100, 10 ego policies x 10 partner policies x both roles.
  - base report: `reports/new_non_zsc_partner_gap_eval100_20260707_163456_base`.
  - old clean qexp4 VAE report: `reports/new_non_zsc_partner_gap_eval100_20260707_163456_oldvae`.
  - Q-trained qexp4 VAE report: `reports/new_non_zsc_partner_gap_eval100_20260707_163456_newvae`.
- New four-method subset (`pqn_soft`, `pqn_wqmix`, `maac`, `happo`):
  - all partners: base `70.491`, old VAE `80.383` (`+9.891`), new VAE `76.948` (`+6.456`), new-old gap `-3.435`.
  - all Q partners: base `87.109`, old `100.660` (`+13.551`), new `92.982` (`+5.873`), new-old `-7.678`.
  - all AC/PG partners: base `53.874`, old `60.106` (`+6.232`), new `60.914` (`+7.040`), new-old `+0.808`.
  - by partner:
    - `pqn_soft`: base `92.043`, old `110.477`, new `101.368`, new-old `-9.109`.
    - `pqn_wqmix`: base `82.174`, old `90.842`, new `84.595`, new-old `-6.247`.
    - `maac`: base `47.157`, old `51.899`, new `52.252`, new-old `+0.353`.
    - `happo`: base `60.591`, old `68.312`, new `69.575`, new-old `+1.263`.
- Method-equal comparison:
  - old original 3-method pair20 (`iql`, `vdn`, `pqn_vdn`): base `81.793`, old `89.017` (`+7.223`), new `89.120` (`+7.327`), new-old gap `+0.103`.
  - new 4-method subset: base `70.491`, old `80.383` (`+9.891`), new `76.948` (`+6.456`), new-old gap `-3.435`.
  - old3 + new4 method-equal aggregate: base `75.335`, old `84.083` (`+8.748`), new `82.164` (`+6.829`), new-old gap `-1.919`.
- Interpretation:
  - Adding the new non-ZSC partners makes the old-vs-new VAE improvement gap smaller and flips it negative under a method-equal expanded benchmark.
  - The flip is driven by the new Q-style partners (`pqn_soft`, `pqn_wqmix`), where old clean VAE remains much stronger.
  - New Q-trained VAE is slightly better on the AC partners (`maac`, `happo`), but that gain is too small to offset the Q-partner loss.

### 2026-07-07 Candidate partner top5 by lift over base

- Clarification: when the metric is improvement over base (`method - base`), top5 selection should optimize lift, not old-vs-new difference.
- Candidate pool used here: `iql`, `vdn`, `pqn_vdn`, `iql_dueling`, `vdn_dueling`, `iql_double`, `pqn_soft`, `pqn_wqmix`, `maac`, `happo`.
  - Failed zero/non-credible partners such as the failed COMA/QPLEX/WQMIX reproductions are excluded.
- Additional eval completed for Q-trained new VAE on the three partner variants:
  - report: `reports/partner_variants_10seed_qtrained_vae_1zsc_100seed_20260707`.
  - `iql_dueling`: base `49.687`, new VAE `46.562`, lift `-3.125`.
  - `vdn_dueling`: base `69.465`, new VAE `74.957`, lift `+5.492`.
  - `iql_double`: base `72.505`, new VAE `74.699`, lift `+2.194`.
- Full per-method lift table:
  - `iql`: base `15.820`, old `15.990` (`+0.170`), new `16.780` (`+0.960`).
  - `vdn`: base `51.690`, old `55.190` (`+3.500`), new `56.510` (`+4.820`).
  - `pqn_vdn`: base `177.870`, old `195.870` (`+18.000`), new `194.070` (`+16.200`).
  - `iql_dueling`: base `49.687`, old `53.325` (`+3.638`), new `46.562` (`-3.125`).
  - `vdn_dueling`: base `69.465`, old `78.465` (`+9.000`), new `74.957` (`+5.492`).
  - `iql_double`: base `72.505`, old `81.405` (`+8.900`), new `74.699` (`+2.194`).
  - `pqn_soft`: base `92.043`, old `110.477` (`+18.434`), new `101.368` (`+9.325`).
  - `pqn_wqmix`: base `82.174`, old `90.842` (`+8.668`), new `84.595` (`+2.421`).
  - `maac`: base `47.157`, old `51.899` (`+4.742`), new `52.252` (`+5.095`).
  - `happo`: base `60.591`, old `68.312` (`+7.721`), new `69.575` (`+8.984`).
- Old clean VAE top5 by lift:
  - partners: `pqn_soft`, `pqn_vdn`, `vdn_dueling`, `iql_double`, `pqn_wqmix`.
  - average base `98.811`, average old VAE `111.412`, average lift `+12.600`.
- Q-trained new VAE top5 by lift:
  - partners: `pqn_vdn`, `pqn_soft`, `happo`, `vdn_dueling`, `maac`.
  - average base `89.425`, average new VAE `98.444`, average lift `+9.019`.
- Interpretation:
  - Under top5 partner selection, old clean VAE has the larger advantage over base (`+12.600`) than Q-trained new VAE (`+9.019`).
  - Adding/selecting from the expanded candidate pool increases the maximum achievable lift for both relative to the original old3 pair20 lift, but old benefits more because `pqn_soft`, `pqn_vdn`, `vdn_dueling`, `iql_double`, and `pqn_wqmix` are especially favorable to it.

### 2026-07-07 PQN-QPLEX native migration

- User asked whether COMA/QPLEX/WQMIX had been migrated to the stable bottom.
  - WQMIX already has a successful stable-bottom migration: `pqn_wqmix`.
  - COMA had PPO/MAPPO-bottom migrations (`ppo_coma`/`coma_ppo`) but remained SP/XP `0/0`.
  - QPLEX did not yet have a successful PQN-bottom migration, so we added one.
- Implementation:
  - new module: `JaxMARL/baselines/QLearning/pqn_qplex_cnn_overcooked.py`.
  - method/subdir/prefix: `pqn_qplex`, `pqn_qplex`, `pqn_qplex_cnn`.
  - training bottom: stable PQN rollout + q-lambda target path, not the old replay-buffer QPLEX path.
  - QPLEX-style training head: duplex-dueling mixer `Q_tot(s,u)=V_tot(s)+sum_i lambda_i(s) A_i(o_i,u_i)`.
  - Deployment/eval contract: only the per-agent Q network is saved, so downstream SP/XP and 1-ZSC evaluators execute greedy per-agent Q like other `pqn_*` methods.
  - Added registry support in `train_classic_partner_ov2.py`, `evaluate_qlearning_object_sp_xp.py`, and `evaluate_mixed_1zsc.py`.
- Sanity checks:
  - import/py_compile passed on A6000.
  - tiny smoke train and evaluator smoke passed:
    - run: `runs/pqn_qplex_smoke_20260707`.
    - report: `reports/pqn_qplex_smoke_spxp_20260707`.
- First pure-QPLEX 1M single-seed sanity:
  - run: `runs/qlearning_native_migrations_pqn_qplex_sp1m_20260707_212230`.
  - report: `reports/qlearning_native_migrations_pqn_qplex_spxp_pqn_qplex_sp1m_20260707_212230`.
  - SP50 `0.000`.
  - Interpretation: pure mixer loss can fit through the mixer while saved per-agent Q remains unusable.
- Stabilized version:
  - added VDN auxiliary TD loss with `QPLEX_VDN_AUX_COEF=1.0` so saved per-agent Q receives direct TD supervision.
  - 1M single-seed sanity passed:
    - run: `runs/qlearning_native_migrations_pqn_qplex_aux_sp1m_20260707_212629`.
    - report: `reports/qlearning_native_migrations_pqn_qplex_spxp_pqn_qplex_aux_sp1m_20260707_212629`.
    - SP50 `200.000`.
- Formal run completed:
  - run id: `pqn_qplex_aux_10seed_20260707_212910`.
  - PID at launch: `87368`.
  - run root: `runs/qlearning_native_migrations_pqn_qplex_aux_10seed_20260707_212910`.
  - logs: `logs/qlearning_native_migrations_pqn_qplex_aux_10seed_20260707_212910`.
  - report: `reports/qlearning_native_migrations_pqn_qplex_spxp_pqn_qplex_aux_10seed_20260707_212910`.
  - settings: `METHODS=pqn_qplex`, `NUM_SEEDS=10`, `TOTAL_TIMESTEPS=10000000`, `RUN_SPXP=1`, `SPXP_EVAL_SEEDS=500`.
  - completed all 10 splits and saved 10 checkpoints.
  - SP/XP500 summary:
    - SP: `206.000` (`num_pairs=10`, `num_episodes=5000`).
    - XP: `159.556` (`num_pairs=90`, `num_episodes=45000`).
  - Interpretation: the VDN-auxiliary PQN-QPLEX migration is valid/nonzero and should be included in the next non-ZSC partner candidate pool.

### 2026-07-08 PQN-QPLEX added to VAE-vs-base 1-ZSC/top5 selection

- User asked to rerun the new/old VAE vs base 1-ZSC gap and top5 partner selection after `pqn_qplex` finished.
- Evaluator maintenance:
  - Added `pqn_qplex` support to `evaluate_mixed_1zsc.py` using the same PQN-style QNetwork loader path as `pqn_vdn`/`pqn_soft`/`pqn_wqmix`.
  - Also synchronized the local remote mirror copy.
  - Loader smoke passed: `reports/pqn_qplex_mixed_1zsc_loader_smoke_20260708`.
- Formal qplex-only eval100:
  - base report: `reports/pqn_qplex_gap_eval100_20260708_220522_base`.
  - old clean qexp4 VAE report: `reports/pqn_qplex_gap_eval100_20260708_220522_oldvae`.
  - Q-trained qexp4 VAE report: `reports/pqn_qplex_gap_eval100_20260708_220522_newvae`.
  - setting: `100` eval seeds, `10` ego policies x `10` partner policies x both roles, greedy qplex partner execution.
- `pqn_qplex` 1-ZSC:
  - base `79.967`.
  - old clean qexp4 VAE `94.008`, lift `+14.041`.
  - Q-trained qexp4 VAE `87.682`, lift `+7.715`.
  - new-old gap on qplex: `-6.326`.
- Expanded candidate pool after adding `pqn_qplex`:
  - `iql`, `vdn`, `pqn_vdn`, `iql_dueling`, `vdn_dueling`, `iql_double`, `pqn_soft`, `pqn_wqmix`, `maac`, `happo`, `pqn_qplex`.
  - This recomputation inherits the previous candidate table's historical screening/eval100 values and newly adds the qplex 10x10 eval100 rows; it is not yet a fresh uniform 11-family full rerun from scratch.
- Method-equal all-11 aggregate:
  - base `72.634`.
  - old clean qexp4 VAE `81.435`, lift `+8.801`.
  - Q-trained qexp4 VAE `78.095`, lift `+5.462`.
  - new-old gap `-3.339`.
- Q-only 9-method aggregate:
  - base `76.802`.
  - old clean qexp4 VAE `86.175`, lift `+9.372`.
  - Q-trained qexp4 VAE `81.914`, lift `+5.111`.
  - new-old gap `-4.261`.
- PG/AC-only aggregate remains the previous two-method result because qplex is Q-family:
  - base `53.874`, old `60.105` (`+6.232`), new `60.913` (`+7.040`), new-old `+0.808`.
- Old clean VAE top5 by lift over base after adding qplex:
  - partners: `pqn_soft`, `pqn_vdn`, `pqn_qplex`, `vdn_dueling`, `iql_double`.
  - average base `98.370`, average old VAE `112.045`, average lift `+13.675`.
  - `pqn_qplex` enters old top5 and replaces `pqn_wqmix`.
- Q-trained new VAE top5 by lift over base after adding qplex:
  - partners: `pqn_vdn`, `pqn_soft`, `happo`, `pqn_qplex`, `vdn_dueling`.
  - average base `95.987`, average new VAE `105.530`, average lift `+9.543`.
  - `pqn_qplex` enters new top5 and replaces `maac`.
- Interpretation:
  - Adding `pqn_qplex` increases both methods' best top5 lift, but increases old clean VAE more.
  - The top5 lift gap old-vs-new expands from `+3.581` before qplex (`12.600 - 9.019`) to `+4.132` after qplex (`13.675 - 9.543`).
  - This strengthens the current pattern: Q-style partners favor the old clean VAE more, while the Q-trained new VAE only has a small edge on AC/PG partners.

## 2026-07-08 Factorized prior-residual Q-family estimator diagnostic

- User wanted to first solve the `query_obs` shortcut in the Q-family VAE before considering online estimator update.
- Implemented a factorized prior-residual decoder:
  - `query_obs -> query/mean policy prior`.
  - `partner history -> z_history -> partner-specific residual`.
  - final logits are `prior_logits + residual_logits`.
  - no-history is structurally forced to equal the query prior by subtracting the zero-history latent baseline.
- Code touched:
  - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/latent_partner_decoder.py`
  - `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/train_latent_partner_decoder.py`
  - added `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_prior_residual_decoder.py`
- Training:
  - dataset: `reports/qfamily_grouped_qcounter_dataset_20260707/qfamily_grouped_latent_decoder_dataset.npz`
  - checkpoint: `reports/qfamily_factorized_prior_residual_vae_train_20260708/latent_partner_decoder.npz`
  - settings: `query_attn`, VAE, `z_dim=128`, `vae_beta=1e-4`, grouped batch `64 x 14`, `steps=3000`, `prior_loss_coef=0.75`, `pairwise_tv_coef=0.2`, `residual_logit_coef=0.2`
- Grouped-val summary:
  - true KL `0.4172`, TV `0.2305`, acc `0.8635`, top2 `0.9464`
  - wrong gap `+0.1414`, random gap `+0.0846`, delayed gap `+0.0003`
  - same-query output/target TV corr `0.7507`
  - embedding retrieval acc `0.8948`
- Prior-residual diagnostic:
  - report: `reports/qfamily_factorized_prior_residual_vae_diag_20260708/prior_residual_summary.md`
  - no-history final exactly equals query prior: `no_history_final_vs_prior_tv = 0.0`
  - query prior vs mean teacher KL `0.0570`
  - query prior vs target KL `1.2850`
  - true history final KL `0.4172`, TV `0.2305`, acc `0.8635`
  - true residual TV to prior `0.4861`
- Same Q-target val leaderboard:
  - Q-trained VAE: KL `0.3539`, TV `0.1021`, acc `0.9160`, top2 `0.9698`, wrong gap `+0.0039`
  - Q-counterfactual VAE: KL `0.3561`, TV `0.1621`, acc `0.8771`, top2 `0.9552`, wrong gap `+0.0713`
  - factorized prior-residual: KL `0.3990`, TV `0.2187`, acc `0.8738`, top2 `0.9435`, wrong gap `+0.1071`
- Pair20 strict 1-ZSC diagnostic:
  - report true: `reports/qfamily_factorized_prior_residual_pair20_1zsc_20260708`
  - report no-history: `reports/qfamily_factorized_prior_residual_nohist_pair20_1zsc_20260708`
  - same-run base: all-Q `81.793`, IQL `15.820`, VDN `51.690`, PQN-VDN `177.870`
  - factorized true history: all-Q `83.740`, IQL `13.700`, VDN `53.190`, PQN-VDN `184.330`
  - factorized no-history: all-Q `81.990`, IQL `14.040`, VDN `54.850`, PQN-VDN `177.080`
- Interpretation:
  - The architecture successfully removes the no-history/query-only shortcut and forces partner-specific differences into the history residual.
  - True history beats no-history by `+1.750` aggregate, mostly via PQN-VDN, so residual is not pure noise.
  - It is still much weaker than old clean qexp4 VAE pair20 `89.017` and Q-trained VAE pair20 `89.120`; no full eval for this version.
  - Next estimator direction should improve residual reconstruction accuracy and closed-loop stability, not merely increase history sensitivity.

## 2026-07-08 Simulated within-episode online latent-z adaptation

- User asked whether we can simulate online learning updates. Implemented a clean within-episode online latent adaptation mode:
  - mode: `ttac_v5_8_latent_decoder_blend_online_z`
  - suffix controls: wrong/random/delayed history work through existing `_resolve_history_action`.
  - per pair/role/episode state resets independently.
  - online update uses only current-episode observed `partner_obs, partner_action`.
  - no partner checkpoint logits, no oracle `pi_partner(. | ego_obs)`, no cross-episode or cross-partner data.
- Code changes:
  - `policy.py`: added online-z mode, warmup/ramp alpha gating, per-episode `z_p` SGD update in `update_after_step`.
  - `latent_partner_decoder.py`: added `encode_partner_online_z(...)` and `apply_latent_partner_decoder_with_z(...)`.
  - `evaluate_mixed_1zsc.py`: added online-z CLI args and auto-infers `TTAC_V5_8_LATENT_DIM` from decoder checkpoint.
- Online objective:
  - optimize only current episode `z_p`.
  - loss: `mean_t[-log pi_decoder(a_partner_t | o_partner_t, z)] + prior_coef * ||z - encoder(current_history)||^2`.
  - frozen: PPO ego, decoder, query prior/pi0, residual basis.
  - control still uses fixed direct blend.
- Smoke:
  - report: `reports/online_z_smoke_20260708`
  - `iql`, `1 ego x 1 partner x 2 roles x 1 seed`, passed.
- Seed10 pair20 diagnostic using Q-trained VAE checkpoint `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`:
  - same-run base: all-Q `81.033`, IQL `14.300`, VDN `51.000`, PQN-VDN `177.800`
  - ordinary Q-trained VAE blend: all-Q `86.967`, IQL `16.700`, VDN `52.900`, PQN-VDN `191.300`
  - online-z aggressive (`warmup=50`, `ramp=30`, `interval=10`, `steps=10`, `lr=0.05`, `prior=0.05`): all-Q `85.967`, IQL `14.800`, VDN `58.000`, PQN-VDN `185.100`
  - online-z conservative (`warmup=50`, `ramp=50`, `interval=10`, `steps=5`, `lr=0.01`, `prior=0.5`): all-Q `85.700`, IQL `15.700`, VDN `56.900`, PQN-VDN `184.500`
  - conservative wrong-history: all-Q `85.900`, IQL `15.700`, VDN `55.300`, PQN-VDN `186.700`
- Interpretation:
  - Online-z improves over base in seed10 and boosts VDN, but does not beat ordinary Q-trained VAE blend.
  - Wrong-history conservative slightly beats true-history, so this version does not demonstrate reliable reward-useful partner-specific online adaptation.
  - Do not run pair20 100-seed or full eval for this version.
  - If continuing online learning, pure partner-action CE on `z` is probably misaligned; it can move latent toward partner-view behavior imitation without preserving reward-useful same-view correction.

## 2026-07-08 Meta-trained online-z diagnostic

- User asked for a smarter online design than naive CE on z. Implemented meta-online-z training:
  - support: current partner `partner_obs/action` samples only.
  - inner update: CE on observed partner actions updates only `z`.
  - query/outer: same-view `query_obs` supervised by `target_partner_probs`.
  - outer loss trains the decoder so CE-updated `z` is useful for same-view policy reconstruction.
- Code:
  - added `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/train_meta_online_z_decoder.py`.
  - `evaluate_mixed_1zsc.py` adds `--ttac_latent_decoder_online_init_z_mode {encoder,zero,no_history}`.
  - `policy.py` uses that init mode for online `z_p`.
- Encoder-init probes:
  - `reports/meta_online_z_smoke_20260708`: first-order, lr `0.05`; `z_delta` only `~0.0008`, showing earlier online-z barely moved.
  - `reports/meta_online_z_second_order_probe_20260708`: second-order, inner lr `5.0`; `z_delta ~0.12`, but val post KL remained about equal to pre KL. Encoder-init lets the model continue relying on the offline encoder/prior.
- Zero-init meta training:
  - checkpoint/report: `reports/meta_online_z_zero_init_1k_20260708`.
  - init checkpoint: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
  - train data: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`.
  - settings: second-order, `steps=1000`, `batch_size=128`, `inner_steps=5`, `inner_lr=5.0`, `inner_prior_coef=0.0`, `init_z_mode=zero`.
- Offline Q-target val:
  - true: pre KL `0.1730`, post KL `0.1664`, TV `0.0828`, acc `0.9386`.
  - wrong: post KL `0.1681`.
  - random: post KL `0.1776`.
  - no-history: post KL `0.1744`.
  - Interpretation: CE-updated zero-init `z` finally improves same-view KL; true history is best, though true-vs-wrong KL gap is small (`~0.0016`).
- Seed10 pair20 1-ZSC:
  - report: `reports/meta_online_z_zero_init_pair20_seed10_20260708`.
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - meta checkpoint ordinary blend: all-Q `80.567`, IQL `17.200`, VDN `44.500`, PQN-VDN `180.000`.
  - meta-online-z true: all-Q `84.300`, IQL `18.800`, VDN `56.900`, PQN-VDN `177.200`.
  - meta-online-z wrong: all-Q `77.167`, IQL `16.700`, VDN `48.600`, PQN-VDN `166.200`.
- Interpretation:
  - This is the first online-z result with strong reward-side true/wrong separation (`+7.133` aggregate).
  - It improves over same-run base by `+3.300`, mostly through IQL and VDN.
  - It remains below the original Q-trained VAE ordinary seed10 control `86.967` and formal Q-trained VAE pair20 anchor `89.120`.
  - Zero-init meta training reshapes the decoder for online inference and hurts ordinary encoder-based blend; no 100-seed/full eval for this version.
  - Mechanism takeaway: meta-training the CE update can make online partner behavior matter, but the current controller is not yet competitive with the best offline VAE blend.

## 2026-07-08 Clean PG-only meta-online-z

- User asked for a clean version after discussing whether `pi0` introduces prior information.
- Implemented clean split support:
  - added `--eval_dataset` to `train_meta_online_z_decoder.py`.
  - train data: PPO/MAPPO only, run0..7: `reports/lpr_pgonly_train_dataset_light_20260706/pg_mixed_latent_decoder_dataset.npz`.
  - heldout PG eval: PPO/MAPPO run8..9: `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`.
  - init checkpoint: clean old VAE `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
  - output checkpoint: `reports/meta_online_z_clean_pgonly_zero_init_1k_20260708/latent_partner_decoder.npz`.
- Training settings:
  - second-order meta-online-z, `steps=1000`, `batch_size=128`, `inner_steps=5`, `inner_lr=5.0`, `inner_prior_coef=0.0`, `init_z_mode=zero`, `pre_update_kl_coef=0.0`.
- Heldout PG offline sanity:
  - true: pre KL `0.4318`, post KL `0.4268`, TV `0.3411`, acc `0.6516`, top2 `0.8419`.
  - wrong: post KL `0.4301`, acc `0.5905`.
  - random: post KL `0.4302`, acc `0.6516`.
  - no-history: post KL `0.4335`, acc `0.5825`.
  - Interpretation: clean PG-only training creates only a small history effect; true beats wrong by about `0.0033` KL and no-history by about `0.0067` KL.
- Pair20 strict 1-ZSC seed10 report: `reports/meta_online_z_clean_pgonly_pair20_seed10_20260708`.
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - clean meta checkpoint ordinary blend: all-Q `85.533`, IQL `15.600`, VDN `57.400`, PQN-VDN `183.600`.
  - clean meta-online-z true: all-Q `85.767`, IQL `15.400`, VDN `58.300`, PQN-VDN `183.600`.
  - clean meta-online-z wrong: all-Q `79.400`, IQL `16.200`, VDN `49.700`, PQN-VDN `172.300`.
  - clean meta-online-z random: all-Q `80.267`, IQL `13.500`, VDN `48.900`, PQN-VDN `178.400`.
- Interpretation:
  - Clean with respect to strict 1-ZSC: no IQL/VDN/PQN-VDN data is used for training/tuning in this run.
  - Reward-side true history control is positive: true beats wrong by `+6.367` and random by `+5.500`.
  - However, true online-z only beats ordinary clean checkpoint blend by `+0.233` and remains below old clean qexp4 VAE formal pair20 anchor `89.017`.
  - Do not run full eval for this version.
  - Takeaway: online partner behavior can matter cleanly, but PG-only clean training currently lacks enough signal to become the main estimator.

## 2026-07-08 Old clean qexp4 VAE online-z ablation

- User asked to test the old clean qexp4 VAE checkpoint on the same pair20 seed10 eval with ordinary blend and online-z controls.
- Checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Report: `reports/old_clean_qexp4_vae_onlinez_pair20_seed10_20260708`.
- Eval settings:
  - strict 1-ZSC partners: IQL/VDN/PQN-VDN only.
  - `max_ego_policies=2`, `max_partner_policies=5`, `num_eval_seeds=10`, both roles.
  - direct blend alpha `0.5`, clip `2.0`.
  - online-z settings matched clean meta run: warmup `50`, ramp `50`, update interval `10`, update steps `5`, online lr `5.0`, prior coef `0.0`, init z `zero`.
- Results:
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - old clean qexp4 VAE ordinary blend: all-Q `90.267`, IQL `16.800`, VDN `59.400`, PQN-VDN `194.600`.
  - old clean qexp4 VAE online-z true: all-Q `87.133`, IQL `18.000`, VDN `57.400`, PQN-VDN `186.000`.
  - old clean qexp4 VAE online-z wrong: all-Q `82.467`, IQL `16.800`, VDN `53.400`, PQN-VDN `177.200`.
  - old clean qexp4 VAE online-z random: all-Q `83.333`, IQL `16.000`, VDN `53.100`, PQN-VDN `180.900`.
- Interpretation:
  - Old clean VAE ordinary blend is very strong in this seed10 run (`90.267`), consistent with formal pair20 100-seed anchor `89.017`.
  - Online-z true remains history-sensitive: it beats wrong by `+4.667` and random by `+3.800`.
  - However, online-z true is `-3.133` below ordinary blend, so the current online CE update is not a drop-in improvement for the best clean checkpoint.
  - Clean meta-online-z created stronger true-vs-wrong online behavior but damaged ordinary blend performance; old clean VAE preserves ordinary performance but is hurt by the zero-init CE online update.
  - Next online direction should preserve the old VAE ordinary latent path and only add online correction through a confidence/trajectory-likelihood gate or a residual update, rather than replacing it with zero-init online z.

## 2026-07-08 Old clean qexp4 VAE + LoRA residual v1

- User suggested LoRA as a way to let `z_p` condition estimator behavior without fully generating decoder weights.
- Implemented LoRA residual:
  - `latent_partner_decoder.py` adds `latent_lora_residual_enabled`.
  - `train_lora_residual_decoder.py` trains only `lora_*`; all old VAE parameters are frozen.
  - Form:
    - `q = CNN(query_obs)`.
    - `z = old VAE encoder(partner history)`.
    - `base_logits = old_VAE_decoder(q, z)`.
    - `delta_logits = gate(q,z) * (((LN(q) @ B) * c(z)) @ A)`.
    - `final_logits = base_logits + clip(delta_logits, -2, 2)`.
- Training:
  - checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_train_20260708`.
  - init checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
  - train data: `reports/latent_decoder_pg2q_full_memsafe_20260701_160807/collect_pg_mixed/pg_mixed_latent_decoder_dataset.npz`.
  - clean: no IQL/VDN/PQN-VDN training data.
  - rank `8`, coeff hidden `64`, gate bias `-2.0`, anchor KL coef `0.05`, residual L2 coef `0.001`, steps `1500`, lr `1e-3`.
- Offline val:
  - true: KL `0.1459` vs frozen old/base KL `0.1565`, TV `0.1652`, acc `0.8000`, top2 `0.9147`, residual TV `0.0451`, gate `0.7718`.
  - wrong: KL `0.1646` vs base `0.1735`, acc `0.7921`.
  - delayed: KL `0.1462` vs base `0.1569`, acc `0.8009`.
  - random: KL `0.1644` vs base `0.1679`, acc `0.7931`.
  - no-history: KL `0.2241` vs base `0.2320`, acc `0.7625`.
  - Interpretation: LoRA improves offline reconstruction, but gate becomes high (`~0.77`) and residual TV is nontrivial (`~0.045`), so v1 is not very conservative.
- Pair20 strict 1-ZSC seed10 report: `reports/old_clean_qexp4_vae_lora_residual_r8_pair20_seed10_20260708`.
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - LoRA residual true: all-Q `89.300`, IQL `17.300`, VDN `57.000`, PQN-VDN `193.600`.
  - LoRA residual wrong: all-Q `88.800`, IQL `15.400`, VDN `55.200`, PQN-VDN `195.800`.
  - LoRA residual random: all-Q `87.600`, IQL `15.200`, VDN `53.200`, PQN-VDN `194.400`.
  - Old clean VAE ordinary reference from same seed10 setup: all-Q `90.267`, IQL `16.800`, VDN `59.400`, PQN-VDN `194.600`.
- Interpretation:
  - LoRA v1 improves strongly over base (`+8.300`) but does not beat old clean VAE ordinary reference (`89.300` vs `90.267`).
  - True beats random by `+1.700`, but wrong is close (`+0.500` gap), so reward-side partner specificity remains weak.
  - Offline KL improvement did not translate into better closed-loop reward; the residual likely changes many states softly without improving the decisive coordination states.
  - Do not run full eval for this v1.
  - Next LoRA direction: make residual more conservative/local by lowering or fixing gate, increasing anchor KL, adding low-disagreement residual penalty, or training only on high-error/high-impact local states.

## 2026-07-08 Old clean qexp4 VAE + LoRA residual v2 frozen-gate hard-query

- A6000 server is usable: `ssh -p 26991 root@hz-4.matpool.com`; GPU check showed NVIDIA RTX A6000 49140 MiB, near-idle. A40 replacement `ssh -p 29898 root@hz-t3.matpool.com` was previously refusing connections.
- Implemented and ran a more conservative LoRA residual:
  - old clean qexp4 VAE frozen.
  - only `lora_*` train.
  - fixed gate via `--freeze_gate`, gate bias `-2.5` so gate `~0.0759`.
  - hard-query mining: `--hard_sample_fraction 0.5`, `--hard_top_fraction 0.3`.
  - stronger regularization: anchor KL `0.2`, residual L2 `0.01`.
- Training checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_frozengate_hard_train_20260708`.
- Offline val:
  - true: KL `0.1517` vs frozen old/base KL `0.1565`, TV `0.1655`, acc `0.7985`, top2 `0.9135`, residual TV `0.0139`, gate `0.0759`.
  - wrong: KL `0.1690` vs base `0.1735`, acc `0.7907`.
  - random: KL `0.1648` vs base `0.1679`, acc `0.7920`.
  - no-history: KL `0.2308` vs base `0.2320`, acc `0.7599`.
  - Interpretation: v2 is much more conservative than v1 and still gives a small clean offline improvement over old VAE.
- Pair20 strict 1-ZSC seed10 report: `reports/old_clean_qexp4_vae_lora_residual_r8_frozengate_hard_pair20_seed10_20260708`.
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - LoRA v2 true: all-Q `89.500`, IQL `18.000`, VDN `55.600`, PQN-VDN `194.900`.
  - LoRA v2 wrong: all-Q `89.633`, IQL `16.300`, VDN `56.700`, PQN-VDN `195.900`.
  - LoRA v2 random: all-Q `87.000`, IQL `14.700`, VDN `52.600`, PQN-VDN `193.700`.
  - Old clean VAE ordinary reference from same seed10 setup: all-Q `90.267`, IQL `16.800`, VDN `59.400`, PQN-VDN `194.600`.
- Interpretation:
  - v2 improves over base by `+8.500`, but still does not beat old clean VAE ordinary reference (`89.500` vs `90.267`).
  - true beats random by `+2.500`, but wrong is slightly higher than true (`89.633` vs `89.500`), so reward-side history specificity remains unreliable.
  - Do not run full eval for v2.
  - Takeaway: conservative LoRA can slightly improve offline local reconstruction without disrupting old VAE much, but direct-blend reward is still dominated by query-prior/base interaction unless history-specific corrections hit decisive closed-loop states.

## 2026-07-08 Old clean qexp4 VAE + standard no-gate LoRA residual

- User questioned whether gate/scale is needed; implemented a cleaner no-gate LoRA:
  - `latent_partner_decoder.py`: optional `lora_no_gate`; when present, `lora_gate=1`.
  - `train_lora_residual_decoder.py`: added `--no_gate` and `--lora_alpha`.
  - Formula: `final_logits = old_VAE_logits + clip((alpha / rank) * LoRA(query_feature, z_p), -2, 2)`.
  - Existing v1/v2 checkpoints remain compatible because they lack `lora_no_gate`.
- Training checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708`.
  - clean PG-only train data: `reports/latent_decoder_pg2q_full_memsafe_20260701_160807/collect_pg_mixed/pg_mixed_latent_decoder_dataset.npz`.
  - no IQL/VDN/PQN-VDN data used.
  - settings: rank `8`, alpha `8`, no gate, anchor KL `0.2`, residual L2 `0.01`, hard sample fraction `0.5`, hard top fraction `0.3`, steps `1500`.
- Offline val:
  - true: KL `0.1475` vs frozen old/base KL `0.1565`, TV `0.1688`, acc `0.7986`, top2 `0.9107`, residual TV `0.0614`, gate `1.0`.
  - wrong: KL `0.1669` vs base `0.1735`, acc `0.7911`.
  - random: KL `0.1678` vs base `0.1679`, acc `0.7904`.
  - no-history: KL `0.2394` vs base `0.2320`, acc `0.7510`.
  - Interpretation: no-gate LoRA learns its own nontrivial correction amplitude; offline true improves over old VAE and true is better than wrong/random, but no-history is worse than the frozen old VAE.
- Pair20 strict 1-ZSC seed10 report: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_pair20_seed10_20260708`.
  - same-run base: all-Q `81.000`, IQL `14.200`, VDN `51.000`, PQN-VDN `177.800`.
  - no-gate LoRA true: all-Q `89.167`, IQL `16.200`, VDN `57.500`, PQN-VDN `193.800`.
  - no-gate LoRA wrong: all-Q `90.667`, IQL `20.500`, VDN `56.700`, PQN-VDN `194.800`.
  - no-gate LoRA random: all-Q `86.833`, IQL `15.100`, VDN `51.500`, PQN-VDN `193.900`.
  - old clean VAE ordinary reference from same seed10 setup: all-Q `90.267`, IQL `16.800`, VDN `59.400`, PQN-VDN `194.600`.
- Interpretation:
  - Removing the gate does not fix the reward issue. True improves over base but is below old VAE reference.
  - Wrong history is higher than true, mostly due to IQL (`20.500` vs `16.200`), so reward-side history specificity remains unreliable.
  - Do not run full eval.
  - Takeaway: learned gate was not the root cause. LoRA can learn stronger offline policy corrections, but direct-blend reward can prefer non-true-history perturbations; offline KL and closed-loop reward remain misaligned.

## 2026-07-08 Online LoRA trajectory-NLL pilot

- User clarified they wanted test-time LoRA updates, not offline-only LoRA. Implemented first online-LoRA path:
  - New eval modes in `policy.py`: `ttac_v5_8_latent_decoder_blend_online_lora` plus wrong/random/delayed controls.
  - Added `online_decoder_params` to `TTACPolicyState`, reset to initial decoder checkpoint at episode end.
  - During rollout, target policy uses `hstate.online_decoder_params`.
  - Online update freezes old VAE/base decoder and updates only trainable `lora_*` parameters.
  - Online data is current episode partner trajectory buffer `(partner_obs, partner_action)`.
  - Objective: support trajectory action NLL under `partner_obs` + anchor KL to initial decoder output + LoRA L2-to-init.
  - Validation gate: split past trajectory into support and latest validation window; accept update only if validation trajectory NLL decreases.
- Pilot report: `reports/online_lora_nogate_alpha8_pair20_seed5_20260708`.
  - Init checkpoint: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708/latent_partner_decoder.npz`.
  - Eval: pair20 strict 1-ZSC seed5; partners IQL/VDN/PQN-VDN.
  - Settings: warmup `50`, ramp `30`, update interval `10`, min history `40`; LoRA lr `1e-3`, update steps `3`, anchor KL `0.05`, L2 `0.001`, val steps `10`.
- Results:
  - base: all-Q `81.533`, IQL `14.600`, VDN `52.400`, PQN-VDN `177.600`.
  - no-update no-gate LoRA: all-Q `89.267`, IQL `16.800`, VDN `56.200`, PQN-VDN `194.800`.
  - online-LoRA true: all-Q `86.200`, IQL `16.400`, VDN `56.600`, PQN-VDN `185.600`.
  - online-LoRA wrong: all-Q `86.267`, IQL `17.000`, VDN `56.000`, PQN-VDN `185.800`.
  - online-LoRA random: all-Q `84.733`, IQL `16.400`, VDN `54.400`, PQN-VDN `183.400`.
- Interpretation:
  - Engineering path works: online LoRA can update inside 1-ZSC rollout from current partner trajectory without partner checkpoint logits.
  - Current full-LoRA trajectory-NLL update is harmful relative to no-update (`86.200` vs `89.267`) and does not create true>wrong reward specificity.
  - true and wrong are nearly identical; random is lower, so the update is not pure noise but reward-side partner specificity is weak.
  - Main damage is PQN-VDN (`194.800 -> 185.600`), suggesting partner_obs behavior-likelihood updates can deform the estimator in ways that do not transfer to ego_obs closed-loop coordination.
  - Do not run full eval or seed10 for this setting. Next: reduce update strength, update only a low-dimensional partner code/coefficient, and/or add accepted-update/NLL-change diagnostics before more reward sweeps.

## 2026-07-08 Meta-learned LoRA online update pilot

- Implemented and tested a first-order meta-LoRA update path:
  - trainer: `train_meta_lora_residual_decoder.py`;
  - checkpoint/report: `reports/meta_lora_from_nogate_r8_inner3_lr002_train_20260708`;
  - init: no-gate LoRA checkpoint `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708/latent_partner_decoder.npz`;
  - inner objective: trajectory NLL `-log pi(a_partner | partner_obs)` + anchor KL + LoRA L2;
  - outer objective: same-view target reconstruction after inner update.
- Offline heldout:
  - true pre/post KL `0.146251 -> 0.146245`, almost no improvement;
  - wrong `0.166010 -> 0.168117`, random `0.166136 -> 0.168805`, no-history `0.233314 -> 0.241207`, all worse after update;
  - interpretation: meta update mostly learns a conservative/no-op correct-history update plus bad-history rejection, not a strong correct-history improvement.
- Pair20 strict 1-ZSC seed5 report: `reports/meta_lora_from_nogate_inner3_lr002_online_pair20_seed5_20260708`.
  - base all-Q `81.533`;
  - meta-LoRA no-update all-Q `89.733`;
  - meta-online-LoRA true all-Q `86.333`;
  - wrong all-Q `83.467`;
  - random all-Q `84.733`.
- Per-family true online: IQL `17.200`, VDN `56.200`, PQN-VDN `185.600`; no-update was IQL `16.800`, VDN `57.600`, PQN-VDN `194.800`.
- Conclusion: meta-LoRA creates some true>wrong/random separation, but true online is still much worse than no-update. Do not run full eval. LoRA is not yet a main method; it remains a diagnostic/possible future path if the update objective can make correct-history post-update same-view prediction materially better.

## 2026-07-08 Old-VAE prior + online residual-C prototype

- Implemented a cleaner residual version of online-z:
  - checkpoint marker: `latent_online_residual_c_enabled`;
  - explicit latent is `[z0, c]`;
  - `z0` comes from the frozen old qexp4 VAE encoder/decoder path;
  - online updates only low-dimensional residual code `c`;
  - prediction is `old_VAE_logits(query_obs, z0) + residual_phi(query_obs, z0, c)`.
- Code touched:
  - `latent_partner_decoder.py`: residual-C decoder support.
  - `policy.py`: online-z path detects residual-C and keeps `z0` fixed while updating `c`.
  - `evaluate_mixed_1zsc.py`: latent dim inference for `[z0,c]`, online smooth coef CLI.
  - new trainer: `train_meta_residual_c_decoder.py`.
- Training/report: `reports/meta_residual_c_pgonly_r8_h128_1k_20260708`.
  - init: old clean qexp4 VAE `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`;
  - train: PG-only `reports/lpr_pgonly_train_dataset_light_20260706/pg_mixed_latent_decoder_dataset.npz`;
  - heldout: PG `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`;
  - settings: `c_dim=8`, hidden `128`, second-order, steps `1000`, inner steps `5`, inner lr `5.0`, inner prior `0.01`.
- Heldout PG offline:
  - true KL `0.4495 -> 0.4463` (`-0.0031`), acc `0.7302`, residual TV `0.0134`;
  - wrong KL `0.4279 -> 0.4229` (`-0.0050`);
  - random KL `0.6553 -> 0.6708` (`+0.0155`);
  - no-history KL `0.4603 -> 0.4469` (`-0.0134`).
- Rollout smoke passed: `reports/meta_residual_c_pgonly_r8_h128_online_smoke_20260708`.
- Conclusion: implementation works and residual `c` moves, but it fails the offline gate. True improves too little, and no-history/wrong also improve, so this version is not yet a clean partner-specific residual correction. Do not run pair20/full for this checkpoint. Next attempt needs explicit negative/no-history preservation or a contrastive residual objective.

## 2026-07-09 Old VAE plug-in on MEP / TrajeDi ego

- User clarified the question: not PPO+oldVAE against MEP/TrajeDi partners, but whether old VAE as a plug-in can improve already-trained ZSC/diverse ego policies.
- Added compatibility for old PPO CNN checkpoints:
  - `models/cnn.py`: `TTAC_LEGACY_CNN` branch for old params `CNNSimple_0`, `LayerNorm_0`, `Dense_0..3`.
  - `evaluate_mixed_1zsc.py`: auto-detects legacy CNN params and sets `TTAC_LEGACY_CNN=True`.
- Light eval setup:
  - MEP ego: `reports/talents_style_staged_pools_20260702_190524/mep_pop_run0`.
  - TrajeDi ego: `reports/talents_style_staged_pools_20260702_190524/trajedi_pop_run0`.
  - adapter: old clean qexp4 VAE `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
  - partners: IQL/VDN/PQN-VDN; `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
  - reports:
    - `reports/oldvae_plugin_mep_ego_q_pair20_seed10_20260709`
    - `reports/oldvae_plugin_trajedi_ego_q_pair20_seed10_20260709`
- Results:
  - MEP ego base all-Q `28.900`; old VAE true `36.033`; wrong `35.267`; random `34.967`.
  - TrajeDi ego base all-Q `29.400`; old VAE true `40.300`; wrong `40.067`; random `40.300`.
  - MEP per-family true: IQL `18.500`, VDN `40.700`, PQN-VDN `48.900`; base was IQL `20.300`, VDN `36.000`, PQN-VDN `30.400`.
  - TrajeDi per-family true: IQL `29.800`, VDN `35.600`, PQN-VDN `55.500`; base was IQL `37.500`, VDN `28.400`, PQN-VDN `22.300`.
- Interpretation:
  - old VAE is additive as a plug-in on these staged ZSC ego checkpoints in this light eval: MEP `+7.133`, TrajeDi `+10.900`.
  - true-history specificity remains weak: MEP true only slightly above wrong/random; TrajeDi true ties random and barely beats wrong.
  - This supports a plug-in cooperative-prior / output-space correction claim, not a strong partner-specific modeling claim.

Correction: the above used staged population members, not final ZSC ego. Do not cite it as the answer to whether old VAE improves true MEP/TrajeDi ZSC ego.

## 2026-07-09 Corrected old qexp4 VAE plug-in on true MEP / TrajeDi ZSC ego

- User clarified: evaluate final ZSC ego, not population members.
- Added evaluator support for nested final ego checkpoints via `--ego_checkpoint_subdir ego`, loading `run_X/ego/ckpt_final`.
- Internal eval mode string is still `ttac_v5_8_latent_decoder_blend`, but the tested method should be described as old clean qexp4 VAE direct blend:
  - `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Corrected light pair20 strict Q-family eval:
  - MEP true ego report: `reports/oldvae_plugin_mep_zsc_ego_q_pair20_seed10_trueonly_20260709`.
  - TrajeDi true ego report: `reports/oldvae_plugin_trajedi_zsc_ego_q_pair20_seed10_trueonly_20260709`.
  - Partners: IQL/VDN/PQN-VDN; `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
  - Only base vs old qexp4 VAE direct blend; wrong/random controls intentionally omitted per user request.
- Results:
  - MEP final ZSC ego: base all-Q `61.600`; old qexp4 VAE direct blend `78.733`; delta `+17.133`.
    - IQL `21.300 -> 32.200`; VDN `52.300 -> 56.300`; PQN-VDN `111.200 -> 147.700`.
  - TrajeDi final ZSC ego: base all-Q `40.033`; old qexp4 VAE direct blend `58.167`; delta `+18.133`.
    - IQL `17.200 -> 30.400`; VDN `44.100 -> 53.300`; PQN-VDN `58.800 -> 90.800`.
- Interpretation:
  - Correct ZSC-ego answer: old qexp4 VAE direct blend is additive on true MEP/TrajeDi final ego in pair20 seed10.
  - But final MEP/TrajeDi ego base scores are surprisingly weak under strict Q-family 1-ZSC, so this does not establish that these ZSC egos are better than PPO state-aug in our setting.
  - Because no wrong/random controls were run in this corrected pass, the result supports plug-in effectiveness, not partner-specific history modeling.

## 2026-07-09 Non-state-aug PPO / MEP / TrajeDi base 1-ZSC

- Ran pair20 seed10 strict Q-family 1-ZSC, base only:
  - partners: IQL/VDN/PQN-VDN;
  - `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
- Reports:
  - PPO: `reports/base_ppo_no_state_1zsc_q_pair20_seed10_20260709`.
  - MEP: `reports/base_mep_no_state_1zsc_q_pair20_seed10_20260709`.
  - TrajeDi: `reports/base_trajedi_no_state_1zsc_q_pair20_seed10_20260709`.
- Checkpoint sources:
  - PPO no-state: `runs/ttac_posthoc_zero_adapter_from_ppo_standard_no_state_aug_64_16_seed42_10seeds_no_state_20260622_133910`, direct `run_X/ckpt_final`.
  - MEP no-state: `runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full`, `run_X/ego/ckpt_final`.
  - TrajeDi no-state: `runs/_archive_legacy_not_main_20260515/superseded_trajedi_run_replaced_by_unified_rerun/trajedi_realpop_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260511-103234/20260511-103246_ggoqyh5r_counter_circuit_avs-full`, `run_X/ego/ckpt_final`.
  - These selected checkpoints have no `STATE_AUG_*` keys in config. First CNN conv is still `(5,5,30,32)`, so channel count alone is not a reliable state-aug indicator here.
  - TrajeDi no-state is from archived/superseded runs; use only as a requested non-state baseline.
- Results:
  - PPO no-state: all-Q `36.733`, IQL `25.500`, VDN `34.400`, PQN-VDN `50.300`.
  - MEP no-state: all-Q `50.200`, IQL `44.700`, VDN `46.500`, PQN-VDN `59.400`.
  - TrajeDi no-state: all-Q `39.433`, IQL `28.000`, VDN `37.400`, PQN-VDN `52.900`.
- Interpretation: non-state-aug PPO is weak; MEP no-state is strongest of the three in this strict Q-family pair20, mostly due to IQL; TrajeDi no-state is only slightly above PPO no-state. State augmentation is a major comparison confounder.
- GAMMA mix0.25 PPO-source base was also evaluated under the same pair20 seed10 strict Q-family setup:
  - source: `runs/gamma_mix025_ppo_standard_source_64_16_10000000_20260514-174603/20260514-174617_nc84elts_counter_circuit_avs-full`;
  - config: `GAMMA.VAE_CHECKPOINT = runs/gamma_full_20260513-090630/vae/gamma_vae.pkl`, `POPULATION_MIX_PROB = 0.25`, no `STATE_AUG_*` keys;
  - report: `reports/base_gamma_mix025_ppo_source_1zsc_q_pair20_seed10_20260709`;
  - result: all-Q `50.867`, IQL `24.600`, VDN `48.700`, PQN-VDN `79.300`.
  - Interpretation: aggregate is roughly tied with MEP no-state (`50.867` vs `50.200`), but GAMMA is much worse on IQL and much better on PQN-VDN.

## 2026-07-09 Old qexp4 VAE plug-in on non-state-aug / GAMMA egos

- Ran old clean qexp4 VAE direct blend on previously untested egos under pair20 seed10 strict Q-family 1-ZSC.
- Setup: partners IQL/VDN/PQN-VDN, `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
- Adapter: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Reports:
  - `reports/oldvae_plugin_ppo_no_state_q_pair20_seed10_trueonly_20260709`
  - `reports/oldvae_plugin_mep_no_state_q_pair20_seed10_trueonly_20260709`
  - `reports/oldvae_plugin_trajedi_no_state_q_pair20_seed10_trueonly_20260709`
  - `reports/oldvae_plugin_gamma_mix025_ppo_source_q_pair20_seed10_trueonly_20260709`
- Results:
  - PPO no-state: base all-Q `36.733`, old VAE `51.200`, delta `+14.467`; IQL `+3.200`, VDN `+8.900`, PQN-VDN `+31.300`.
  - MEP no-state: base all-Q `50.200`, old VAE `62.533`, delta `+12.333`; IQL `+0.400`, VDN `+6.400`, PQN-VDN `+30.200`.
  - TrajeDi no-state: base all-Q `39.433`, old VAE `51.100`, delta `+11.667`; IQL `+3.500`, VDN `+7.100`, PQN-VDN `+24.400`.
  - GAMMA mix0.25 PPO-source: base all-Q `50.867`, old VAE `68.133`, delta `+17.267`; IQL `+5.700`, VDN `+12.800`, PQN-VDN `+33.300`.
- Interpretation: old qexp4 VAE plug-in improves all four; gains are again dominated by PQN-VDN, with consistently positive VDN and smaller IQL gains. This strengthens the plug-in cooperative-correction claim across both state-aug and non-state/GAMMA egos.
