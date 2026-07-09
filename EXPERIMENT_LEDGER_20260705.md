# Experiment Ledger 2026-07-05

Authoritative remote workspace: `/teams/ius_1663576043/hby/rl/ov2`.
Current active server used in this round: replacement A40 `ssh -p 29898 root@hz-t3.matpool.com`.

This ledger separates full evaluations from 20-pair evaluations for the recent TTAC v5.8 / latent partner decoder / estimator repair line.

## Full Evaluations

### XP Full500

| experiment | report | SP | XP | ALL | note |
|---|---|---:|---:|---:|---|
| base no test adapt | `reports/base_no_adapt_full500_20260630_192203` | 201.624 | 156.940 | 161.408 | PPO state-aug base |
| old v5.8 logit bias | `reports/ttac_v5_8_logit_bias_full_500_v58full_20260630_020744` | 199.788 | 163.811 | 167.408 | +6.871 XP over base |
| policy-bank direct blend | `reports/ttac_policy_bank_direct_blend_full500_20260630_174246` | 201.448 | 175.924 | 178.477 | strong XP, but same-seed bank can become SP-like / answer-copying |
| latent decoder direct blend alpha=0.5 | `reports/latent_decoder_xp_full_alpha050_20260702_111538` | 191.312 | 161.995 | 164.926 | weaker than old v5.8 |
| latent decoder + logit bias | `reports/latent_decoder_logit_bias_xp_full_20260702_123108` | 200.724 | 163.264 | 167.010 | near old v5.8, slightly lower |

### 1-ZSC Full100, Healthy Q Pool Only

`1-ZSC` is the evaluation metric/protocol. The table reports the 1-ZSC reward, with the first numeric column using the report's `all_q_partners/both_roles` aggregate over `IQL`, `VDN`, and `PQN-VDN`. `QMIX/SHAQ` are excluded from the healthy-Q pool because their SP sanity is broken.

| experiment | report | 1-ZSC aggregate over healthy Q partners | IQL | VDN | PQN-VDN | note |
|---|---|---:|---:|---:|---:|---|
| base no test adapt | `reports/mixed_1zsc_100seed_20260630_111256` | 57.343 | 27.839 | 55.143 | 89.047 | baseline |
| old v5.8 logit bias | `reports/mixed_1zsc_100seed_20260630_111256` | 62.265 | 29.110 | 60.498 | 97.187 | +4.922 over base |
| latent decoder + logit bias | `reports/latent_decoder_logit_bias_pg2q_full_20260702_123108` | 62.162 | 29.928 | 60.914 | 95.645 | similar to old v5.8 |
| query-attn qexp4 VAE direct blend | `reports/query_attn_qexp4_vae_bottleneck_pg2q_all10_100seed_20260703` | 64.194 | 30.435 | 60.654 | 101.494 | current best clean full 1-ZSC result |

### Full100 With Broken QMIX/SHAQ Included

| experiment | report | 1-ZSC aggregate over 5 Q partner families | qmix_rnn | shaq_ps | note |
|---|---|---:|---:|---:|---|
| base, 5-Q aggregate | `reports/mixed_1zsc_5q_100seed_20260630_155109_resume` | 35.861 | 3.529 | 3.746 | aggregate depressed by broken partners |
| old v5.8, 5-Q aggregate | `reports/mixed_1zsc_5q_100seed_20260630_155109_resume` | 38.884 | 3.302 | 4.324 | not a good main metric |
| VAE direct, 5-Q aggregate | `reports/query_attn_qexp4_vae_bottleneck_5q_all10_100seed_20260703` | 40.360 | 4.235 | 4.982 | slight aggregate lift but still dominated by broken qmix/shaq |

## 20-Pair Evaluations

### XP Pair20

| experiment | report | XP | note |
|---|---|---:|---|
| base no test adapt | `reports/ttac_v5_8_fast_online_pair20_v58controls_20260630_015646` | 161.030 | pair20 base |
| old v5.8 logit bias | same as above | 167.020 | improves reward, but wrong/random/delayed are 166.58/166.91/167.10 |
| policy-bank direct blend true | `reports/ttac_policy_bank_direct_blend_pair20_20260630_170853` | 178.330 | true-history gap large |
| policy-bank direct blend wrong/random/delayed | same as above | 165.460 / 166.600 / 166.020 | strong evidence bank selection uses history, but same-seed bank has cheating risk |
| latent decoder direct alpha=0.5 | `reports/latent_decoder_xp_pair20_alpha050_20260702_111057` | 164.640 | below old v5.8 |
| latent decoder direct alpha=0.75 | `reports/latent_decoder_xp_pair20_alpha075_20260702_111321` | 154.030 | too aggressive |
| latent decoder + logit bias | `reports/latent_decoder_logit_bias_xp_pair20_20260702_115450` | 165.870 | near old v5.8 |
| query-attn decoder, no qexp | `reports/query_attn_decoder_xp_pair20_20260702_143456` | 164.470 direct / 165.880 logit | no clear XP win |
| query-attn qexp4 decoder | `reports/query_attn_qexp4_decoder_xp_pair20_20260702_145733` | 168.520 direct / 166.950 logit | direct is better than old v5.8 on pair20 XP |
| query-attn qexp4 VAE bottleneck | `reports/query_attn_qexp4_vae_bottleneck_xp_pair20_20260702_184052` | 171.150 direct / 167.600 logit | best non-bank XP pair20 |
| v5.8 MLP qexp4 | `reports/v58_mlp_qexp4_pair20_xp_20260702_165503` | 167.850 | small improvement over old v5.8 pair20 |
| TALENTS-style non-Q expanded VAE | `reports/talents_style_nonq_qexp4_vae3k_pair20_summary_20260702_190524` | 169.780 direct / 162.030 logit | XP ok, but strict 1-ZSC regresses badly |

### 1-ZSC Pair20, IQL/VDN/PQN-VDN

The first numeric column is the 1-ZSC reward aggregated over the listed healthy Q partner families.

| experiment | report | 1-ZSC aggregate over healthy Q partners | IQL | VDN | PQN-VDN | note |
|---|---|---:|---:|---:|---:|---|
| base no test adapt | `reports/query_attn_qexp4_pg2q_pair20_1zsc_20260702_150432` | 81.427 | 15.640 | 50.850 | 177.790 | pair20 base |
| v5.2 latest, weighted estimator, 100-seed formal | `reports/v52_latest_pg2q_pair20_1zsc_100seed_2x5_20260705` | 86.237 | 16.360 | 55.020 | 187.330 | correct pair20 slice `2 ego x 5 partners x 2 roles`; same-run base was 81.793 |
| v5.2 latest, weighted estimator, seed10 quick | `reports/v52_latest_pg2q_pair20_1zsc_seed10_2x5_20260705` | 84.233 | 13.000 | 55.300 | 184.400 | correct pair20 slice `2 ego x 5 partners x 2 roles`; 10 eval seeds/role, not the 100-seed formal result; seed10 base in same run was 81.133 |
| old v5.8 logit bias | same as above | 88.243 | 17.230 | 58.420 | 189.080 | pair20 gain |
| query-attn qexp4 direct | same as above | 87.897 | 15.540 | 54.800 | 193.350 | improves PQN, weaker IQL |
| query-attn qexp4 logit | same as above | 87.237 | 16.050 | 56.920 | 188.740 | roughly old v5.8 |
| v5.8 MLP qexp4 | `reports/v58_mlp_qexp4_pg2q_pair20_1zsc_20260702_165629` | 88.143 | 17.750 | 57.380 | 189.300 | similar to old v5.8 |
| query-attn qexp4 VAE direct | `reports/query_attn_qexp4_vae_bottleneck_pg2q_pair20_1zsc_20260702_184235` | 89.017 | 15.990 | 55.190 | 195.870 | best pair20 strict 1-ZSC among healthy-Q tests |
| query-attn qexp4 VAE logit | same as above | 87.517 | 15.700 | 58.400 | 188.450 | lower than direct |
| TALENTS-style non-Q expanded VAE direct | `reports/talents_style_nonq_qexp4_vae3k_pg2q_pair20_1zsc_20260702_190524` | 65.790 | 0.020 | 89.250 | 108.100 | IQL collapses; not usable |
| TALENTS-style non-Q expanded VAE logit | same as above | 68.020 | 0.000 | 96.170 | 107.890 | IQL collapses |
| non-Q no-FCP direct/logit | `reports/talents_style_nonq_no_fcp_qexp4_vae3k_pg2q_pair20_1zsc_20260702_190524` | 64.280 / 67.070 | about 0 | 84.380 / 95.570 | 108.440 / 105.640 | removing FCP did not fix collapse |
| non-Q quality80 direct/logit | `reports/talents_style_nonq_quality80_qexp4_vae3k_pg2q_pair20_1zsc_20260703` | 66.007 / 65.690 | 0 | 85.570 / 93.380 | 112.450 / 103.690 | quality filter did not fix collapse |
| reward-weighted VAE partner loss | `reports/pg_mixed_qexp4_vae_b001_reward_partner_pg2q_pair20_1zsc_20260703` | 70.220 | 0.000 | 95.040 | 115.620 | reward weighting changes family bias, still bad |
| reward-compatible VAE | `reports/pg_mixed_qexp4_vae_b001_reward_compat_c05_pg2q_pair20_1zsc_20260703` | 67.627 | 0.000 | 94.820 | 108.060 | not better |
| oracle same-view partner direct, alpha=0.25 | `reports/oracle_partner_direct_blend_pair20_1zsc_alpha025_20260705` | 90.497 | 17.500 | 59.060 | 194.930 | diagnostic upper bound; true Q partner policy on ego/query obs |
| oracle same-view partner direct, alpha=0.50 | `reports/oracle_partner_direct_blend_pair20_1zsc_alpha050_20260705` | 95.050 | 20.450 | 64.460 | 200.240 | diagnostic upper bound |
| oracle same-view partner direct, alpha=0.75 | `reports/oracle_partner_direct_blend_pair20_1zsc_alpha075_20260705` | 99.767 | 25.930 | 71.760 | 201.610 | diagnostic upper bound |
| oracle same-view partner direct, alpha=1.00 | `reports/oracle_partner_direct_blend_pair20_1zsc_alpha100_20260705` | 103.703 | 30.700 | 81.710 | 198.700 | best aggregate oracle point; not a deployable estimator |
| LPR contrast q-attn VAE, PG-only light | `reports/lpr_and_prototype_true_pair20_1zsc_20260706` | 86.183 | 16.740 | 49.910 | 191.900 | fixed direct blend alpha=0.5; offline history KL gap failed |
| PPO prototype posterior bank | `reports/lpr_and_prototype_true_pair20_1zsc_20260706` | 86.730 | 16.390 | 56.460 | 187.340 | PLASTIC-style posterior over PPO train bank `run_0..7`; MAPPO not included in bank due incompatible param tree |
| counterfactual disagreement q-attn VAE + z-norm | `reports/cf_disagreement_znorm_pair20_1zsc_20260706` | 86.187 | 14.470 | 52.690 | 191.400 | fixed direct blend alpha=0.5; offline history sensitivity works, but pair20 reward below current VAE best `89.017` |
| counterfactual disagreement q-attn VAE + z-norm, sensitivity gate | `reports/cf_znorm_gated_sens_t010_pair20_1zsc_20260706` | 87.113 | 14.650 | 54.700 | 191.990 | gate `sensitivity_no_history`, threshold `0.1`, scale `12`, min `0.05`; improves over ungated `86.187`, but true-control gap is only about `1.1` reward and still below VAE best `89.017` |
| counterfactual disagreement q-attn VAE + z-norm, alpha=1 clip=10 | `reports/cf_disagreement_znorm_pair20_1zsc_alpha1_clip10_20260706` | 55.103 | 7.910 | 26.240 | 131.160 | approximates direct estimator replacement; collapses below base despite true history beating corrupted controls |
| prior-anchor local residual + sensitivity gate | `reports/prior_anchor_residual_sens_t010_pair20_1zsc_20260707` | 88.257 | 16.710 | 56.420 | 191.640 | old qexp4 VAE frozen as prior anchor, local residual trained on grouped PG-only data; close to old VAE `89.017`, better IQL/VDN, worse PQN-VDN; wrong-history control is higher (`88.467`) so partner-specific evidence remains weak |

Oracle same-view diagnostic note: these rows use the actual test-time Q partner checkpoint to produce `pi_partner(. | ego/query obs)` and direct-blend the ego logits toward that distribution. This is intentionally not a clean 1-ZSC method because it accesses the held-out partner policy; it is an upper-bound test for the paper hypothesis that same-view policy matching can be a reward-useful adaptation target if the estimator can reconstruct it.

## Offline Estimator Sanity

| estimator | report | true KL | wrong gap | delayed gap | random gap | true acc | note |
|---|---|---:|---:|---:|---:|---:|---|
| v5.8 MLP qexp4 | `reports/v58_mlp_qexp4_agreement_train_20260702_164934` | 0.1339 | +0.0188 | +0.0019 | +0.0230 | 0.8633 | strong fit, delayed gate fails |
| query-attn qexp4 | `reports/query_attn_qexp4_decoder_train_20260702_144831` | 0.1907 | +0.0640 | +0.0170 | +0.0721 | 0.7535 | stronger corruption gap, worse fit |
| query-attn qexp4 VAE | `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039` | 0.1566 | +0.0170 | +0.0004 | +0.0109 | 0.7958 | best full 1-ZSC despite weak delayed gap |
| TALENTS-style non-Q expanded VAE | `reports/talents_style_nonq_qexp4_vae_b001_3k_train_20260702_190524` | 0.2042 | +0.0419 | +0.0026 | +0.0167 | 0.5457 | bigger data but much worse fit/reward |
| non-Q no-FCP VAE | `reports/talents_style_nonq_no_fcp_qexp4_vae_b001_3k_train_20260702_190524` | 0.2203 | +0.0372 | +0.0024 | +0.0103 | 0.6659 | removing FCP did not recover reward |
| reward-weighted VAE | `reports/pg_mixed_qexp4_vae_b001_reward_partner_3k_train_20260703` | 0.1803 | +0.0223 | +0.0005 | +0.0101 | 0.7850 | reward loss did not align with final cooperation |
| LPR contrast q-attn VAE, PG-only light | `reports/lpr_contrast_qattn_vae_pgonly_light_train_20260706` | 0.5402 | +0.00003 | +0.00000 | -0.00009 | 0.6674 | embedding retrieval 0.741, but decoder output still not history-sensitive |
| counterfactual disagreement q-attn VAE, no z-norm | `reports/cf_disagreement_qattn_vae_pgonly_train_20260706` | 0.9741 | +0.00005 | +0.00001 | -0.00002 | 0.5056 | embedding retrieval 0.813, but same-query output TV only 0.00076 vs target TV 0.593; decoder ignored history latent |
| counterfactual disagreement q-attn VAE + z-norm | `reports/cf_disagreement_qattn_vae_znorm_pgonly_train_20260706` | 0.4156 | +0.0882 | +0.0152 | +0.0944 | 0.6534 | same-query output TV 0.4416 vs target TV 0.593; first clear history-sensitive estimator in this branch |

### Accuracy-First Unified Heldout-PG

Heldout dataset: `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`, PPO/MAPPO run8-9 only, 60k rows. These metrics are not directly comparable to the older internal validation rows above; they use a stricter heldout-PG split and a unified evaluator.

| estimator | report | true KL | true TV | argmax acc | top2 acc | wrong gap | random gap | high-error KL | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| old qexp4 VAE | `reports/accuracy_first_ABC_leaderboard_20260706` | 0.4484 | 0.3130 | 0.7267 | 0.8508 | -0.0212 | -0.0388 | 1.4184 | best single-model global accuracy anchor |
| A qexp4 VAE rerun b512 | `reports/accuracy_first_ABC_leaderboard_20260706` | 0.4672 | 0.3281 | 0.6438 | 0.8721 | +0.2423 | +0.1284 | 1.1076 | better hard slice, worse global |
| B multi-query VAE | `reports/accuracy_first_ABC_leaderboard_20260706` | 0.5722 | 0.3572 | 0.7017 | 0.7604 | +0.1789 | +0.0676 | 1.2710 | multi-query/cross-context did not improve global reconstruction |
| C hard-query VAE | `reports/accuracy_first_ABC_leaderboard_20260706` | 0.5268 | 0.3539 | 0.6700 | 0.8087 | +0.2078 | +0.1108 | 1.1912 | hard mining improves hard slice but hurts global |
| QA-PPNP query-conditioned VAE | `reports/qa_ppnp_qcond_vae_heldout_leaderboard_20260706` | 0.6321 | 0.3963 | 0.5704 | 0.6842 | +0.2409 | +0.0547 | 1.2704 | single-model ANP-style attempt; history-sensitive but poor heldout accuracy |
| old+A diagnostic ensemble | `reports/accuracy_first_ensemble_probe_20260706` | 0.3786 | 0.3046 | 0.7654 | 0.8788 | +0.0816 | +0.0293 | 1.0282 | diagnostic only; not a final method |
| old+C diagnostic ensemble | `reports/accuracy_first_ensemble_probe_20260706` | 0.3979 | 0.3171 | 0.7675 | 0.8800 | +0.0691 | +0.0232 | 1.0254 | diagnostic only; not a final method |

Accuracy-first decision: no new standalone estimator A/B/C or QA-PPNP passed the single-model gate against old qexp4 VAE. The ensemble probe passes offline accuracy but is a diagnostic, not the intended method; after user pushback, do not use ensemble or distillation as the method narrative. The ensemble pair20 1-ZSC run was stopped and should not be cited as a formal reward result. The next estimator direction should be a new single-model training objective/architecture that improves policy reconstruction accuracy under PG-only strict data, not a model-averaging trick.

### Local Decision-Point Estimator Benchmark

Diagnostic only; Q partner data is used for evaluation buckets, not estimator training.

| benchmark | report | rows | key finding |
|---|---|---:|---|
| old VAE vs counterfactual z-norm VAE on Q local decision buckets | `reports/local_decision_benchmark_old_vs_new_20260706` | 1834 | new z-norm VAE is closer to oracle on high partner-disagreement and high oracle-base-TV buckets, but hurts low oracle-base-TV ordinary states more than old VAE/base |

All-partner KL to oracle target:

| bucket | base | old VAE | new z-norm VAE | note |
|---|---:|---:|---:|---|
| all | 2.9062 | 1.6247 | 1.6726 | old slightly better globally |
| high partner disagreement | 3.3158 | 1.8575 | 1.7731 | new better |
| high oracle-base TV | 6.9741 | 2.5701 | 2.1011 | new much better |
| high reward joint | 2.5501 | 1.7041 | 1.7004 | nearly tied |
| low oracle-base TV | 0.0376 | 0.4017 | 0.5461 | base best; new perturbs ordinary states most |

## Q-Partner Repair And New Partner Variants

- Existing VDN sanity check: `reports/vdn_existing_run0_sp100_sanity_20260703`, SP100 `180.000`, so evaluator is valid.
- QMIX/SHAQ repair attempts remain unsuccessful as usable partners:
  - old qmix shaping/persistent/CNN/seq16 and native CNN-QMIX/residual all had SP sanity `0.000` in the 1M/10M single-seed tests.
  - native residual CNN-QMIX 10M report: `reports/native_qmix_residual_10m_seed0_20260703_sp100`, SP100 `0.000`.
- Implemented four new partner sources on 2026-07-05:
  - `iql_dueling`, `vdn_dueling`, `iql_double`, `a2c`.
  - Smoke training and SP5 loader rollouts passed; all short smoke rewards were `0.000`, expected for tiny training.
  - Full 10M single-seed training was started in parallel under `runs/partner_variants_10m_seed42_20260705`; results pending.

## Current Conclusions

- The cleanest current full 1-ZSC improvement is query-attention qexp4 VAE direct blend: 1-ZSC aggregate over healthy Q partners `64.194` vs base `57.343` and old v5.8 `62.265`.
- This supports the claim that partner modeling is mildly useful, but the effect is still small and strongly partner-family dependent.
- The oracle same-view pair20 diagnostic is much stronger: alpha `1.0` reaches `103.703` aggregate vs same-run base `81.793` and v5.2_latest `86.237`. This supports the target definition, while also showing that learned estimator error is the main bottleneck.
- XP pair20 can be improved more strongly, especially by policy-bank direct blend, but policy-bank XP has same-seed / answer-copying risk and is not a clean 1-ZSC estimator result.
- Naive data expansion with MEP/TrajeDi/FCP-style non-Q data hurts strict 1-ZSC because it changes the learned partner-family prior; IQL collapses even when offline history corruption gaps look better.
- Offline teacher KL/accuracy is not sufficient for cooperation reward. VAE improves reward despite weak delayed-history separation, while expanded-data models often have larger wrong-history gaps but worse final reward.
- Counterfactual/disagreement training plus z-normalized VAE query-attention fixes the estimator-history-use diagnostic, but pair20 1-ZSC remains `86.187`, below the older qexp4 VAE direct result `89.017`; reward alignment across Q families remains the bottleneck.
- Directly replacing/overriding base with the learned estimator is not viable yet: alpha `1.0`, clip `10.0` on the history-sensitive estimator collapses pair20 1-ZSC to `55.103`, so the base policy anchor and conservative alpha are currently essential.
- Identifiable grouped training confirms the history signal is not absent, but the current residual-gated single-model estimator overfits grouped same-query reconstruction and fails the original heldout-PG accuracy gate; no pair20 reward eval was run for it.
- QMIX/SHAQ should remain excluded from the main partner pool until their SP sanity is fixed.

## 2026-07-07 Identifiable Local Policy Estimator

Implemented a PG-only strict single-model residual-gated estimator:

```text
query_obs -> query prior logits
partner history -> VAE latent
prior logits + learned gate * clipped history residual -> pi_hat_partner(. | query_obs)
```

New tooling:

- grouped dataset builder: `build_identifiable_grouped_dataset.py`
- identifiability diagnostic: `diagnose_identifiable_dataset.py`
- trainer support: `--residual_gated`, prior KL, same-query pairwise TV loss, low-disagreement residual penalty, query-group-balanced batches
- evaluator support: low/high `query_disagreement_tv` slices

Stage 0 anchor was reproduced exactly on the original heldout-PG dataset:

| estimator | heldout dataset | KL | TV | acc | top2 |
|---|---|---:|---:|---:|---:|
| old qexp4 VAE | `reports/accuracy_first_pg_heldout_dataset_20260706` | 0.4484 | 0.3130 | 0.7267 | 0.8508 |

Stage 1 grouped datasets:

| dataset | report | rows | policies | mean query TV | p50 TV | p90 TV | mean residual TV |
|---|---|---:|---:|---:|---:|---:|---:|
| train run0-7 | `reports/identifiable_grouped_train_pgonly_20260707` | 160000 | 16 | 0.4618 | 0.4738 | 0.5721 | 0.3424 |
| heldout run8-9 | `reports/identifiable_grouped_heldout_pg_20260707` | 60000 | 4 | 0.5569 | 0.5474 | 0.7778 | 0.3629 |

Identifiability diagnostic using PG prototype history likelihood:

| dataset | report | rows used | retrieval acc | true posterior mean | high-disagreement low-separability fraction |
|---|---|---:|---:|---:|---:|
| train grouped | `reports/identifiable_diag_train_pgonly_20260707` | 8000 | 0.9589 | 0.2609 | 0.5388 |
| heldout grouped | `reports/identifiable_diag_heldout_pg_20260707` | 8000 | 0.9499 | 0.7614 | 0.0024 |

Original heldout-PG accuracy gate:

| estimator | report | KL | TV | acc | top2 | wrong gap | random gap | delayed gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old qexp4 VAE | `reports/identifiable_stage0_anchor_20260707` | 0.4484 | 0.3130 | 0.7267 | 0.8508 | -0.0212 | -0.0388 | +0.0041 |
| residual-gated | `reports/identifiable_residual_gated_original_heldout_eval_20260707` | 0.6062 | 0.4063 | 0.5054 | 0.6762 | +0.2209 | +0.0676 | -0.0491 |
| no pairwise loss | `reports/identifiable_residual_no_pairwise_original_heldout_eval_20260707` | 0.7223 | 0.4353 | 0.4996 | 0.6321 | +0.1152 | -0.0278 | -0.0323 |
| no low-disagreement penalty | `reports/identifiable_residual_no_lowpen_original_heldout_eval_20260707` | 0.6273 | 0.4073 | 0.5042 | 0.6504 | +0.2036 | +0.0581 | -0.0464 |

Grouped heldout-PG reconstruction:

| estimator | report | KL | acc | high-disagreement KL | low-disagreement KL | note |
|---|---|---:|---:|---:|---:|---|
| old qexp4 VAE | `reports/identifiable_residual_gated_grouped_heldout_eval_20260707` | 0.6753 | 0.4538 | 0.9546 | 0.0894 | strong ordinary-state prior |
| residual-gated | same | 0.5391 | 0.4479 | 0.7655 | 0.1437 | better grouped/high-disagreement KL, worse low-disagreement |
| no pairwise loss | `reports/identifiable_residual_no_pairwise_grouped_heldout_eval_20260707` | 0.6115 | 0.4486 | 0.9657 | 0.0940 | pairwise loss helps the intended grouped objective |
| no low-disagreement penalty | `reports/identifiable_residual_no_lowpen_grouped_heldout_eval_20260707` | 0.5437 | 0.4528 | 0.7929 | 0.1359 | low penalty has only small effect |

Decision:

- No residual-gated variant passed the original heldout-PG accuracy gate (`KL < 0.4484`, acc `> 0.7267`, top2 `>= 0.8508`).
- No pair20 strict 1-ZSC was run at this gate stage.
- Interpretation: the history signal is identifiable, and grouped training can improve the targeted same-query/high-disagreement reconstruction objective, but the current model sacrifices the natural rollout/global prior that made old qexp4 VAE useful.

User-requested pair20 follow-up despite the offline gate failure:

| estimator / mode | report | all_q | IQL | VDN | PQN-VDN | note |
|---|---|---:|---:|---:|---:|---|
| same-run base | `reports/identifiable_residual_gated_sens_t010_pair20_1zsc_20260707` | 81.793 | 15.820 | 51.690 | 177.870 | base_no_test_adapt |
| residual-gated, sensitivity gate | same | 85.800 | 14.840 | 54.110 | 188.450 | `sensitivity_no_history`, threshold 0.10, scale 12, min gate 0.05 |
| residual-gated, ungated | `reports/identifiable_residual_ungated_pair20_1zsc_20260707` | 85.593 | 14.240 | 52.940 | 189.600 | direct latent blend without eval-time gate |
| residual-gated, wrong history | `reports/identifiable_residual_gated_sens_t010_pair20_1zsc_20260707` | 85.217 | 15.670 | 52.170 | 187.810 | control |
| residual-gated, random history | same | 84.360 | 15.590 | 51.460 | 186.030 | control |
| residual-gated, delayed history | same | 85.553 | 15.090 | 54.020 | 187.550 | control |

Interpretation:

- The residual-gated estimator improves this pair20 strict 1-ZSC aggregate over same-run base by `+4.007`, mostly from PQN-VDN (`+10.580`) and VDN (`+2.420`), while hurting IQL (`-0.980`).
- The simple sensitivity gate barely changes aggregate reward versus ungated (`85.800` vs `85.593`, `+0.207`), so the pair20 gain mainly comes from the residual estimator target rather than the gate.
- True-history control separation is weak: true beats wrong by `+0.583`, random by `+1.440`, and delayed by `+0.247`, all below the planned `+2.0` criterion.
- The method is therefore a useful local-policy modeling signal, but not a solved estimator and not stronger than the old qexp4 VAE pair20 anchor (`89.017`).

## 2026-07-07 Prior-Anchor Local Residual Estimator

Implemented a single-checkpoint prior-anchor residual variant:

```text
old qexp4 VAE logits(query_obs, history)  -- frozen prior anchor
    + learned gate(query_obs, history) * learned residual(query_obs, history)
    -> pi_hat_partner(. | query_obs, history)
```

Implementation notes:

- Code changed:
  - `latent_partner_decoder.py`: supports embedded `prior_anchor__*` params and uses them as frozen prior logits inside the residual-gated decoder.
  - `train_latent_partner_decoder.py`: adds `--prior_decoder_path`, matching-parameter initialization from the prior, frozen prior-anchor gradients, `--gate_target_coef`, and `--residual_tv_coef`.
- Prior anchor: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Formal train: `reports/prior_anchor_residual_pgonly_train_20260707`.
- Training data: grouped PG-only train set `reports/identifiable_grouped_train_pgonly_20260707/identifiable_grouped_latent_decoder_dataset.npz`.
- Key train settings: `steps=3000`, `lr=2e-4`, `query_group_balanced_batch`, `pairwise_tv_coef=0.2`, `low_disagreement_residual_coef=0.2`, `residual_tv_coef=0.02`, `gate_target_coef=0.05`, `residual_clip=2.0`.

Internal validation from the train script:

| metric | value |
|---|---:|
| true KL | 0.3641 |
| true TV | 0.2942 |
| argmax acc | 0.6196 |
| top2 acc | 0.7794 |
| wrong KL gap | +0.0258 |
| random KL gap | +0.0330 |
| delayed KL gap | +0.0021 |
| same-query output TV / target TV | 0.2832 / 0.4678 |
| output-target TV corr | 0.4043 |
| residual TV | 0.1979 |
| gate mean | 0.7996 |

Offline heldout evaluation:

| dataset / slice | old qexp4 VAE KL | prior-anchor residual KL | interpretation |
|---|---:|---:|---|
| original heldout global | 0.4484 | 0.5195 | global accuracy worsens |
| original high-baseline-error | 1.4185 | 1.1757 | local correction helps where old prior is wrong |
| grouped heldout global | 0.6753 | 0.5675 | grouped/local objective improves |
| grouped high-disagreement | 0.9546 | 0.7678 | local same-query disagreement improves |
| grouped low-disagreement | 0.0894 | 0.1477 | ordinary low-disagreement states degrade |
| grouped high-baseline-error | 1.8642 | 1.2735 | large local improvement |

Pair20 strict 1-ZSC with sensitivity gate (`threshold=0.10`, `scale=12`, `min_gate=0.05`):

| mode | all_q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.793 | 15.820 | 51.690 | 177.870 |
| prior-anchor residual true history | 88.257 | 16.710 | 56.420 | 191.640 |
| prior-anchor residual wrong history | 88.467 | 17.950 | 56.480 | 190.970 |
| prior-anchor residual random history | 86.740 | 16.580 | 53.360 | 190.280 |
| prior-anchor residual delayed history | 87.923 | 16.480 | 56.500 | 190.790 |

Interpretation:

- This version is much closer to the paper direction than the old VAE alone: it explicitly uses old VAE as a population/same-view prior and learns a local residual correction.
- It improves pair20 strict 1-ZSC over same-run base by `+6.463` and is essentially tied with old v5.8 logit-bias pair20 (`88.243`), but remains below the old qexp4 VAE direct anchor (`89.017`).
- It improves IQL and VDN relative to old qexp4 VAE pair20 (`IQL 16.710 vs 15.990`, `VDN 56.420 vs 55.190`) but loses on PQN-VDN (`191.640 vs 195.870`).
- The failure mode is still the same: true history is not clearly better than corrupted histories. Wrong history is slightly higher (`88.467` vs true `88.257`), delayed is very close (`87.923`), and random is lower (`86.740`).
- Conclusion: prior-anchor residual confirms the local-correction idea has usable reward signal and improves the right local slices, but the estimator/gate still does not reliably attach that correction to the correct partner history. Do not run full eval for this version.

## 2026-07-07 Q-Family Estimator Training Diagnostic

Purpose: user requested a deliberately non-strict diagnostic to test whether the current estimator bottleneck is mainly a training-data distribution problem. Unlike the clean PG-only setting, this experiment trains the estimator directly on the 1-ZSC target partner families (`IQL`, `VDN`, `PQN-VDN`). This result must not be reported as a clean strict 1-ZSC method.

Dataset:

- Raw collector output: `reports/qfamily_diagnostic_raw_dataset_20260707/pg_mixed_latent_decoder_dataset.npz`, 144k rows.
- Q-target filtered dataset: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, 96k rows.
- Partner-family rows: `PQN-VDN 38k`, `VDN 36k`, `IQL 22k`.
- Collection settings: `q_methods=iql,vdn,pqn_vdn`, `max_q_policies=5`, `max_q_pairings_per_block=12`, `num_eval_seeds=20`, `transition_stride=4`, `history_len=50`, `q_action_mode=greedy`.

Estimator:

- Checkpoint: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
- Architecture/settings: query-attn qexp4 VAE, `vae_beta=0.001`, `decoder_z_norm=true`, `query_expansion_k=4`, `query_expansion_scope=partner_policy`, `cross_context_fraction=0.25`, `steps=3000`, `batch_size=512`.

Offline validation from the train script:

| metric | value |
|---|---:|
| train KL/CE at step 3000 | 0.0927 |
| train TV at step 3000 | 0.0543 |
| train argmax acc at step 3000 | 0.9644 |
| train top2 acc at step 3000 | 0.9941 |
| val true KL | 0.3581 |
| val true TV | 0.1021 |
| val argmax acc | 0.9160 |
| val top2 acc | 0.9698 |
| wrong KL gap | +0.0041 |
| delayed KL gap | +0.0009 |
| random KL gap | +0.0140 |
| embedding retrieval acc | 0.6362 |

Pair20 strict 1-ZSC diagnostic with this Q-trained estimator:

| mode | all_q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.793 | 15.820 | 51.690 | 177.870 |
| Q-trained estimator true history | 89.120 | 16.780 | 56.510 | 194.070 |
| Q-trained estimator wrong history | 88.573 | 15.740 | 56.330 | 193.650 |
| Q-trained estimator random history | 89.287 | 16.710 | 56.780 | 194.370 |
| Q-trained estimator delayed history | 88.777 | 16.470 | 56.170 | 193.690 |
| old clean qexp4 VAE true history anchor | 89.017 | 15.990 | 55.190 | 195.870 |

Interpretation:

- Directly training on Q-family target data raises Q-family policy reconstruction accuracy substantially and reaches the old clean qexp4 VAE pair20 aggregate (`89.120` vs `89.017`).
- It improves IQL and VDN relative to old clean qexp4 VAE (`IQL 16.780 vs 15.990`, `VDN 56.510 vs 55.190`), but is lower on PQN-VDN (`194.070 vs 195.870`).
- The control result is the important warning: random history is slightly higher than true history (`89.287` vs `89.120`), and wrong/delayed are close. Therefore the reward gain is mostly from learning a Q-family same-view action prior / target distribution, not from reliable partner-specific history binding.
- Conclusion: this supports the data-distribution hypothesis, but only partially. Q-family data fixes much of the Q-policy reconstruction gap, while the separate problem of using short history to identify the correct local partner tendency remains unsolved.
- No full eval: the experiment is intentionally contaminated and fails the true-history control criterion.

Follow-up: old clean qexp4 VAE was evaluated on the same Q-family offline val dataset for a fair reconstruction comparison.

- Report: `reports/qfamily_diagnostic_old_vs_qtrained_offline_20260707/accuracy_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, `split=val`, 19,400 rows.

| estimator | true KL | true TV | argmax acc | top2 acc | wrong gap | random gap | delayed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| old clean qexp4 VAE | 1.5416 | 0.5824 | 0.5048 | 0.6327 | -0.0093 | +0.0064 | +0.0012 |
| Q-trained qexp4 VAE | 0.3539 | 0.1021 | 0.9160 | 0.9698 | +0.0039 | +0.0141 | +0.0009 |

Interpretation of this follow-up:

- On the same Q-family offline reconstruction set, Q-trained VAE is dramatically more accurate than old clean VAE.
- This means the old clean VAE's strong pair20 reward does not come from accurately reconstructing Q-family partner policies.
- The remaining puzzle is therefore sharper: large reconstruction improvement (`KL 1.5416 -> 0.3539`, `acc 0.5048 -> 0.9160`) only gives negligible pair20 reward improvement (`89.017 -> 89.120`).
- Most plausible explanation: old VAE already provides a reward-useful cooperative prior, while Q-trained VAE improves many offline states that are not reward-critical; additionally, Q-trained still fails to show reliable true-history conditioning.

Query-observation sensitivity diagnostic on the same Q-family val split:

| estimator | query input | KL to target | TV to target | argmax acc | top2 acc | TV vs real-query output |
|---|---|---:|---:|---:|---:|---:|
| old clean qexp4 VAE | real query | 1.5416 | 0.5824 | 0.5048 | 0.6327 | 0.0000 |
| old clean qexp4 VAE | shuffled query | 2.5006 | 0.7864 | 0.2426 | 0.4095 | 0.5360 |
| old clean qexp4 VAE | zero query | 2.9865 | 0.8349 | 0.1432 | 0.2554 | 0.6926 |
| old clean qexp4 VAE | gaussian query | 2.7666 | 0.7904 | 0.2216 | 0.4032 | 0.5819 |
| Q-trained qexp4 VAE | real query | 0.3539 | 0.1021 | 0.9160 | 0.9698 | 0.0000 |
| Q-trained qexp4 VAE | shuffled query | 6.9045 | 0.7516 | 0.2472 | 0.4617 | 0.7489 |
| Q-trained qexp4 VAE | zero query | 6.4070 | 0.8913 | 0.0980 | 0.2770 | 0.8764 |
| Q-trained qexp4 VAE | gaussian query | 5.9453 | 0.7533 | 0.2497 | 0.4369 | 0.7438 |

Interpretation: old VAE is not a constant action prior. Its output changes strongly when `query_obs` is shuffled, zeroed, or randomized. Its reward usefulness is therefore better described as a query-conditioned cooperative prior with weak/poor partner-history binding, not as an observation-independent fixed distribution.

Pair20 1-ZSC query-randomization diagnostic:

- Code path: `TTAC_LATENT_DECODER_QUERY_MODE=gaussian`, which replaces only the latent decoder's `query_obs` with step-dependent Gaussian noise. The ego base policy still receives the true environment observation, and partner history remains true.
- Report: `reports/old_vae_gaussian_query_pair20_1zsc_20260707/mixed_1zsc_summary.md`.
- Estimator: old clean qexp4 VAE `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.

| mode | all_q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.793 | 15.820 | 51.690 | 177.870 |
| old qexp4 VAE real query | 89.017 | 15.990 | 55.190 | 195.870 |
| old qexp4 VAE gaussian query | 77.447 | 15.370 | 48.010 | 168.960 |

Interpretation:

- Randomizing only the estimator query drops old VAE from `89.017` to `77.447`, a `-11.570` drop relative to real-query VAE and `-4.347` below base.
- All three Q families degrade, especially PQN-VDN (`195.870 -> 168.960`).
- Therefore old VAE's 1-ZSC gain strongly depends on seeing the real current ego/query observation. It is not an observation-independent constant prior.
- This supports the refined interpretation: old VAE is a query-conditioned cooperative prior with weak partner-history binding, not a constant distribution and not accurate Q-family partner reconstruction.

Q-trained VAE direct-policy diagnostic:

- Code added: `ttac_v5_8_latent_decoder_policy` eval mode in `ttac_v5_8_fast_online/policy.py`.
- Behavior: when history is available, ego action distribution is replaced by the latent decoder's `target_probs` directly. No base-policy blend is used; base only acts as the no-history first-step fallback.
- Report: `reports/qtrained_vae_policy_replace_pair20_1zsc_20260707/mixed_1zsc_summary.md`.
- Estimator: Q-family diagnostic qexp4 VAE `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.

| mode | all_q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.793 | 15.820 | 51.690 | 177.870 |
| Q-trained VAE direct blend | 89.120 | 16.780 | 56.510 | 194.070 |
| Q-trained VAE direct policy replacement | 42.923 | 10.000 | 21.140 | 97.630 |

Interpretation:

- Directly executing the estimator distribution collapses pair20 1-ZSC (`42.923`), far below both base (`81.793`) and blend (`89.120`).
- Therefore accurate same-view partner-policy reconstruction is not itself a good ego policy.
- The useful signal comes from using the estimator output as a bounded correction target for a competent base policy, not from replacing the base policy.
- This supports keeping the adapter as a conservative output-space correction while improving estimator/gating, rather than turning the estimator into a standalone controller.

Old-vs-Q-trained VAE output diagnostic on the same Q-family val split:

- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val, 19,400 rows.
- Diagnostic script: `/tmp/compare_old_qtrained_outputs.py` on the A40 server.
- Note: Q-family targets were collected with `q_action_mode=greedy`, so `target_partner_probs` are effectively one-hot (`target entropy 0.0`).

| estimator / history | KL to target | TV to target | argmax acc | entropy | mean max prob |
|---|---:|---:|---:|---:|---:|
| old VAE true history | 1.5416 | 0.5824 | 0.5048 | 1.0754 | 0.6210 |
| old VAE no history | 1.4154 | 0.5948 | 0.5158 | 1.1453 | 0.6024 |
| Q-trained VAE true history | 0.3539 | 0.1021 | 0.9160 | 0.1275 | 0.9530 |
| Q-trained VAE no history | 0.4379 | 0.1422 | 0.8811 | 0.1900 | 0.9341 |

Pairwise output comparison:

| comparison | mean TV | top-action same |
|---|---:|---:|
| old true vs Q-trained true | 0.5533 | 0.5080 |
| old true vs old no-history | 0.1299 | 0.9246 |
| Q-trained true vs Q-trained no-history | 0.0763 | 0.9376 |

By partner family:

| family | rows | old KL | Q-trained KL | old-vs-Q TV | old acc | Q-trained acc |
|---|---:|---:|---:|---:|---:|---:|
| IQL | 4,300 | 2.2420 | 0.2382 | 0.7178 | 0.4316 | 0.9616 |
| VDN | 6,900 | 1.7726 | 0.4130 | 0.6214 | 0.4201 | 0.9199 |
| PQN-VDN | 8,200 | 0.9801 | 0.3648 | 0.4097 | 0.6145 | 0.8888 |

Interpretation:

- Old VAE and Q-trained VAE are not learning the same output distribution. Their mean output TV is large (`0.5533`), and their top action matches only `50.8%` of the time.
- Q-trained VAE learns a much sharper deterministic Q-greedy predictor (`entropy 0.1275`, max prob `0.9530`), while old VAE remains much softer (`entropy 1.0754`, max prob `0.6210`).
- However, neither model uses history strongly: old true-vs-no-history TV is only `0.1299`, and Q-trained true-vs-no-history TV is only `0.0763`; top action stays the same in more than `92%` of rows for both.
- Therefore the two models differ fundamentally in query-conditioned target type, not in solving partner-specific history-conditioned modeling.

## 2026-07-07 Q-Family Same-Query Counterfactual VAE

Purpose: user asked to continue method A: keep the Q-family diagnostic setup (`IQL`, `VDN`, `PQN-VDN`) but reconstruct the training data so the same `query_obs` is paired with multiple concrete Q partner histories and targets. This tests whether the previous Q-trained VAE was stuck at a Q-family query prior because history was not forced to disambiguate same-query partner differences.

Implementation:

- Added builder: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/build_qfamily_grouped_dataset.py`.
- Source dataset: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`.
- Output dataset: `reports/qfamily_grouped_qcounter_dataset_20260707/qfamily_grouped_latent_decoder_dataset.npz`.
- Loaded Q policies: 14 concrete policies from `IQL`, `VDN`, `PQN-VDN`.
  - `iql:run_4` was skipped because it had no support rows in the source dataset.
- Dataset shape:
  - rows: `159,992`
  - query groups: `11,428`
  - policies per query: `14`
  - mean query disagreement TV: `0.6936`
  - p50 query disagreement TV: `0.7033`
  - p90 query disagreement TV: `0.8022`
  - mean residual target TV: `0.6441`
  - same-episode support fraction: `0.0`
  - target entropy: `0.0` because Q targets are greedy one-hot.

Training:

- Checkpoint: `reports/qfamily_grouped_qcounter_vae_k1_train_20260707/latent_partner_decoder.npz`.
- Architecture: same query-attn VAE family as Q-trained VAE, but with same-query grouped training.
- Key settings:
  - `arch=query_attn`
  - `vae=true`
  - `vae_beta=0.001`
  - `decoder_z_norm=true`
  - `query_expansion_k=1`
  - `query_expansion_scope=none`
  - `query_group_balanced_batch=true`
  - `query_groups_per_batch=64`
  - `rows_per_query_group=14`
  - `steps=3000`
  - `batch_size=1024`
- Rationale for `query_expansion_k=1`: qexp4 would pair a history with extra non-identical queries and dilute the same-query counterfactual pressure. This first diagnostic keeps the same-query signal clean.

Internal grouped validation:

| metric | value |
|---|---:|
| true KL | 0.3757 |
| true TV | 0.1588 |
| argmax acc | 0.8726 |
| top2 acc | 0.9544 |
| wrong KL gap | +0.0592 |
| random KL gap | +0.0213 |
| delayed KL gap | +0.0008 |
| same-query output TV | 0.6715 |
| same-query target TV | 0.7011 |
| output-target TV corr | 0.7964 |
| output/target TV ratio | 0.9579 |
| embedding retrieval acc | 0.8991 |

Same Q-target val comparison against ordinary Q-trained VAE:

- Report: `reports/qfamily_qtrained_vs_qcounter_offline_20260707/accuracy_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val, 19,400 rows.

| estimator | true KL | true TV | argmax acc | top2 acc | wrong gap | random gap | delayed gap | no-history gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary Q-trained qexp4 VAE | 0.3539 | 0.1021 | 0.9160 | 0.9698 | +0.0039 | +0.0141 | +0.0009 | +0.0840 |
| Q same-query counterfactual VAE | 0.3561 | 0.1621 | 0.8771 | 0.9552 | +0.0713 | +0.0161 | +0.0030 | +1.4431 |

Pair20 1-ZSC:

- Report: `reports/qfamily_grouped_qcounter_vae_k1_pair20_1zsc_20260707/mixed_1zsc_summary.md`.

| mode | all_q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.793 | 15.820 | 51.690 | 177.870 |
| Q same-query counterfactual VAE true history | 84.700 | 13.720 | 52.680 | 187.700 |
| Q same-query counterfactual VAE wrong history | 84.997 | 13.910 | 53.280 | 187.800 |
| Q same-query counterfactual VAE random history | 84.360 | 13.270 | 52.310 | 187.500 |
| Q same-query counterfactual VAE delayed history | 85.037 | 13.760 | 52.740 | 188.610 |
| ordinary Q-trained qexp4 VAE true history anchor | 89.120 | 16.780 | 56.510 | 194.070 |
| old clean qexp4 VAE true history anchor | 89.017 | 15.990 | 55.190 | 195.870 |

Interpretation:

- Same-query counterfactual training works offline in the narrow sense: it strongly increases history dependence, improves wrong/no-history gaps, and learns same-query policy variation well (`output TV 0.6715` vs target TV `0.7011`, corr `0.7964`).
- However, this does not translate to better pair20 1-ZSC. True-history reward is only `84.700`, below ordinary Q-trained VAE (`89.120`) and old clean VAE (`89.017`).
- The reward controls still fail the partner-specific criterion: wrong (`84.997`) and delayed (`85.037`) are higher than true (`84.700`), and random (`84.360`) is close.
- Family breakdown suggests the same-query model hurts IQL relative to base (`13.720` vs `15.820`) and mainly gains on PQN-VDN (`187.700` vs `177.870`), but not enough to match prior anchors.
- Conclusion: the data construction successfully forces the network to represent same-query partner differences offline, but the learned correction is not reward-useful under fixed direct blend. Same-query counterfactual alone does not solve test-time partner modeling; it may need a gate, qexp/natural-query mixture, or online latent fitting before it is usable.
- No full eval for this version.

Follow-up policy-geometry diagnostic:

- Purpose: user asked whether the three trained estimators shrink or enlarge the policy-distribution gap to 1-ZSC partner models when used as estimators.
- Added diagnostic script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_estimator_policy_geometry.py`.
- Report: `reports/estimator_policy_geometry_ppoego_qval_20260707/policy_geometry_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val.
- Filter: only `ppo:run_0` and `ppo:run_1` ego rows, 4,700 rows, to match pair20 1-ZSC ego policies.
- Metric: compare true Q partner policy `pi_partner(.|ego_obs)` with base PPO and with direct-blended PPO (`alpha=0.5`, `clip=2.0`). Negative delta means blend moves ego closer to partner.

| estimator | partner | base TV | estimator TV | blend TV | delta TV | shrink rows | base KL | estimator KL | blend KL | delta KL | KL shrink rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old clean VAE | all_q | 0.5238 | 0.5860 | 0.5246 | +0.0008 | 55.7% | 1.8360 | 1.3882 | 1.5359 | -0.3001 | 55.7% |
| Q-trained VAE | all_q | 0.5238 | 0.1202 | 0.4135 | -0.1102 | 92.7% | 1.8360 | 0.4432 | 1.0994 | -0.7365 | 92.7% |
| Q same-query counterfactual VAE | all_q | 0.5238 | 0.1963 | 0.4396 | -0.0841 | 82.2% | 1.8360 | 0.4610 | 1.1560 | -0.6800 | 82.2% |

Family breakdown:

| estimator | IQL delta TV | VDN delta TV | PQN-VDN delta TV | IQL shrink rows | VDN shrink rows | PQN shrink rows |
|---|---:|---:|---:|---:|---:|---:|
| old clean VAE | +0.0958 | -0.0308 | -0.0138 | 10.0% | 79.2% | 47.5% |
| Q-trained VAE | -0.0449 | -0.1382 | -0.1088 | 97.5% | 93.0% | 88.5% |
| Q same-query counterfactual VAE | -0.0397 | -0.1145 | -0.0622 | 89.9% | 86.9% | 67.5% |

Additional distribution shape:

| estimator | estimator argmax acc | blend argmax acc | estimator entropy |
|---|---:|---:|---:|
| old clean VAE | 0.5877 | 0.5543 | 1.1703 |
| Q-trained VAE | 0.8957 | 0.6160 | 0.1519 |
| Q same-query counterfactual VAE | 0.8481 | 0.5787 | 0.3610 |

Estimator argmax acc by partner family, before blend:

| estimator | all-Q acc | IQL acc | VDN acc | PQN-VDN acc |
|---|---:|---:|---:|---:|
| old clean VAE | 0.5877 | 0.9400 | 0.3892 | 0.6831 |
| Q-trained VAE | 0.8957 | 0.9640 | 0.8971 | 0.8408 |
| Q same-query counterfactual VAE | 0.8481 | 0.9410 | 0.8667 | 0.7423 |

Interpretation:

- In policy-geometry terms, Q-trained VAE clearly moves the blended ego distribution closer to true Q partners (`delta TV -0.1102`, `92.7%` shrink rows). Q-counterfactual VAE also moves closer, but less strongly (`delta TV -0.0841`, `82.2%` rows).
- Old clean VAE does not consistently move closer in TV overall (`+0.0008`) and strongly moves away on IQL (`+0.0958`, only `10%` rows shrink), although it reduces KL overall because it assigns more mass to the target action on high-KL rows.
- Therefore previous reward behavior cannot be explained by "moving closer to partner distribution" alone. Q-trained is best at shrinking the policy gap, but its reward is only tied with old VAE; qcounter shrinks the gap but has worse reward.
- This supports a sharper conclusion: partner-policy closeness is achievable and measurable, but fixed direct blend reward depends on which states/families are moved and whether the induced correction remains compatible with base ego behavior.

Follow-up noise-history distance diagnostic for Q-trained VAE:

- Purpose: user asked whether Q-trained VAE with correct history is clearly closer to true Q partner policy than with random-noise history.
- Script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_noise_history_distance.py`.
- Report: `reports/qtrained_vae_noise_history_distance_20260707/noise_history_distance_summary.md`.
- Dataset/split: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`, val.
- Decoder: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.

All Q-family val rows:

| history input | KL | TV | argmax acc | entropy |
|---|---:|---:|---:|---:|
| true history | 0.3539 | 0.1021 | 0.9160 | 0.1275 |
| shuffled real row history | 0.8480 | 0.1870 | 0.8281 | 0.1662 |
| Gaussian std1 obs + uniform actions | 0.8057 | 0.1744 | 0.8387 | 0.1533 |
| Gaussian matched obs + empirical actions | 1.1457 | 0.1983 | 0.8120 | 0.1285 |
| zero obs + uniform actions | 0.4518 | 0.1418 | 0.8747 | 0.1843 |

Pair20-relevant PPO ego rows (`ppo:run_0/run_1`, 4,700 rows):

| history input | KL | TV | argmax acc | entropy |
|---|---:|---:|---:|---:|
| true history | 0.4432 | 0.1202 | 0.8957 | 0.1519 |
| shuffled real row history | 0.5788 | 0.1553 | 0.8634 | 0.1825 |
| Gaussian std1 obs + uniform actions | 0.5685 | 0.1503 | 0.8691 | 0.1785 |
| Gaussian matched obs + empirical actions | 0.5863 | 0.1507 | 0.8677 | 0.1664 |
| zero obs + uniform actions | 0.5347 | 0.1465 | 0.8728 | 0.1844 |

Interpretation:

- Correct history is consistently closer to the true partner policy than random/noise histories, so the Q-trained VAE is not ignoring history.
- The gap is moderate rather than decisive on the pair20-relevant subset: true TV `0.1202` vs noise TV about `0.1465`-`0.1553`, and true acc `0.8957` vs noise acc about `0.8634`-`0.8728`.
- This supports the current diagnosis: Q-trained VAE uses history, but much of its apparent accuracy still comes from a strong query-conditioned Q-family prior rather than precise per-partner identification.
- No full eval for this diagnostic.

Follow-up direct on-policy accuracy diagnostic for Q-trained VAE:

- Purpose: user asked why Q-trained VAE can look close to the real partner policy offline but performs poorly when directly used as the ego controller.
- Script: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/diagnose_direct_onpolicy_accuracy.py`.
- Report: `reports/qtrained_direct_onpolicy_accuracy_pair20s20_20260707/direct_onpolicy_accuracy_summary.md`.
- Setup:
  - ego mode: `ttac_v5_8_latent_decoder_policy`, direct estimator policy replacement.
  - estimator: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`.
  - pair20-style diagnostic: `max_ego_policies=2`, `max_partner_policies=5`, partners `IQL/VDN/PQN-VDN`, both roles, `20` eval seeds.
  - Metrics compare estimator output `pi_hat(.|ego_obs, partner_history)` to the true Q partner policy on the same `ego_obs`; no blend/base is used in the metric.

Offline/base-like PPO-ego reference from policy geometry:

| setting | reward | KL | TV | argmax acc | top2 acc |
|---|---:|---:|---:|---:|---:|
| Q-trained VAE on PPO-ego Q val rows | n/a | 0.4432 | 0.1202 | 0.8957 | 0.9585 |

Direct-estimator on-policy states:

| partner pool | subset | reward | KL | TV | argmax acc | top2 acc | pred entropy | pred maxprob |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all-Q | all steps | 41.500 | 3.2600 | 0.5184 | 0.4909 | 0.6600 | 0.2624 | 0.9042 |
| all-Q | t>=50 | 41.500 | 3.3645 | 0.5275 | 0.4813 | 0.6506 | 0.2527 | 0.9079 |
| IQL | t>=50 | 10.300 | 3.8653 | 0.6024 | 0.4159 | 0.6082 | 0.2400 | 0.9154 |
| VDN | t>=50 | 20.750 | 3.9734 | 0.6124 | 0.3874 | 0.6048 | 0.2791 | 0.8936 |
| PQN-VDN | t>=50 | 93.450 | 2.2548 | 0.3677 | 0.6407 | 0.7387 | 0.2391 | 0.9149 |

Interpretation:

- Direct estimator rollout causes severe closed-loop distribution shift. Q-trained VAE is accurate on offline/base-like PPO ego states, but not on the states it creates when it directly controls ego.
- The collapse persists after history warmup (`t>=50`), so it is not only a first-50-steps missing-history issue.
- The estimator is confidently wrong on on-policy direct states (`pred maxprob ~=0.91`, entropy `~0.25`), which makes errors compound through the trajectory.
- This explains the direct-policy replacement failure: `90%` offline partner-policy argmax accuracy is not a `90%` closed-loop controller. Blend works better because it keeps base policy closed-loop stability and uses the estimator only as a bounded correction target.
- No full eval for this diagnostic.

## 2026-07-05 New 10-Seed Partner Variant 1-ZSC Follow-Up

After training `iql_dueling`, `vdn_dueling`, `iql_double`, and `a2c` as 10-seed / 10M partner pools, XP500 self/cross evaluation found:

| method | SP | XP |
|---|---:|---:|
| iql_dueling | 150.000 | 45.111 |
| vdn_dueling | 174.000 | 62.889 |
| iql_double | 162.000 | 89.333 |
| a2c | 0.000 | 0.000 |

`a2c` remains unusable and should be excluded from healthy partner pools. Using the three healthy new partner variants (`iql_dueling`, `vdn_dueling`, `iql_double`), query-attention qexp4 VAE direct blend was evaluated under full 1-ZSC / 100 eval seeds and compared against same-pool base:

| partner pool | base | VAE direct | delta |
|---|---:|---:|---:|
| all_q_partners | 63.886 | 71.065 | +7.179 |
| iql_dueling | 49.687 | 53.325 | +3.638 |
| vdn_dueling | 69.465 | 78.465 | +9.000 |
| iql_double | 72.505 | 81.405 | +8.900 |

VAE role split:

| partner pool | VAE ego_agent0 | VAE ego_agent1 |
|---|---:|---:|
| all_q_partners | 68.239 | 73.891 |
| iql_dueling | 50.228 | 56.422 |
| vdn_dueling | 77.426 | 79.504 |
| iql_double | 77.062 | 85.748 |

Reports:
- VAE: `reports/partner_variants_10seed_vae_only_1zsc_100seed_20260705/mixed_1zsc_summary.md`.
- Base: `reports/partner_variants_10seed_base_only_1zsc_100seed_20260705/mixed_1zsc_summary.md`.

Interpretation: VAE direct blend transfers well to the new healthy partner variants, improving same-pool base by `+7.179` aggregate 1-ZSC. The strongest gains are on `vdn_dueling` and `iql_double`; `iql_dueling` improves but remains harder. Same-pool old-v5.8 control was not completed in this follow-up.

## 2026-07-06 Classic MARL Partner Expansion

User requested trying classic non-ZSC MARL networks beyond methods already present in the repo. Implemented and smoke-tested three new partner families on the active A6000 server `ssh -p 26991 root@hz-4.matpool.com`:

| method | implementation | smoke train | loader check | note |
|---|---|---|---|---|
| QPLEX | `JaxMARL/baselines/QLearning/qplex_cnn_overcooked.py` | passed | passed | QPLEX-style duplex-dueling mixer; greedy per-agent Q execution |
| WQMIX | `JaxMARL/baselines/QLearning/wqmix_cnn_overcooked.py` | passed | passed | weighted TD objective with `WQMIX_ALPHA=0.1`; greedy per-agent Q execution |
| COMA | `JaxMARL/baselines/QLearning/coma_cnn_overcooked.py` | passed after JAX reshape fix | passed | decentralized CNN actor + centralized counterfactual critic |

Added remote runner and loader support:

- Train wrapper: `experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py`.
- Queue runner: `experiments/run_classic_marl_ov2_partners.sh`.
- Updated `evaluate_mixed_1zsc.py` and `evaluate_qlearning_object_sp_xp.py` to load `qplex`, `wqmix`, and `coma`.

Smoke roots and loader reports:

- QPLEX/WQMIX smoke: `runs/classic_marl_ov2_partners_classic_smoke_20260706_005808`.
- COMA smoke: `runs/classic_marl_ov2_partners_classic_smoke_coma_20260706_010014`.
- Loader checks: `reports/classic_smoke_qplex_loader_check`, `reports/classic_smoke_wqmix_loader_check`, `reports/classic_smoke_coma_loader_check`.

Formal 10seed/10M queue started:

- run root: `runs/classic_marl_ov2_partners_classic_marl_10seed_20260706_010208`.
- log dir: `logs/classic_marl_ov2_partners_classic_marl_10seed_20260706_010208`.
- settings: `METHODS=qplex,wqmix,coma`, `NUM_SEEDS=10`, `TOTAL_TIMESTEPS=10000000`, `RUN_SPXP=1`, `SPXP_EVAL_SEEDS=500`.
- observed status at launch: QPLEX split `0/10` running, GPU active.

Decision rule: do not add these methods to the healthy non-ZSC partner pool until SP/XP500 sanity is nonzero. If one or more families pass, evaluate base vs query-attn qexp4 VAE direct blend 1-ZSC on the healthy subset.

## 2026-07-08 Factorized Prior-Residual Q-Family VAE

Purpose: test whether the Q-family VAE is using a strong `query_obs` shortcut by structurally separating:

```text
query_obs -> average/query prior
partner history -> partner-specific residual
final logits = prior logits + residual logits
```

This is a diagnostic trained on Q-family data, not a clean strict 1-ZSC method.

Implementation:

- Added `factorized_prior_residual` support to `latent_partner_decoder.py` and `train_latent_partner_decoder.py`.
- For this decoder, no-history is forced to match the query-only prior by subtracting the zero-history latent baseline before the residual head.
- Added diagnostic script: `diagnose_prior_residual_decoder.py`.

Training:

- Dataset: `reports/qfamily_grouped_qcounter_dataset_20260707/qfamily_grouped_latent_decoder_dataset.npz`.
- Checkpoint: `reports/qfamily_factorized_prior_residual_vae_train_20260708/latent_partner_decoder.npz`.
- Main settings: `query_attn`, VAE, `z_dim=128`, `vae_beta=1e-4`, `decoder_z_norm`, grouped batch `64 x 14`, `steps=3000`, `prior_loss_coef=0.75`, `pairwise_tv_coef=0.2`, `residual_logit_coef=0.2`.

Grouped-val training summary:

| metric | value |
|---|---:|
| true KL | 0.4172 |
| true TV | 0.2305 |
| argmax acc | 0.8635 |
| top2 acc | 0.9464 |
| wrong KL gap | +0.1414 |
| random KL gap | +0.0846 |
| delayed KL gap | +0.0003 |
| same-query output/target TV corr | 0.7507 |
| embedding retrieval acc | 0.8948 |

Prior-residual diagnostic:

- Report: `reports/qfamily_factorized_prior_residual_vae_diag_20260708/prior_residual_summary.md`.
- No-history final exactly equals query prior: `no_history_final_vs_prior_tv = 0.0`.
- Query prior fits the mean teacher well: prior vs mean-teacher KL `0.0570`.
- Query prior alone is far from each partner target: prior vs target KL `1.2850`.
- True history improves target reconstruction to KL `0.4172`, TV `0.2305`, acc `0.8635`.
- True residual TV to prior is `0.4861`, so partner history carries a substantial residual.

Q-target val comparison against previous Q-family estimators:

| estimator | true KL | TV | argmax acc | top2 acc | wrong gap | random gap | delayed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q-trained VAE | 0.3539 | 0.1021 | 0.9160 | 0.9698 | +0.0039 | +0.0141 | +0.0009 |
| Q-counterfactual VAE | 0.3561 | 0.1621 | 0.8771 | 0.9552 | +0.0713 | +0.0161 | +0.0030 |
| factorized prior-residual | 0.3990 | 0.2187 | 0.8738 | 0.9435 | +0.1071 | +0.0521 | +0.0003 |

Interpretation: factorized prior-residual does the intended structural thing: it prevents no-history/query-only target prediction and makes partner history responsible for the partner-specific residual. However, this cleaner decomposition currently sacrifices reconstruction accuracy versus the Q-trained VAE.

Pair20 strict 1-ZSC reward diagnostic:

| mode | report | all-Q | IQL | VDN | PQN-VDN |
|---|---|---:|---:|---:|---:|
| same-run base | `reports/qfamily_factorized_prior_residual_pair20_1zsc_20260708` | 81.793 | 15.820 | 51.690 | 177.870 |
| factorized true history | `reports/qfamily_factorized_prior_residual_pair20_1zsc_20260708` | 83.740 | 13.700 | 53.190 | 184.330 |
| factorized no history | `reports/qfamily_factorized_prior_residual_nohist_pair20_1zsc_20260708` | 81.990 | 14.040 | 54.850 | 177.080 |

Reward interpretation:

- True history beats no-history by `+1.750` aggregate, mainly through PQN-VDN (`184.330` vs `177.080`), so the residual is not pure noise.
- VDN is better with no-history (`54.850` vs `53.190`) and IQL is below base for both true and no-history.
- The method improves over same-run base by only `+1.947`, far below old clean qexp4 VAE pair20 (`89.017`) and Q-trained VAE pair20 (`89.120`).
- Do not run full eval for this version. The useful result is mechanistic: query shortcut can be structurally removed, but the residual estimator needs higher accuracy and better closed-loop stability before it can replace the current VAE anchor.

## 2026-07-08 Simulated Online Latent-Z Adaptation

Purpose: test whether within-episode test-time learning can fit the current partner better than one-shot history encoding. This experiment uses the Q-trained VAE checkpoint as a contaminated diagnostic prior/decoder, not a clean strict method:

- checkpoint: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`
- online data boundary: each pair/role/episode resets independently and only uses current-episode observed `partner_obs, partner_action`.
- no partner checkpoint logits, no same-view oracle labels, no cross-partner or cross-episode data.
- warmup: first `50` environment steps use base/no blend while collecting partner behavior.
- ramp: alpha ramps for `30` or `50` steps after warmup.
- frozen components: PPO ego, `pi0`/query prior, decoder, residual basis.
- online variable: only current episode `z_p`.

Implementation:

- Added eval mode:
  - `ttac_v5_8_latent_decoder_blend_online_z`
  - plus wrong/random/delayed suffix compatibility.
- Added decoder helpers:
  - `encode_partner_online_z(...)`
  - `apply_latent_partner_decoder_with_z(...)`
- Added eval args:
  - `--ttac_latent_decoder_online_warmup_steps`
  - `--ttac_latent_decoder_online_ramp_steps`
  - `--ttac_latent_decoder_online_update_interval`
  - `--ttac_latent_decoder_online_update_steps`
  - `--ttac_latent_decoder_online_lr`
  - `--ttac_latent_decoder_online_prior_coef`
  - `--ttac_latent_decoder_online_min_history`
  - `--ttac_latent_decoder_online_z_clip`

Online objective:

```text
min_z mean_t[-log pi_decoder(a_partner_t | o_partner_t, z)]
      + prior_coef * ||z - encoder(current_history)||^2
```

Then the same `z` is used to query `pi_decoder(. | ego_obs, z)` for fixed direct logit blend.

Smoke:

- report: `reports/online_z_smoke_20260708`
- setting: `iql`, `1 ego x 1 partner x 2 roles x 1 seed`
- passed JIT/rollout.

Seed10 pair20 diagnostic, aggressive update:

- report: `reports/qtrained_vae_online_z_pair20_seed10_20260708`
- settings: warmup `50`, ramp `30`, update interval `10`, update steps `10`, lr `0.05`, prior coef `0.05`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.033 | 14.300 | 51.000 | 177.800 |
| Q-trained VAE blend | 86.967 | 16.700 | 52.900 | 191.300 |
| online-z aggressive | 85.967 | 14.800 | 58.000 | 185.100 |

Seed10 pair20 diagnostic, conservative update:

- report: `reports/qtrained_vae_online_z_conservative_pair20_seed10_20260708`
- settings: warmup `50`, ramp `50`, update interval `10`, update steps `5`, lr `0.01`, prior coef `0.5`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| online-z conservative true history | 85.700 | 15.700 | 56.900 | 184.500 |
| online-z conservative wrong history | 85.900 | 15.700 | 55.300 | 186.700 |

Interpretation:

- Online-z is implementable under the clean within-episode boundary and improves over same-run base in seed10.
- It does not beat ordinary Q-trained VAE blend; the best seed10 ordinary blend is `86.967`, while online-z is `85.967` aggressive and `85.700` conservative.
- VDN improves under online z, but PQN-VDN and/or IQL are hurt.
- Wrong-history conservative control slightly exceeds true-history (`85.900` vs `85.700`), so this version does not prove reward-useful partner-specific online adaptation.
- Do not run pair20 100-seed or full eval for this version.
- Next direction: if pursuing online learning, avoid pure partner-action CE on `z`; it likely pulls the latent toward behavior imitation under partner-view states without guaranteeing the same-view correction remains reward-useful.

## 2026-07-08 Meta-Trained Online-Z Diagnostic

Purpose: train the online CE update itself to be useful for same-view policy reconstruction. This is still a contaminated Q-family diagnostic, not a clean strict method.

Design:

```text
support S: partner_obs/action samples only
inner update: z_K = z_0 - eta * grad_z CE(a_partner | o_partner, z)
query Q: ego/same-view obs
outer target: pi_partner(. | query_obs)
outer loss: KL(target_partner_probs || decoder(query_obs, z_K))
```

Implementation:

- Added trainer: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/train_meta_online_z_decoder.py`.
- Added `--ttac_latent_decoder_online_init_z_mode {encoder,zero,no_history}` to mixed 1-ZSC eval.
- Updated online policy so test-time `z_p` can start from zero/no-history instead of the offline history encoder.

Initial encoder-init probes:

- `reports/meta_online_z_smoke_20260708`: first-order, lr `0.05`; `z_delta` only about `0.0008`, so the earlier online-z update was effectively not moving.
- `reports/meta_online_z_second_order_probe_20260708`: second-order, inner lr `5.0`; `z_delta` about `0.12`, but val post KL remained essentially equal to pre KL. Encoder-init makes it too easy to rely on the existing history encoder/prior.

Zero-init meta training:

- report/checkpoint: `reports/meta_online_z_zero_init_1k_20260708`
- init checkpoint: `reports/qfamily_diagnostic_qexp4_vae_train_20260707/latent_partner_decoder.npz`
- train data: `reports/qfamily_diagnostic_qtarget_dataset_20260707/qtarget_latent_decoder_dataset.npz`
- settings: second-order unroll, `steps=1000`, `batch_size=128`, `inner_steps=5`, `inner_lr=5.0`, `inner_prior_coef=0.0`, `init_z_mode=zero`, `pre_update_kl_coef=0.0`

Offline Q-target val:

| history | pre KL | post KL | post TV | acc | top2 |
|---|---:|---:|---:|---:|---:|
| true | 0.1730 | 0.1664 | 0.0828 | 0.9386 | 0.9897 |
| wrong | 0.1730 | 0.1681 | 0.0841 | 0.9384 | 0.9897 |
| random | 0.1730 | 0.1776 | 0.0888 | 0.9385 | 0.9897 |
| no_history | 0.1730 | 0.1744 | 0.0878 | 0.9374 | 0.9896 |

Interpretation: zero-init meta training finally makes CE-updated `z` improve same-view KL, and true history is best. The true-vs-wrong gap is small offline (`0.0016` KL), but random/no-history are worse.

Seed10 pair20 1-ZSC:

- report: `reports/meta_online_z_zero_init_pair20_seed10_20260708`
- settings: online warmup `50`, ramp `50`, update interval `10`, update steps `5`, online lr `5.0`, prior coef `0.0`, init z `zero`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| meta checkpoint ordinary blend | 80.567 | 17.200 | 44.500 | 180.000 |
| meta-online-z true | 84.300 | 18.800 | 56.900 | 177.200 |
| meta-online-z wrong | 77.167 | 16.700 | 48.600 | 166.200 |

Reward interpretation:

- This is the first online-z variant with a strong reward-side true/wrong separation: true beats wrong by `+7.133` aggregate.
- It improves over same-run base by `+3.300`, mainly via IQL and VDN.
- It does not beat the original Q-trained VAE ordinary seed10 control (`86.967`) or the formal Q-trained VAE pair20 anchor (`89.120`).
- The ordinary blend from the meta-trained checkpoint drops below base, meaning zero-init meta training reshapes the decoder for online inference and hurts the old encoder-based usage.
- Do not run 100-seed/full for this version. It is a useful mechanism result: meta-training the CE update can make online partner behavior matter, but the resulting controller is not yet competitive with the best offline VAE blend.

## 2026-07-08 Clean PG-Only Meta-Online-Z

Purpose: make the meta-online-z idea clean under the strict 1-ZSC protocol. The decoder/prior is initialized from the clean old qexp4 VAE and trained only on PPO/MAPPO data; IQL/VDN/PQN-VDN are used only for final pair20 evaluation.

Implementation note:

- Added `--eval_dataset` to `train_meta_online_z_decoder.py`, so training can use PPO/MAPPO run0..7 while offline sanity uses heldout PPO/MAPPO run8..9.

Training:

- checkpoint/report: `reports/meta_online_z_clean_pgonly_zero_init_1k_20260708`
- init checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`
- train data: `reports/lpr_pgonly_train_dataset_light_20260706/pg_mixed_latent_decoder_dataset.npz`
- heldout PG eval data: `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`
- settings: second-order, `steps=1000`, `batch_size=128`, `inner_steps=5`, `inner_lr=5.0`, `inner_prior_coef=0.0`, `init_z_mode=zero`, `pre_update_kl_coef=0.0`

Heldout PG offline:

| history | pre KL | post KL | post TV | acc | top2 |
|---|---:|---:|---:|---:|---:|
| true | 0.4318 | 0.4268 | 0.3411 | 0.6516 | 0.8419 |
| wrong | 0.4318 | 0.4301 | 0.3425 | 0.5905 | 0.8415 |
| random | 0.4318 | 0.4302 | 0.3432 | 0.6516 | 0.8419 |
| no-history | 0.4318 | 0.4335 | 0.3435 | 0.5825 | 0.8419 |

Interpretation: clean PG-only training learns a small history effect (`true post KL` beats wrong by about `0.0033` and no-history by about `0.0067`), but the offline signal is weak.

Pair20 strict 1-ZSC seed10:

- report: `reports/meta_online_z_clean_pgonly_pair20_seed10_20260708`
- eval partners: IQL/VDN/PQN-VDN only.
- settings: warmup `50`, ramp `50`, update interval `10`, update steps `5`, online lr `5.0`, prior coef `0.0`, init z `zero`, blend alpha `0.5`, clip `2.0`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| clean meta checkpoint ordinary blend | 85.533 | 15.600 | 57.400 | 183.600 |
| clean meta-online-z true | 85.767 | 15.400 | 58.300 | 183.600 |
| clean meta-online-z wrong | 79.400 | 16.200 | 49.700 | 172.300 |
| clean meta-online-z random | 80.267 | 13.500 | 48.900 | 178.400 |

Interpretation:

- This is clean with respect to strict 1-ZSC: no Q-family trajectory/logit/action-label data enters training.
- True online-z beats wrong by `+6.367` and random by `+5.500` aggregate, so the reward-side history control is positive in the clean version.
- It improves over same-run base by `+4.767`, but only beats ordinary blend by `+0.233`.
- It does not beat the formal old clean qexp4 VAE pair20 anchor (`89.017`) or the Q-trained diagnostic VAE (`89.120`), so do not run full eval.
- Takeaway: clean meta-online-z supports the mechanism that online partner behavior can matter, but the current clean PG-only training signal is too weak to become the main method.

## 2026-07-08 Old Clean qexp4 VAE Online-Z Ablation

Purpose: isolate whether online-z itself helps the strongest clean checkpoint. This run uses the original clean qexp4 VAE checkpoint without meta-online-z finetuning, on the same pair20 seed10 strict 1-ZSC protocol.

- checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`
- report: `reports/old_clean_qexp4_vae_onlinez_pair20_seed10_20260708`
- eval partners: IQL/VDN/PQN-VDN only.
- settings: warmup `50`, ramp `50`, update interval `10`, update steps `5`, online lr `5.0`, prior coef `0.0`, init z `zero`, blend alpha `0.5`, clip `2.0`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| old clean qexp4 VAE ordinary blend | 90.267 | 16.800 | 59.400 | 194.600 |
| old clean qexp4 VAE online-z true | 87.133 | 18.000 | 57.400 | 186.000 |
| old clean qexp4 VAE online-z wrong | 82.467 | 16.800 | 53.400 | 177.200 |
| old clean qexp4 VAE online-z random | 83.333 | 16.000 | 53.100 | 180.900 |

Interpretation:

- Old clean VAE ordinary blend remains the strongest clean seed10 result (`90.267`), consistent with the formal pair20 100-seed anchor (`89.017`).
- Online-z true is history-sensitive: true beats wrong by `+4.667` and random by `+3.800`.
- But online-z true is `-3.133` below ordinary blend. The current online CE update damages the strong old encoder/prior usage even with correct history.
- Clean meta-online-z improved online true-vs-wrong separation but reduced ordinary blend performance; old clean VAE preserves strong ordinary performance but does not benefit from this online-z update.
- Takeaway: online-z as currently implemented is not a drop-in improvement for the best clean checkpoint. The next online direction should avoid overwriting the strong ordinary latent path, or should gate online correction only when it demonstrably improves confidence/trajectory likelihood.

## 2026-07-08 Old Clean qexp4 VAE + LoRA Residual v1

Purpose: test a safer partner-conditioned parameter modulation idea. Instead of replacing the old VAE latent path or doing zero-init online-z, freeze the old clean qexp4 VAE and train only a small `z_p`-conditioned LoRA residual on top of old decoder logits.

Implementation:

- `latent_partner_decoder.py`: added `latent_lora_residual_enabled` branch.
- `train_lora_residual_decoder.py`: new trainer that freezes all old VAE parameters and updates only `lora_*`.
- LoRA form:

```text
q = CNN(query_obs)
z = old_VAE_encoder(partner_history)
base_logits = old_VAE_decoder(q, z)
delta_logits = gate(q,z) * (((LN(q) @ B) * c(z)) @ A)
final_logits = base_logits + clip(delta_logits, -2, 2)
```

Training:

- checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_train_20260708`
- init checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`
- train data: `reports/latent_decoder_pg2q_full_memsafe_20260701_160807/collect_pg_mixed/pg_mixed_latent_decoder_dataset.npz`
- clean: no IQL/VDN/PQN-VDN data used for training.
- settings: rank `8`, coeff hidden `64`, gate bias `-2.0`, anchor KL coef `0.05`, residual L2 coef `0.001`, steps `1500`, lr `1e-3`.

Offline val:

| history | KL | old/base KL | TV | acc | top2 | residual TV | gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 0.1459 | 0.1565 | 0.1652 | 0.8000 | 0.9147 | 0.0451 | 0.7718 |
| wrong | 0.1646 | 0.1735 | 0.1786 | 0.7921 | 0.9073 | 0.0473 | 0.7713 |
| delayed | 0.1462 | 0.1569 | 0.1653 | 0.8009 | 0.9148 | 0.0452 | 0.7723 |
| random | 0.1644 | 0.1679 | 0.1804 | 0.7931 | 0.9070 | 0.0469 | 0.7818 |
| no-history | 0.2241 | 0.2320 | 0.2211 | 0.7625 | 0.8842 | 0.0575 | 0.7647 |

Interpretation: LoRA improves offline policy reconstruction over the frozen old VAE, especially true KL (`0.1565 -> 0.1459`), but the learned gate is high (`~0.77`) and residual TV is nontrivial (`~0.045`), so this v1 is not as conservative as intended.

Pair20 strict 1-ZSC seed10:

- report: `reports/old_clean_qexp4_vae_lora_residual_r8_pair20_seed10_20260708`
- eval partners: IQL/VDN/PQN-VDN only.
- direct blend alpha `0.5`, clip `2.0`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| LoRA residual true | 89.300 | 17.300 | 57.000 | 193.600 |
| LoRA residual wrong | 88.800 | 15.400 | 55.200 | 195.800 |
| LoRA residual random | 87.600 | 15.200 | 53.200 | 194.400 |
| old clean VAE ordinary reference | 90.267 | 16.800 | 59.400 | 194.600 |

Interpretation:

- LoRA v1 improves over base by `+8.300`, but does not beat old clean VAE ordinary reference in the same seed10 setting (`89.300` vs `90.267`).
- True history beats random by `+1.700`, but wrong is very close (`+0.500` gap), so reward-side partner specificity is weak.
- Offline KL improvement does not translate into reward improvement; likely the residual changes many states softly without improving closed-loop coordination.
- Do not run full eval for this v1.
- Next LoRA direction: make it more conservative and local: lower gate bias or freeze gate, increase anchor KL, add low-disagreement residual penalty, or train LoRA only on high-value/high-error local states.

## 2026-07-08 Old Clean qexp4 VAE + LoRA Residual v2 Frozen-Gate Hard Queries

Purpose: retry LoRA residual with a more conservative/local design after v1 improved offline KL but did not improve reward over old clean qexp4 VAE.

Server:

- A6000 server `ssh -p 26991 root@hz-4.matpool.com` is available.
- GPU check: NVIDIA RTX A6000, 49140 MiB, near-idle at launch.
- A40 replacement `ssh -p 29898 root@hz-t3.matpool.com` was previously refusing connections, so this run used the A6000 server.

Implementation:

- Same frozen old clean qexp4 VAE base as v1.
- Train only `lora_*` parameters.
- New trainer flags:
  - `--freeze_gate`: keep gate fixed at `sigmoid(gate_bias)`.
  - `--hard_sample_fraction 0.5 --hard_top_fraction 0.3`: sample half the training batch from frozen-old-VAE high-KL rows.

Training:

- checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_frozengate_hard_train_20260708`
- init checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`
- train data: `reports/latent_decoder_pg2q_full_memsafe_20260701_160807/collect_pg_mixed/pg_mixed_latent_decoder_dataset.npz`
- clean: no IQL/VDN/PQN-VDN data used for training.
- settings: rank `8`, coeff hidden `64`, gate bias `-2.5`, freeze gate, anchor KL coef `0.2`, residual L2 coef `0.01`, residual clip `2.0`, steps `1500`, lr `1e-3`.

Offline val:

| history | KL | old/base KL | TV | acc | top2 | residual TV | gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 0.1517 | 0.1565 | 0.1655 | 0.7985 | 0.9135 | 0.0139 | 0.0759 |
| wrong | 0.1690 | 0.1735 | 0.1780 | 0.7907 | 0.9067 | 0.0140 | 0.0759 |
| delayed | 0.1521 | 0.1569 | 0.1656 | 0.7985 | 0.9134 | 0.0139 | 0.0759 |
| random | 0.1648 | 0.1679 | 0.1778 | 0.7920 | 0.9067 | 0.0145 | 0.0759 |
| no-history | 0.2308 | 0.2320 | 0.2205 | 0.7599 | 0.8866 | 0.0148 | 0.0759 |

Interpretation: frozen-gate hard-query LoRA gives a small clean offline improvement over the old VAE (`true KL 0.1565 -> 0.1517`) while keeping residual TV small (`~0.014`) and gate fixed (`~0.076`). It is much more conservative than v1, but the offline gain is also smaller.

Pair20 strict 1-ZSC seed10:

- report: `reports/old_clean_qexp4_vae_lora_residual_r8_frozengate_hard_pair20_seed10_20260708`
- eval partners: IQL/VDN/PQN-VDN only.
- direct blend alpha `0.5`, clip `2.0`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| LoRA v2 true | 89.500 | 18.000 | 55.600 | 194.900 |
| LoRA v2 wrong | 89.633 | 16.300 | 56.700 | 195.900 |
| LoRA v2 random | 87.000 | 14.700 | 52.600 | 193.700 |
| old clean VAE ordinary reference | 90.267 | 16.800 | 59.400 | 194.600 |

Interpretation:

- LoRA v2 improves over base by `+8.500`, but still does not beat old clean VAE ordinary reference (`89.500` vs `90.267`).
- True beats random by `+2.500`, but wrong is slightly higher than true (`89.633` vs `89.500`), so reward-side partner specificity is not reliable.
- Compared with v1, v2 is more conservative and has better IQL (`18.000` vs `17.300`) and aggregate (`89.500` vs `89.300`), but lower VDN and no decisive reward improvement.
- Do not run full eval for this v2. The result supports: offline local reconstruction can be improved, but direct-blend reward remains dominated by query-prior/base interaction unless history-specific corrections affect the decisive closed-loop states.

## 2026-07-08 Old Clean qexp4 VAE + Standard No-Gate LoRA Residual

Purpose: test the user's concern that learned/fixed gates may be unnecessary and may interfere with LoRA. This run removes the learned gate and uses a more standard LoRA residual:

```text
final_logits = old_VAE_logits + clip((alpha / rank) * LoRA(query_feature, z_p), -2, 2)
```

Implementation:

- `latent_partner_decoder.py`: added optional `lora_no_gate`; when present, `lora_gate=1`.
- `train_lora_residual_decoder.py`: added `--no_gate` and `--lora_alpha`.
- Existing v1/v2 checkpoints remain compatible because they do not contain `lora_no_gate`; they still use the old gate branch.

Training:

- checkpoint/report: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708`
- init checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`
- train data: `reports/latent_decoder_pg2q_full_memsafe_20260701_160807/collect_pg_mixed/pg_mixed_latent_decoder_dataset.npz`
- clean: no IQL/VDN/PQN-VDN data used for training.
- settings: rank `8`, alpha `8`, no gate, anchor KL coef `0.2`, residual L2 coef `0.01`, residual clip `2.0`, hard sample fraction `0.5`, hard top fraction `0.3`, steps `1500`, lr `1e-3`.

Offline val:

| history | KL | old/base KL | TV | acc | top2 | residual TV | gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 0.1475 | 0.1565 | 0.1688 | 0.7986 | 0.9107 | 0.0614 | 1.0000 |
| wrong | 0.1669 | 0.1735 | 0.1822 | 0.7911 | 0.9044 | 0.0632 | 1.0000 |
| delayed | 0.1477 | 0.1569 | 0.1690 | 0.7983 | 0.9109 | 0.0615 | 1.0000 |
| random | 0.1678 | 0.1679 | 0.1846 | 0.7904 | 0.9029 | 0.0623 | 1.0000 |
| no-history | 0.2394 | 0.2320 | 0.2253 | 0.7510 | 0.8804 | 0.0569 | 1.0000 |

Interpretation:

- No-gate LoRA does learn its own nontrivial correction amplitude: residual TV `~0.061`, larger than v2 (`~0.014`) and comparable to/slightly stronger than v1 (`~0.045`).
- Offline true KL improves over the frozen old VAE (`0.1565 -> 0.1475`) and true is clearly better than wrong/random.
- However, no-history becomes worse than the frozen old VAE (`0.2394` vs `0.2320`), showing the residual can damage the query prior when history is uninformative.
- Offline true KL is slightly worse than v1 (`0.1475` vs `0.1459`), but the architecture is cleaner.

Pair20 strict 1-ZSC seed10:

- report: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_pair20_seed10_20260708`
- eval partners: IQL/VDN/PQN-VDN only.
- direct blend alpha `0.5`, clip `2.0`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| same-run base | 81.000 | 14.200 | 51.000 | 177.800 |
| no-gate LoRA true | 89.167 | 16.200 | 57.500 | 193.800 |
| no-gate LoRA wrong | 90.667 | 20.500 | 56.700 | 194.800 |
| no-gate LoRA random | 86.833 | 15.100 | 51.500 | 193.900 |
| old clean VAE ordinary reference | 90.267 | 16.800 | 59.400 | 194.600 |

Interpretation:

- Removing the gate does not solve the reward issue. True history improves over base by `+8.167`, but remains below old clean VAE reference (`89.167` vs `90.267`).
- Wrong history is unexpectedly higher than true (`90.667` vs `89.167`), mostly from IQL (`20.500` vs `16.200`).
- Random history is lower (`86.833`), so the model is not completely history-insensitive, but the reward optimum is not aligned with correct-history policy reconstruction.
- Do not run full eval. This result strengthens the diagnosis: the bottleneck is not the learned gate itself. LoRA can learn a stronger static correction, but direct-blend reward still rewards some non-true-history perturbations and remains dominated by closed-loop dynamics rather than offline KL.

## 2026-07-08 Online LoRA Trajectory-NLL Pilot

Purpose: test the user's intended version of LoRA: not offline-only LoRA, but test-time LoRA adaptation using the current partner's observed trajectory. Since true partner policy logits are unavailable at test time, the online objective uses partner behavior likelihood under `partner_obs`.

Implementation:

- `policy.py`:
  - added eval modes:
    - `ttac_v5_8_latent_decoder_blend_online_lora`
    - `_wrong_history`
    - `_random_history`
    - `_delayed_history`
  - added `online_decoder_params` to `TTACPolicyState`; reset to the initial decoder checkpoint at episode end.
  - in online-LoRA mode, target policy uses `hstate.online_decoder_params`.
  - online update only updates trainable `lora_*` parameters; old VAE / base decoder parameters are frozen.
  - update data is the current episode's observed partner trajectory buffer `(partner_obs, partner_action)`.
  - update objective:

```text
support trajectory NLL:
  - mean log pi_online(a_partner | partner_obs)
+ anchor KL:
  KL(pi_init(.|partner_obs) || pi_online(.|partner_obs))
+ LoRA L2-to-init
```

- validation gate:
  - split the recent trajectory by time.
  - support = older valid trajectory steps.
  - validation = latest `val_steps`.
  - accept the LoRA update only if validation trajectory NLL decreases.

Parameters in pilot:

- checkpoint: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708/latent_partner_decoder.npz`
- report: `reports/online_lora_nogate_alpha8_pair20_seed5_20260708`
- eval: pair20 strict 1-ZSC, seed5, partners IQL/VDN/PQN-VDN.
- warmup `50`, ramp `30`, update interval `10`, min history `40`.
- LoRA lr `1e-3`, update steps `3`, anchor KL coef `0.05`, L2 coef `0.001`, validation steps `10`, accept margin `0.0`.

Results:

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.533 | 14.600 | 52.400 | 177.600 |
| no-update no-gate LoRA | 89.267 | 16.800 | 56.200 | 194.800 |
| online-LoRA true | 86.200 | 16.400 | 56.600 | 185.600 |
| online-LoRA wrong | 86.267 | 17.000 | 56.000 | 185.800 |
| online-LoRA random | 84.733 | 16.400 | 54.400 | 183.400 |

Interpretation:

- Engineering path works: online LoRA can be updated inside 1-ZSC rollout from current partner trajectory without accessing partner checkpoint logits.
- Current update is harmful relative to no-update no-gate LoRA (`86.200` vs `89.267`).
- true and wrong are nearly identical (`86.200` vs `86.267`), while random is lower (`84.733`). So the update is not pure noise, but it is not reliably partner-specific in the reward sense.
- Main damage is PQN-VDN (`194.800` no-update -> `185.600` online true), suggesting behavior-likelihood updates on partner_obs can overfit/deform the estimator in ways that do not transfer to ego_obs closed-loop coordination.
- Do not run full eval or seed10 yet. Next direction should either:
  - reduce online update strength (`lr`, steps, stronger anchor), or
  - update a smaller partner-specific code/coefficient instead of full LoRA, or
  - add diagnostics for accepted update count and validation NLL change before further reward sweeps.

## 2026-07-08 Meta-learned LoRA online update pilot

Purpose: test whether LoRA is useful as a test-time adaptation channel only when its update directions are meta-trained for trajectory-likelihood adaptation, rather than learned as a static offline residual.

Implementation/training:

- Trainer: `experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/train_meta_lora_residual_decoder.py`.
- Checkpoint/report: `reports/meta_lora_from_nogate_r8_inner3_lr002_train_20260708`.
- Init checkpoint: `reports/old_clean_qexp4_vae_lora_residual_r8_nogate_alpha8_hard_train_20260708/latent_partner_decoder.npz`.
- Reused existing no-gate LoRA parameters via `--reuse_lora_if_present`.
- Inner objective uses current partner trajectory likelihood under `partner_obs`:

```text
inner loss =
  - log pi_online(a_partner | partner_obs)
+ anchor KL to current meta params
+ LoRA L2
```

- Outer objective optimizes same-view target reconstruction after the inner update:

```text
outer loss =
  KL(pi_partner(. | query_obs) || pi_online_after_update(. | query_obs))
+ anchor/residual regularization
```

- Settings: rank `8`, alpha `8`, no gate, first-order MAML, inner lr `0.02`, inner steps `3`, inner anchor KL `0.05`, inner L2 `0.001`, outer anchor KL `0.1`, outer residual L2 `0.01`, steps `1000`.

Offline heldout metrics:

| history | pre KL | post KL | delta KL | pre acc | post acc | residual TV |
|---|---:|---:|---:|---:|---:|---:|
| true | 0.1463 | 0.1462 | -0.0000 | 0.7975 | 0.7977 | 0.0097 |
| wrong | 0.1660 | 0.1681 | +0.0021 | 0.7913 | 0.7900 | 0.0150 |
| delayed | 0.1465 | 0.1467 | +0.0003 | 0.7975 | 0.7983 | 0.0128 |
| random | 0.1661 | 0.1688 | +0.0027 | 0.7923 | 0.7905 | 0.0134 |
| no-history | 0.2333 | 0.2412 | +0.0079 | 0.7562 | 0.7493 | 0.0419 |

Reward pilot:

- Report: `reports/meta_lora_from_nogate_inner3_lr002_online_pair20_seed5_20260708`.
- Eval: pair20 strict 1-ZSC seed5; partners IQL/VDN/PQN-VDN; direct blend alpha `0.5`, clip `2.0`.
- Online eval settings: warmup `50`, ramp `30`, update interval `10`, min history `40`, LoRA lr `0.02`, LoRA update steps `3`, anchor KL `0.05`, L2 `0.001`, validation steps `10`.

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 81.533 | 14.600 | 52.400 | 177.600 |
| meta-LoRA no-update | 89.733 | 16.800 | 57.600 | 194.800 |
| meta-online-LoRA true | 86.333 | 17.200 | 56.200 | 185.600 |
| meta-online-LoRA wrong | 83.467 | 14.800 | 52.000 | 183.600 |
| meta-online-LoRA random | 84.733 | 16.200 | 54.400 | 183.600 |

Interpretation:

- Meta training creates some correct-history separation: true online (`86.333`) is higher than wrong (`83.467`) and random (`84.733`).
- But the true online update is still much worse than no-update (`86.333` vs `89.733`), so LoRA online adaptation is not yet a usable improvement over the static old-VAE/no-gate prior.
- Offline post-update KL barely improves for true (`0.146251 -> 0.146245`), while wrong/random/no-history get worse. This matches the reward pattern: the update can reject bad histories, but it does not make correct history materially better.
- The main reward damage again comes from PQN-VDN (`194.800` no-update -> `185.600` true online). This suggests that partner_obs trajectory NLL is still too weak/misaligned as the only online objective for same-view closed-loop control.
- Do not run full eval. Current conclusion: LoRA is only worth keeping if we meta-learn an update direction that improves correct-history same-view prediction by a meaningful margin; this first-order LoRA version mostly learns a conservative/no-op update with a bad-history rejection effect.

## 2026-07-08 Old-VAE Prior + Online Residual-C Prototype

Purpose: test the cleaner version of the online latent route:

```text
fixed old VAE decoder:
  pi_old(. | query_obs, z0)

online inference:
  c_t = argmin_c NLL(partner_action | partner_obs, z0, c)
        + prior regularization

same-view prediction:
  logits = logits_old(query_obs, z0) + residual_phi(query_obs, z0, c_t)
```

Key difference from earlier online-z:

- Earlier online-z replaced/redirected the whole latent used by the decoder.
- This prototype freezes the old clean qexp4 VAE path and only adds a low-dimensional residual code `c`.
- The checkpoint stores explicit latent as `[z0, c]`; `z0` is produced by the old VAE encoder and kept fixed during online updates, while `c` is optimized from trajectory NLL.

Implementation:

- `latent_partner_decoder.py`:
  - added `latent_online_residual_c_enabled` checkpoint marker.
  - `apply_latent_partner_decoder_with_z` now supports `[z0, c]`.
  - output is old VAE logits plus a query-conditioned residual basis multiplied by `c`.
- `policy.py`:
  - existing `ttac_v5_8_latent_decoder_blend_online_z` path now detects residual-C checkpoints.
  - for residual-C checkpoints, online update keeps `z0` fixed and updates only `c`.
  - added optional online smooth coefficient.
- `evaluate_mixed_1zsc.py`:
  - infers latent dim as `dim(z0) + dim(c)`.
  - added `--ttac_latent_decoder_online_smooth_coef`.
- New trainer:
  - `train_meta_residual_c_decoder.py`.
  - freezes old VAE parameters.
  - trains only residual-C basis by second-order inner/outer objective.

Training:

- checkpoint/report: `reports/meta_residual_c_pgonly_r8_h128_1k_20260708`.
- init: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- train data: `reports/lpr_pgonly_train_dataset_light_20260706/pg_mixed_latent_decoder_dataset.npz`.
- heldout data: `reports/accuracy_first_pg_heldout_dataset_20260706/pg_mixed_latent_decoder_dataset.npz`.
- settings: `c_dim=8`, hidden `128`, second-order, steps `1000`, batch `128`, inner steps `5`, inner lr `5.0`, inner prior coef `0.01`, residual L2 coef `0.001`.

Heldout PG offline:

| history | pre KL | post KL | delta KL | acc | top2 | c norm | residual TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 0.4495 | 0.4463 | -0.0031 | 0.7302 | 0.8502 | 0.0216 | 0.0134 |
| wrong | 0.4279 | 0.4229 | -0.0050 | 0.7598 | 0.8337 | 0.0227 | 0.0167 |
| random | 0.6553 | 0.6708 | +0.0155 | 0.6309 | 0.7972 | 0.0248 | 0.0156 |
| no-history | 0.4603 | 0.4469 | -0.0134 | 0.6195 | 0.8030 | 0.0252 | 0.0143 |

Rollout smoke:

- report: `reports/meta_residual_c_pgonly_r8_h128_online_smoke_20260708`.
- setting: IQL, 1 ego seed x 1 partner seed x both roles x 1 eval seed.
- passed JIT/rollout with residual-C checkpoint.

Interpretation:

- The implementation works, and the residual code actually moves: training batches reached visible post-vs-pre KL reductions and residual TV around `0.05-0.07`.
- Heldout true KL improves only slightly (`-0.0031`), below the pre-set `0.005-0.01` offline gate.
- More importantly, wrong history also improves (`-0.0050`) and no-history improves even more (`-0.0134`). This means the learned residual is not yet a clean partner-specific trajectory correction; it partially acts as another generic corrective prior.
- Random history gets worse, so the method is not completely history-insensitive, but the control pattern is not good enough for reward evaluation.
- Do not run pair20/full for this checkpoint. Next residual-C attempt should add explicit negative/no-history controls in training or constrain residual-C to improve true while preserving no-history/wrong, otherwise it repeats the old problem in a smaller latent space.

## 2026-07-09 Old VAE as a Plug-in on MEP / TrajeDi Ego Policies

Question: old clean qexp4 VAE is an output-space plug-in adapter. Does it still add value when the ego policy is already a stronger ZSC/diverse-policy method such as MEP or TrajeDi?

Compatibility note:

- The staged MEP/TrajeDi ego checkpoints in `reports/talents_style_staged_pools_20260702_190524` use an older PPO CNN parameterization:
  - `CNNSimple_0`, `LayerNorm_0`, `Dense_0..3`.
- Current v5.8 PPO wrapper expected:
  - `feature_norm`, `base_actor_*`, `ttac_adapter_*`.
- Added a narrow legacy compatibility path:
  - `models/cnn.py`: `TTAC_LEGACY_CNN` branch uses old CNN head names and exposes `base_logits` for direct blend.
  - `evaluate_mixed_1zsc.py`: auto-detects legacy params and sets `TTAC_LEGACY_CNN=True`.

Experiment:

- MEP ego run dir: `reports/talents_style_staged_pools_20260702_190524/mep_pop_run0`.
- TrajeDi ego run dir: `reports/talents_style_staged_pools_20260702_190524/trajedi_pop_run0`.
- Adapter checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Partners: IQL / VDN / PQN-VDN only.
- Setting: `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`, alpha `0.5`, clip `2.0`.
- Reports:
  - `reports/oldvae_plugin_mep_ego_q_pair20_seed10_20260709`
  - `reports/oldvae_plugin_trajedi_ego_q_pair20_seed10_20260709`

MEP ego:

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 28.900 | 20.300 | 36.000 | 30.400 |
| old VAE true | 36.033 | 18.500 | 40.700 | 48.900 |
| old VAE wrong | 35.267 | 19.300 | 38.800 | 47.700 |
| old VAE random | 34.967 | 18.400 | 37.300 | 49.200 |

TrajeDi ego:

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 29.400 | 37.500 | 28.400 | 22.300 |
| old VAE true | 40.300 | 29.800 | 35.600 | 55.500 |
| old VAE wrong | 40.067 | 30.900 | 33.400 | 55.900 |
| old VAE random | 40.300 | 33.700 | 32.300 | 54.900 |

Interpretation:

- Old VAE is additive on these staged ZSC ego checkpoints in seed10 pair20:
  - MEP: `+7.133` aggregate over base.
  - TrajeDi: `+10.900` aggregate over base.
- But true-history specificity is weak:
  - MEP true is only `+0.767` over wrong and `+1.067` over random.
  - TrajeDi true is only `+0.233` over wrong and exactly tied with random on aggregate.
- The gain is mostly a plug-in cooperative prior / output-space correction, not evidence of strong partner-specific modeling.
- Family-level behavior is mixed:
  - MEP: IQL drops (`20.300 -> 18.500`), VDN improves, PQN-VDN improves strongly.
  - TrajeDi: IQL drops (`37.500 -> 29.800`), VDN improves, PQN-VDN improves strongly.
- Do not claim this proves estimator history modeling. It supports the narrower claim that old VAE direct blend can be a plug-in adapter on top of existing ZSC/diverse ego policies, at least in this light evaluation.
- A seed100 run is only worth doing if we want to make the plug-in/additive claim stronger; it is not necessary for the partner-specific modeling claim because controls already show weak true-history dependence.

Correction: the above MEP/TrajeDi results used staged population-member checkpoints under `reports/talents_style_staged_pools_20260702_190524`, not final ZSC ego checkpoints. They should not be cited as the answer to "does old VAE improve ZSC ego?" They only show behavior on staged population members.

## 2026-07-09 Correction: Old qexp4 VAE Plug-in on True MEP / TrajeDi ZSC Ego

Question: user clarified that the intended ego should be the final ZSC ego, not a population member. Re-ran the light pair20 strict Q-family evaluation on `run_X/ego/ckpt_final`.

Implementation note:

- Added `--ego_checkpoint_subdir ego` to `evaluate_mixed_1zsc.py` so final ZSC checkpoints nested as `run_X/ego/ckpt_final` are loaded explicitly.
- Internal eval hook remains `ttac_v5_8_latent_decoder_blend`, but the tested method is the old clean qexp4 VAE checkpoint plus direct blend:
  - `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Report names:
  - `reports/oldvae_plugin_mep_zsc_ego_q_pair20_seed10_trueonly_20260709`
  - `reports/oldvae_plugin_trajedi_zsc_ego_q_pair20_seed10_trueonly_20260709`

Setup:

- MEP true ZSC ego: `runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full`, with `--ego_checkpoint_subdir ego`.
- TrajeDi true ZSC ego: `runs/trajedi_realpop_mm1_mp1_K5_div0.05_64_16_10000000_seed10_20260516-151722/20260516-151735_1gd6vw78_counter_circuit_avs-full`, with `--ego_checkpoint_subdir ego`.
- Partners: IQL / VDN / PQN-VDN only.
- Setting: `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`, alpha `0.5`, clip `2.0`.
- Only tested base vs old qexp4 VAE direct blend; wrong/random controls were intentionally not run in this corrected pass.

MEP true ZSC ego:

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 61.600 | 21.300 | 52.300 | 111.200 |
| old qexp4 VAE direct blend | 78.733 | 32.200 | 56.300 | 147.700 |
| delta | +17.133 | +10.900 | +4.000 | +36.500 |

TrajeDi true ZSC ego:

| mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| base | 40.033 | 17.200 | 44.100 | 58.800 |
| old qexp4 VAE direct blend | 58.167 | 30.400 | 53.300 | 90.800 |
| delta | +18.133 | +13.200 | +9.200 | +32.000 |

Interpretation:

- This is the correct ZSC-ego answer: old qexp4 VAE direct blend is additive on true MEP and TrajeDi final ego checkpoints in pair20 seed10.
- The gain is large in both cases, especially on PQN-VDN, and also positive on IQL/VDN.
- However, these final MEP/TrajeDi ego checkpoints have surprisingly weak strict Q-family base scores relative to PPO state-aug baselines, so this does not show MEP/TrajeDi are stronger egos in our strict Q 1-ZSC setting.
- Since this corrected pass did not include wrong/random controls by user request, it supports the plug-in effectiveness claim, not a partner-specific history modeling claim.

## 2026-07-09 Non-State-Aug PPO / MEP / TrajeDi Base 1-ZSC

Question: compute base strict Q-family 1-ZSC for non-state-aug PPO, MEP, and TrajeDi ego policies.

Setup:

- Metric: pair20 seed10 strict Q-family 1-ZSC, base only.
- Partners: IQL / VDN / PQN-VDN.
- Setting: `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
- Reports:
  - PPO: `reports/base_ppo_no_state_1zsc_q_pair20_seed10_20260709`
  - MEP: `reports/base_mep_no_state_1zsc_q_pair20_seed10_20260709`
  - TrajeDi: `reports/base_trajedi_no_state_1zsc_q_pair20_seed10_20260709`

Checkpoint sources:

- PPO no-state: `runs/ttac_posthoc_zero_adapter_from_ppo_standard_no_state_aug_64_16_seed42_10seeds_no_state_20260622_133910`, direct `run_X/ckpt_final`.
- MEP no-state: `runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full`, loaded with `--ego_checkpoint_subdir ego`.
- TrajeDi no-state: `runs/_archive_legacy_not_main_20260515/superseded_trajedi_run_replaced_by_unified_rerun/trajedi_realpop_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260511-103234/20260511-103246_ggoqyh5r_counter_circuit_avs-full`, loaded with `--ego_checkpoint_subdir ego`.
- State-aug check: these selected checkpoints have no `STATE_AUG_*` keys in checkpoint config. CNN first conv is still `(5, 5, 30, 32)`, so channel count alone is not a reliable state-aug indicator in this codebase.
- Caveat: TrajeDi no-state source is archived/superseded by later state-aug reruns; use it only for the requested non-state-aug baseline, not as the current main TrajeDi result.

Results:

| ego | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| PPO no-state | 36.733 | 25.500 | 34.400 | 50.300 |
| MEP no-state | 50.200 | 44.700 | 46.500 | 59.400 |
| TrajeDi no-state | 39.433 | 28.000 | 37.400 | 52.900 |

Interpretation:

- In this strict Q-family pair20 setting, non-state-aug PPO is much weaker than PPO state-aug baselines previously used for our main 1-ZSC comparisons.
- MEP no-state is the strongest among these three, mainly because it improves IQL substantially over PPO/TrajeDi.
- TrajeDi no-state is only slightly above PPO no-state on aggregate in this light eval.
- This reinforces that state augmentation is a major confounder for comparing ZSC ego methods in our current experiment suite.

Additional GAMMA result:

- GAMMA source: `runs/gamma_mix025_ppo_standard_source_64_16_10000000_20260514-174603/20260514-174617_nc84elts_counter_circuit_avs-full`.
- Config markers: `GAMMA.VAE_CHECKPOINT = runs/gamma_full_20260513-090630/vae/gamma_vae.pkl`, `POPULATION_MIX_PROB = 0.25`, no `STATE_AUG_*` keys.
- Report: `reports/base_gamma_mix025_ppo_source_1zsc_q_pair20_seed10_20260709`.

| ego | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|
| GAMMA mix0.25 PPO-source | 50.867 | 24.600 | 48.700 | 79.300 |

Interpretation:

- GAMMA mix0.25 PPO-source is roughly tied with MEP no-state on aggregate (`50.867` vs `50.200`), but the profile is different.
- GAMMA is much worse than MEP on IQL (`24.600` vs `44.700`), similar/slightly better on VDN (`48.700` vs `46.500`), and much better on PQN-VDN (`79.300` vs `59.400`).
- This makes GAMMA look like a PQN-VDN-friendly baseline rather than uniformly stronger ZSC behavior in this pair20 strict Q setting.

## 2026-07-09 Old qexp4 VAE Plug-in on Non-State-Aug / GAMMA Egos

Question: test the old clean qexp4 VAE direct-blend plug-in on the previously untested non-state-aug PPO, MEP, TrajeDi, and GAMMA egos.

Setup:

- Metric: pair20 seed10 strict Q-family 1-ZSC.
- Partners: IQL / VDN / PQN-VDN.
- Setting: `max_ego_policies=2`, `max_partner_policies=5`, both roles, `num_eval_seeds=10`.
- Modes: `base_no_test_adapt` and old clean qexp4 VAE direct blend.
- Adapter checkpoint: `reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz`.
- Note: internal eval hook is `ttac_v5_8_latent_decoder_blend`, but the method should be described as old qexp4 VAE direct blend.
- Reports:
  - PPO no-state: `reports/oldvae_plugin_ppo_no_state_q_pair20_seed10_trueonly_20260709`
  - MEP no-state: `reports/oldvae_plugin_mep_no_state_q_pair20_seed10_trueonly_20260709`
  - TrajeDi no-state: `reports/oldvae_plugin_trajedi_no_state_q_pair20_seed10_trueonly_20260709`
  - GAMMA mix0.25 PPO-source: `reports/oldvae_plugin_gamma_mix025_ppo_source_q_pair20_seed10_trueonly_20260709`

Results:

| ego | base all-Q | old qexp4 VAE all-Q | delta | IQL delta | VDN delta | PQN-VDN delta |
|---|---:|---:|---:|---:|---:|---:|
| PPO no-state | 36.733 | 51.200 | +14.467 | +3.200 | +8.900 | +31.300 |
| MEP no-state | 50.200 | 62.533 | +12.333 | +0.400 | +6.400 | +30.200 |
| TrajeDi no-state | 39.433 | 51.100 | +11.667 | +3.500 | +7.100 | +24.400 |
| GAMMA mix0.25 PPO-source | 50.867 | 68.133 | +17.267 | +5.700 | +12.800 | +33.300 |

Full per-family values:

| ego | mode | all-Q | IQL | VDN | PQN-VDN |
|---|---:|---:|---:|---:|---:|
| PPO no-state | base | 36.733 | 25.500 | 34.400 | 50.300 |
| PPO no-state | old qexp4 VAE | 51.200 | 28.700 | 43.300 | 81.600 |
| MEP no-state | base | 50.200 | 44.700 | 46.500 | 59.400 |
| MEP no-state | old qexp4 VAE | 62.533 | 45.100 | 52.900 | 89.600 |
| TrajeDi no-state | base | 39.433 | 28.000 | 37.400 | 52.900 |
| TrajeDi no-state | old qexp4 VAE | 51.100 | 31.500 | 44.500 | 77.300 |
| GAMMA mix0.25 PPO-source | base | 50.867 | 24.600 | 48.700 | 79.300 |
| GAMMA mix0.25 PPO-source | old qexp4 VAE | 68.133 | 30.300 | 61.500 | 112.600 |

Interpretation:

- The old qexp4 VAE plug-in improves all four previously untested egos under the same pair20 strict Q setting.
- The largest aggregate lift is on GAMMA mix0.25 (`+17.267`), followed by PPO no-state (`+14.467`), MEP no-state (`+12.333`), and TrajeDi no-state (`+11.667`).
- The lift is again dominated by PQN-VDN, but VDN is consistently positive too.
- IQL gains are small for MEP no-state (`+0.400`) and modest for the others; this keeps the same pattern seen in earlier experiments: old VAE is not uniformly strong across Q-family partners, but it is a robust output-space cooperative correction.
