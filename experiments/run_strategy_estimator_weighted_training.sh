#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
DATASET="${DATASET:-$ROOT/reports/ttac_v5_3_temporal_estimator_formal_small_20260622_120809/dataset/agreement_dataset.npz}"
INIT_ESTIMATOR="${INIT_ESTIMATOR:-$ROOT/reports/ttac_v5_3_temporal_estimator_formal_small_20260622_120809/agreement_estimator_temporal/agreement_estimator.npz}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/reports/strategy_estimator_weighted_training_$STAMP}"

cd "$ROOT"

"$PYTHON" -m py_compile \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/train_strategy_estimator_weighted.py" \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/strategy_estimator_benchmark.py"

"$PYTHON" \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/train_strategy_estimator_weighted.py" \
  --dataset "$DATASET" \
  --run_dir "$RUN_DIR" \
  --init_from "$INIT_ESTIMATOR" \
  --output_dir "$OUT_DIR" \
  --seed 42 \
  --steps "${STEPS:-1500}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --lr "${LR:-0.0001}" \
  --high_tv_quantile "${HIGH_TV_QUANTILE:-0.75}" \
  --high_tv_weight "${HIGH_TV_WEIGHT:-4.0}" \
  --contrastive_coef "${CONTRASTIVE_COEF:-0.5}" \
  --contrastive_margin "${CONTRASTIVE_MARGIN:-0.03}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}" \
  --policy_prob_batch_size "${POLICY_PROB_BATCH_SIZE:-1024}"

"$PYTHON" \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/strategy_estimator_benchmark.py" \
  --dataset "$DATASET" \
  --estimator "$OUT_DIR/strategy_estimator_weighted.npz" \
  --run_dir "$RUN_DIR" \
  --output_dir "$OUT_DIR/post_train_benchmark" \
  --seed 42 \
  --max_samples 50000 \
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}" \
  --policy_prob_batch_size "${POLICY_PROB_BATCH_SIZE:-1024}"

echo "[weighted_estimator] OUT_DIR=$OUT_DIR"
