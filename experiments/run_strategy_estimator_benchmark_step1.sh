#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
DATASET="${DATASET:-$ROOT/reports/ttac_v5_3_temporal_estimator_formal_small_20260622_120809/dataset/agreement_dataset.npz}"
ESTIMATOR="${ESTIMATOR:-$ROOT/reports/ttac_v5_3_temporal_estimator_formal_small_20260622_120809/agreement_estimator_temporal/agreement_estimator.npz}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/reports/strategy_estimator_benchmark_step1_$STAMP}"

cd "$ROOT"

"$PYTHON" -m py_compile \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/strategy_estimator_benchmark.py"

"$PYTHON" \
  "$ROOT/experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/strategy_estimator_benchmark.py" \
  --dataset "$DATASET" \
  --estimator "$ESTIMATOR" \
  --run_dir "$RUN_DIR" \
  --output_dir "$OUT_DIR" \
  --seed 42 \
  --max_samples 50000 \
  --eval_batch_size 512 \
  --policy_prob_batch_size 1024

echo "[strategy_estimator] OUT_DIR=$OUT_DIR"
