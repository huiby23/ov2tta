#!/usr/bin/env bash
set -euo pipefail

ROOT=/teams/ius_1663576043/hby/rl/ov2
PYTHON=/root/miniconda3/envs/myconda/bin/python
cd "$ROOT"

export PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RUN_DIR="runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="reports/ttac_target_effect_audit_${STAMP}"
LOG_DIR="logs/ttac_target_effect_audit_${STAMP}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

"$PYTHON" experiments/overcooked_v2_experiments/ttac_v2/utils/target_effect_audit.py \
  --run_dir "$RUN_DIR" \
  --output_dir "$OUT_DIR" \
  --layout counter_circuit \
  --seed 42 \
  --num_episodes "${NUM_EPISODES:-2}" \
  --prefix_steps "${PREFIX_STEPS:-80}" \
  --eval_suffix_start "${EVAL_SUFFIX_START:-80}" \
  --max_pairs "${MAX_PAIRS:-12}" \
  --compatibility_sample_limit "${COMPATIBILITY_SAMPLE_LIMIT:-128}" \
  --modes "${AUDIT_MODES:-base_no_update,agreement,advantage_weighted,projected_confident,kl_only,aw_shuffled_history}" \
  --ttac_adapter_scale "${TTAC_ADAPTER_SCALE:-0.5}" \
  --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF:-0.0}" \
  --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF:-0.01}" \
  --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF:-0.01}" \
  --ttac_test_lr "${TTAC_TEST_LR:-0.003}" \
  --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS:-3}" \
  --ttac_history_len "${TTAC_HISTORY_LEN:-50}" \
  --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA:-2.0}" \
  --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}" \
  --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}" \
  --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER:-1.0}" \
  > "$LOG_DIR/audit.log" 2>&1

echo "Output: $OUT_DIR"
tail -80 "$OUT_DIR/target_effect_summary.md"
