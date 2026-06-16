#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v3_alignment_sweep_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v3_alignment_target_audit_${TS}}"
SEED="${SEED:-42}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA:-2.0}"
TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}"
TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER:-1.0}"
TTAC_TEST_CONTRAST_BETA="${TTAC_TEST_CONTRAST_BETA:-1.0}"
TTAC_TEST_CONTRAST_FLOOR="${TTAC_TEST_CONTRAST_FLOOR:-0.0}"
TTAC_TEST_SEMANTIC_LAMBDA="${TTAC_TEST_SEMANTIC_LAMBDA:-0.25}"
TTAC_V3_SEMANTIC_COEF="${TTAC_V3_SEMANTIC_COEF:-0.25}"
TTAC_V3_MARGIN="${TTAC_V3_MARGIN:-0.1}"
TTAC_V3_USE_CHANGE_GATE="${TTAC_V3_USE_CHANGE_GATE:-1}"
TTAC_V3_CHANGE_GATE_FLOOR="${TTAC_V3_CHANGE_GATE_FLOOR:-0.1}"
TTAC_V3_MARGIN_COEF="${TTAC_V3_MARGIN_COEF:-0.1}"
TTAC_V3_RECENCY_TAU="${TTAC_V3_RECENCY_TAU:-10.0}"

log "TARGET_AUDIT_START report=${REPORT_DIR} run=${RUN_DIR}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v3/utils/target_effect_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${REPORT_DIR}/target_effect" \
  --seed "${SEED}" \
  --max_pairs "${MAX_PAIRS:-12}" \
  --num_episodes "${NUM_EPISODES:-2}" \
  --prefix_steps "${PREFIX_STEPS:-80}" \
  --eval_suffix_start "${EVAL_SUFFIX_START:-80}" \
  --compatibility_sample_limit "${COMPATIBILITY_SAMPLE_LIMIT:-128}" \
  --modes base_no_update,v3_support_aw,v3_gated_semantic,v3_gated_margin,v3_margin_only \
  --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" \
  --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
  --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
  --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
  --ttac_test_lr "${TTAC_TEST_LR}" \
  --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
  --ttac_history_len "${TTAC_HISTORY_LEN}" \
  --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" \
  --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
  --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" \
  --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
  --ttac_test_contrast_beta "${TTAC_TEST_CONTRAST_BETA}" \
  --ttac_test_contrast_floor "${TTAC_TEST_CONTRAST_FLOOR}" \
  --ttac_test_semantic_lambda "${TTAC_TEST_SEMANTIC_LAMBDA}" \
  --ttac_v3_semantic_coef "${TTAC_V3_SEMANTIC_COEF}" \
  --ttac_v3_margin_coef "${TTAC_V3_MARGIN_COEF}" \
  --ttac_v3_margin "${TTAC_V3_MARGIN}" \
  --ttac_v3_recency_tau "${TTAC_V3_RECENCY_TAU}" \
  --ttac_v3_use_change_gate "${TTAC_V3_USE_CHANGE_GATE}" \
  --ttac_v3_change_gate_floor "${TTAC_V3_CHANGE_GATE_FLOOR}" \
  > "${LOG_DIR}/target_effect.log" 2>&1
cp "${REPORT_DIR}/target_effect/target_effect_summary.csv" "${REPORT_DIR}/target_effect_summary.csv"
cp "${REPORT_DIR}/target_effect/target_effect_summary.md" "${REPORT_DIR}/target_effect_summary.md"
log "TARGET_AUDIT_DONE"
