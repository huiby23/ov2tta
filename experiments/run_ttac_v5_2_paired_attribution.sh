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
ESTIMATOR="${ESTIMATOR:-reports/ttac_v5_agreement_estimator_20260609_180926/agreement_estimator/agreement_estimator.npz}"
DATASET="${DATASET:-reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_state_selection_${TS}/stage2_paired_attribution}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_state_selection_${TS}/stage2_paired_attribution}"
BASE_CSV="${BASE_CSV:-${RUN_DIR}/reward_summary_cross_v5_2_stage1_base_no_test_adapt.csv}"
TTAC_CSV="${TTAC_CSV:-${RUN_DIR}/reward_summary_cross_v5_2_stage1_ttac_v5_2_latest.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SEED="${SEED:-42}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

if [[ ! -f "${BASE_CSV}" ]]; then
  echo "missing BASE_CSV=${BASE_CSV}" >&2
  exit 2
fi
if [[ ! -f "${TTAC_CSV}" ]]; then
  echo "missing TTAC_CSV=${TTAC_CSV}" >&2
  exit 2
fi

"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/paired_attribution_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${REPORT_DIR}" \
  --dataset "${DATASET}" \
  --base_csv "${BASE_CSV}" \
  --ttac_csv "${TTAC_CSV}" \
  --ttac_v5_estimator_path "${ESTIMATOR}" \
  --history_len 50 \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  > "${LOG_DIR}/paired_attribution.log" 2>&1
cat "${REPORT_DIR}/paired_attribution_summary.md"
