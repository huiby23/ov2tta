#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
DATASET="${DATASET:-reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz}"
ESTIMATOR="${ESTIMATOR:-reports/ttac_v5_agreement_estimator_20260609_180926/agreement_estimator/agreement_estimator.npz}"
BASE_CSV="${BASE_CSV:-${RUN_DIR}/reward_summary_cross_v5_2_full_base_base_no_test_adapt.csv}"
TTAC_CSV="${TTAC_CSV:-${RUN_DIR}/reward_summary_cross_v5_2_tv003_coef20_full_sharded_20260616_1413_merged.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/ttac_v5_2_attribution_audit_${TS}}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SEED="${SEED:-42}"
HISTORY_LEN="${HISTORY_LEN:-50}"

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
DATASET=${DATASET}
ESTIMATOR=${ESTIMATOR}
BASE_CSV=${BASE_CSV}
TTAC_CSV=${TTAC_CSV}
OUTPUT_DIR=${OUTPUT_DIR}
MAX_SAMPLES=${MAX_SAMPLES}
SEED=${SEED}
HISTORY_LEN=${HISTORY_LEN}
EOF

"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/paired_attribution_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --base_csv "${BASE_CSV}" \
  --ttac_csv "${TTAC_CSV}" \
  --ttac_v5_estimator_path "${ESTIMATOR}" \
  --history_len "${HISTORY_LEN}" \
  --max_samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

echo "[attribution] summary: ${OUTPUT_DIR}/paired_attribution_summary.md"
