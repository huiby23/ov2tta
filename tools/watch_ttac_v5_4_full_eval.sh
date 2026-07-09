#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
TAG="${TAG:-v5_4_delta_weighted_estimator_full_500_seedchunk100_20260623}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/ttac_v5_4_delta_full_500_seedchunk100_20260623/partial_summary}"
LOG="${LOG:-logs/ttac_v5_4_delta_full_500_seedchunk100_20260623/watcher.log}"
INTERVAL="${INTERVAL:-60}"
TARGET="${TARGET:-50}"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")" "${OUTPUT_DIR}"

last_count="-1"
while true; do
  count="$(ls "${RUN_DIR}"/reward_summary_cross_"${TAG}"_seedchunk*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${count}" != "${last_count}" ]]; then
    echo "[$(date '+%F %T')] completed_csvs=${count}" | tee -a "${LOG}"
    "${PYTHON}" tools/summarize_ttac_sharded_eval.py \
      --run_dir "${RUN_DIR}" \
      --tag "${TAG}" \
      --output_dir "${OUTPUT_DIR}" >> "${LOG}" 2>&1 || true
    last_count="${count}"
  fi
  if [[ "${count}" -ge "${TARGET}" ]]; then
    echo "[$(date '+%F %T')] target reached: ${count}/${TARGET}" | tee -a "${LOG}"
    break
  fi
  sleep "${INTERVAL}"
done
