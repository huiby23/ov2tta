#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
TAG="${TAG:-v5_4_delta_weighted_estimator_full_500_seedchunk100_20260623}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_4_delta_full_500_seedchunk100_20260623}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_4_delta_full_500_seedchunk100_20260623}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-50}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
HISTORICAL_V5_2_XP="${HISTORICAL_V5_2_XP:-164.209}"
MARGIN="${MARGIN:-0.0}"
DECISION_DIR="${DECISION_DIR:-${REPORT_DIR}/auto_decision}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${DECISION_DIR}"
LOG_FILE="${LOG_DIR}/auto_decision.log"
DECISION_FILE="${DECISION_DIR}/decision.env"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

count_shards() {
  ls "${RUN_DIR}/reward_summary_cross_${TAG}"_seedchunk*_shard*.csv 2>/dev/null | wc -l | tr -d ' '
}

parse_xp() {
  local summary="$1"
  "${PYTHON}" - "$summary" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
match = re.search(r"\|\s*XP\s*\|\s*([0-9.]+)\s*\|", text)
if not match:
    raise SystemExit("XP not found")
print(match.group(1))
PY
}

if [[ -s "${DECISION_FILE}" ]]; then
  log "decision already exists: ${DECISION_FILE}"
  exit 0
fi

log "auto decision watcher started tag=${TAG} expected_shards=${EXPECTED_SHARDS}"
while true; do
  completed="$(count_shards)"
  log "completed_shards=${completed}/${EXPECTED_SHARDS}"
  if [[ "${completed}" -ge "${EXPECTED_SHARDS}" ]]; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

SUMMARY_DIR="${REPORT_DIR}/final_summary"
mkdir -p "${SUMMARY_DIR}"
"${PYTHON}" tools/summarize_ttac_sharded_eval.py \
  --run_dir "${RUN_DIR}" \
  --tag "${TAG}" \
  --output_dir "${SUMMARY_DIR}" \
  > "${DECISION_DIR}/summarize_stdout.log" 2> "${DECISION_DIR}/summarize_stderr.log"

SUMMARY_MD="${SUMMARY_DIR}/partial_or_full_summary.md"
xp="$(parse_xp "${SUMMARY_MD}")"
threshold="$("${PYTHON}" - <<PY
print(float("${HISTORICAL_V5_2_XP}") + float("${MARGIN}"))
PY
)"

log "final_xp=${xp} historical_v5_2_xp=${HISTORICAL_V5_2_XP} threshold=${threshold}"

action=""
cmd=""
if "${PYTHON}" - <<PY
import sys
sys.exit(0 if float("${xp}") > float("${threshold}") else 1)
PY
then
  action="launch_v5_2_latest_weighted_full"
  cmd="mkdir -p logs/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623 && nohup tools/run_ttac_v5_2_latest_weighted_full.sh > logs/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623/nohup.log 2>&1 &"
else
  action="launch_v5_5_delta_push_pair20"
  ts="$(date +%Y%m%d_%H%M%S)"
  cmd="TS=${ts} nohup tools/run_ttac_v5_5_delta_push_pair20.sh > logs/ttac_v5_5_delta_push_pair20_${ts}.nohup.log 2>&1 &"
fi

{
  echo "ACTION=${action}"
  echo "XP=${xp}"
  echo "HISTORICAL_V5_2_XP=${HISTORICAL_V5_2_XP}"
  echo "THRESHOLD=${threshold}"
  echo "COMMAND=${cmd}"
  echo "SUMMARY_MD=${SUMMARY_MD}"
} > "${DECISION_FILE}"

log "decision=${action}"
log "command=${cmd}"
bash -lc "${cmd}"
log "launched ${action}"
