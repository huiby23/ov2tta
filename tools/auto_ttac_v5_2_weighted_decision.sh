#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"

V5_4_DECISION_FILE="${V5_4_DECISION_FILE:-reports/ttac_v5_4_delta_full_500_seedchunk100_20260623/auto_decision/decision.env}"
V5_4_SUMMARY="${V5_4_SUMMARY:-reports/ttac_v5_4_delta_full_500_seedchunk100_20260623/final_summary/partial_or_full_summary.md}"
V5_2_TAG="${V5_2_TAG:-v5_2_latest_weighted_estimator_full_500_seedchunk100_20260623}"
V5_2_REPORT_DIR="${V5_2_REPORT_DIR:-reports/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623}"
V5_2_LOG_DIR="${V5_2_LOG_DIR:-logs/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-50}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MARGIN="${MARGIN:-0.0}"
DECISION_DIR="${DECISION_DIR:-reports/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623/auto_decision}"

cd "${ROOT}"
mkdir -p "${V5_2_LOG_DIR}" "${DECISION_DIR}"
LOG_FILE="${V5_2_LOG_DIR}/auto_decision.log"
DECISION_FILE="${DECISION_DIR}/decision.env"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

count_shards() {
  ls "${RUN_DIR}/reward_summary_cross_${V5_2_TAG}"_seedchunk*_shard*.csv 2>/dev/null | wc -l | tr -d ' '
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

log "v5.2 weighted follow-up watcher started"
while [[ ! -s "${V5_4_DECISION_FILE}" ]]; do
  log "waiting for v5.4 decision file: ${V5_4_DECISION_FILE}"
  sleep "${SLEEP_SECONDS}"
done

# shellcheck disable=SC1090
source "${V5_4_DECISION_FILE}"
log "v5.4 decision action=${ACTION:-unknown}"
if [[ "${ACTION:-}" != "launch_v5_2_latest_weighted_full" ]]; then
  {
    echo "ACTION=noop"
    echo "REASON=v5.4 decision did not launch v5.2 weighted"
    echo "V5_4_ACTION=${ACTION:-}"
  } > "${DECISION_FILE}"
  log "no-op because v5.4 decision was ${ACTION:-unknown}"
  exit 0
fi

while true; do
  completed="$(count_shards)"
  log "v5_2_completed_shards=${completed}/${EXPECTED_SHARDS}"
  if [[ "${completed}" -ge "${EXPECTED_SHARDS}" ]]; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

SUMMARY_DIR="${V5_2_REPORT_DIR}/final_summary"
mkdir -p "${SUMMARY_DIR}"
"${PYTHON}" tools/summarize_ttac_sharded_eval.py \
  --run_dir "${RUN_DIR}" \
  --tag "${V5_2_TAG}" \
  --output_dir "${SUMMARY_DIR}" \
  > "${DECISION_DIR}/summarize_stdout.log" 2> "${DECISION_DIR}/summarize_stderr.log"

V5_2_SUMMARY="${SUMMARY_DIR}/partial_or_full_summary.md"
v5_2_xp="$(parse_xp "${V5_2_SUMMARY}")"
v5_4_xp="$(parse_xp "${V5_4_SUMMARY}")"
threshold="$("${PYTHON}" - <<PY
print(float("${v5_2_xp}") + float("${MARGIN}"))
PY
)"

log "v5_4_xp=${v5_4_xp} v5_2_weighted_xp=${v5_2_xp} threshold=${threshold}"

if "${PYTHON}" - <<PY
import sys
sys.exit(0 if float("${v5_4_xp}") > float("${threshold}") else 1)
PY
then
  {
    echo "ACTION=v5_4_beats_v5_2_weighted"
    echo "V5_4_XP=${v5_4_xp}"
    echo "V5_2_WEIGHTED_XP=${v5_2_xp}"
    echo "SUMMARY_V5_4=${V5_4_SUMMARY}"
    echo "SUMMARY_V5_2=${V5_2_SUMMARY}"
  } > "${DECISION_FILE}"
  log "v5.4 beats weighted v5.2; no follow-up launched"
else
  ts="$(date +%Y%m%d_%H%M%S)"
  cmd="TS=${ts} nohup tools/run_ttac_v5_5_delta_push_pair20.sh > logs/ttac_v5_5_delta_push_pair20_${ts}.nohup.log 2>&1 &"
  {
    echo "ACTION=launch_v5_5_delta_push_pair20"
    echo "V5_4_XP=${v5_4_xp}"
    echo "V5_2_WEIGHTED_XP=${v5_2_xp}"
    echo "COMMAND=${cmd}"
    echo "SUMMARY_V5_4=${V5_4_SUMMARY}"
    echo "SUMMARY_V5_2=${V5_2_SUMMARY}"
  } > "${DECISION_FILE}"
  log "v5.4 did not beat weighted v5.2; launching v5.5 pair20"
  log "command=${cmd}"
  bash -lc "${cmd}"
fi
