#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MARGIN="${MARGIN:-0.0}"
DECISION_DIR="${DECISION_DIR:-reports/ttac_v5_5_pair20_auto_decision_20260623}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_5_pair20_auto_decision_20260623}"

cd "${ROOT}"
mkdir -p "${DECISION_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/auto_decision.log"
DECISION_FILE="${DECISION_DIR}/decision.env"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

latest_summary() {
  find reports -path '*/delta_push_pair20_summary.csv' -type f \
    | grep 'ttac_v5_5_delta_push_pair20_' \
    | sort \
    | tail -1
}

if [[ -s "${DECISION_FILE}" ]]; then
  log "decision already exists: ${DECISION_FILE}"
  exit 0
fi

log "v5.5 pair20 decision watcher started"
while true; do
  summary="$(latest_summary || true)"
  if [[ -n "${summary}" && -s "${summary}" ]]; then
    if "${PYTHON}" - "${summary}" <<'PY'
import csv
import sys

path = sys.argv[1]
required = {
    "base_no_test_adapt",
    "ttac_v5_2_latest",
    "ttac_v5_4_delta",
    "ttac_v5_4_delta_push",
    "ttac_v5_5_top_delta",
}
rows = {}
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        rows[row["mode"]] = row
missing = [m for m in required if m not in rows or int(float(rows[m].get("rows") or 0)) <= 0]
raise SystemExit(1 if missing else 0)
PY
    then
      break
    fi
    log "found summary but required modes are incomplete: ${summary}"
  else
    log "waiting for v5.5 pair20 summary"
  fi
  sleep "${SLEEP_SECONDS}"
done

summary="$(latest_summary)"
decision_json="$("${PYTHON}" - "${summary}" "${MARGIN}" <<'PY'
import csv
import json
import sys

path = sys.argv[1]
margin = float(sys.argv[2])
rows = {}
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        rows[row["mode"]] = row

def xp(mode):
    value = rows.get(mode, {}).get("XP", "")
    return float(value) if value else float("-inf")

baseline = max(xp("ttac_v5_2_latest"), xp("ttac_v5_4_delta"))
candidates = ["ttac_v5_4_delta_push", "ttac_v5_5_top_delta"]
winner = max(candidates, key=xp)
payload = {
    "summary": path,
    "baseline_xp": baseline,
    "winner": winner,
    "winner_xp": xp(winner),
    "passes": xp(winner) > baseline + margin,
    "margin": margin,
    "rows": {m: rows[m] for m in rows if m in ["base_no_test_adapt", "ttac_v5_2_latest", "ttac_v5_4_delta", *candidates]},
}
print(json.dumps(payload, sort_keys=True))
PY
)"

log "decision_json=${decision_json}"
passes="$("${PYTHON}" - "${decision_json}" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1])["passes"] else "0")
PY
)"
winner="$("${PYTHON}" - "${decision_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["winner"])
PY
)"
winner_xp="$("${PYTHON}" - "${decision_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["winner_xp"])
PY
)"
baseline_xp="$("${PYTHON}" - "${decision_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["baseline_xp"])
PY
)"

if [[ "${passes}" != "1" ]]; then
  {
    echo "ACTION=noop"
    echo "REASON=no_v5_5_candidate_beats_pair20_baseline"
    echo "WINNER=${winner}"
    echo "WINNER_XP=${winner_xp}"
    echo "BASELINE_XP=${baseline_xp}"
    echo "SUMMARY=${summary}"
    echo "DECISION_JSON=${decision_json}"
  } > "${DECISION_FILE}"
  log "no v5.5 candidate passed; no full eval launched"
  exit 0
fi

ts="$(date +%Y%m%d_%H%M%S)"
safe_winner="${winner#ttac_}"
output_tag="${safe_winner}_weighted_estimator_full_500_seedchunk100_${ts}"
report_dir="reports/${safe_winner}_full_500_seedchunk100_${ts}"
full_log_dir="logs/${safe_winner}_full_500_seedchunk100_${ts}"
cmd="mkdir -p ${full_log_dir} && MODE=${winner} OUTPUT_TAG=${output_tag} REPORT_DIR=${report_dir} LOG_DIR=${full_log_dir} EVAL_NUM_SEEDS=100 SEED_CHUNKS=5 SEED_CHUNK_STRIDE=1000 TOTAL_PAIRINGS=100 SHARD_SIZE=10 PARALLEL_JOBS=2 EVAL_BATCHES=10 SKIP_EXISTING=1 RESET_QUEUE=0 nohup tools/run_ttac_v5_5_full_eval_sharded.sh > ${full_log_dir}/nohup.log 2>&1 &"

{
  echo "ACTION=launch_full_eval"
  echo "WINNER=${winner}"
  echo "WINNER_XP=${winner_xp}"
  echo "BASELINE_XP=${baseline_xp}"
  echo "SUMMARY=${summary}"
  echo "OUTPUT_TAG=${output_tag}"
  echo "REPORT_DIR=${report_dir}"
  echo "LOG_DIR=${full_log_dir}"
  echo "COMMAND=${cmd}"
  echo "DECISION_JSON=${decision_json}"
} > "${DECISION_FILE}"

log "launching full eval for ${winner}"
log "command=${cmd}"
bash -lc "${cmd}"
