#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
source "${ROOT}/experiments/repro_env.sh"
JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
PREFIX="${PREFIX:-fcp_mep_frozen_mm_K${POPULATION_SIZE:-10}_${NUM_ENVS:-64}_${NUM_MINIBATCHES:-16}_${TOTAL_TIMESTEPS:-10000000}_${JOB_TAG}}"
LOG_DIR="${LOG_DIR:-logs/fcp_mep_frozen_mm}"
mkdir -p "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/${PREFIX}.log}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
latest_run_dir() { find "runs/${PREFIX}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true; }
summarize_cross() {
  local csv_path="$1"
  "${PYTHON}" - "${csv_path}" <<'PY'
import csv, math, re, statistics, sys
sp=[]; xp=[]
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        m=re.match(r"cross-(\d+)_(\d+)$", r.get("policy_labels", ""))
        if not m: continue
        v=float(r["total_reward"])
        (sp if m.group(1)==m.group(2) else xp).append(v)
print(f"[fcp-mep-frozen-mm] result SP={statistics.mean(sp) if sp else math.nan:.4f} XP={statistics.mean(xp) if xp else math.nan:.4f} n_sp={len(sp)} n_xp={len(xp)} csv={sys.argv[1]}")
PY
}
run_pipeline() {
  echo "[fcp-mep-frozen-mm] train_start=$(date -Is) prefix=${PREFIX} log=${LOG}"
  PREFIX="${PREFIX}" bash experiments/run_fcp_mep_frozen_mm_64_16.sh
  local run_dir
  run_dir="$(latest_run_dir)"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_0/ckpt_final" ]]; then
    echo "[fcp-mep-frozen-mm] missing run_dir/checkpoint prefix=${PREFIX} run_dir=${run_dir}" >&2
    exit 1
  fi
  echo "[fcp-mep-frozen-mm] train_done=$(date -Is) run_dir=${run_dir}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.utils.visualize_ppo \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz
  summarize_cross "${run_dir}/reward_summary_cross.csv"
  echo "[fcp-mep-frozen-mm] eval_done=$(date -Is) run_dir=${run_dir}"
}
if [[ "${1:-}" == "status" ]]; then
  pgrep -af "${PREFIX}|overcooked_v2_experiments.fcp_mep_frozen_mm|visualize_ppo" || true
  [[ -f "${LOG}" ]] && tail -n 120 "${LOG}" || true
  exit 0
fi
run_pipeline 2>&1 | tee "${LOG}"
