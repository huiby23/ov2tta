#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
RUN_ID="${RUN_ID:-$(cat reports/acp_latest_run_id.txt)}"
DIAG_EPISODES="${DIAG_EPISODES:-20}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
COMPAT_LIMIT="${COMPAT_LIMIT:-128}"
POLL_SECONDS="${POLL_SECONDS:-120}"
REPORT_DIR="reports/acp_${RUN_ID}"
DISTILL_DIR="runs_by_config/counter_circuit_full_obs_64env_16mb_10M/acp/${RUN_ID}_distill_only"
NOHIST_DIR="runs_by_config/counter_circuit_full_obs_64env_16mb_10M/acp/${RUN_ID}_no_history"

log() { echo "[$(date +%F' '%T)] $*"; }
wait_path() {
  local path="$1"
  while [[ ! -e "${path}" ]]; do
    log "waiting for ${path}"
    sleep "${POLL_SECONDS}"
  done
}
wait_all_run_finals() {
  local run_dir="$1"
  local expected="$2"
  while true; do
    local count
    count=$(find "${run_dir}" -mindepth 2 -maxdepth 2 -type d -name ckpt_final 2>/dev/null | wc -l); count="${count//[[:space:]]/}"
    if [[ "${count}" -ge "${expected}" ]]; then
      log "found ${count}/${expected} final checkpoints in ${run_dir}"
      break
    fi
    log "waiting for final checkpoints in ${run_dir}: ${count}/${expected}"
    sleep "${POLL_SECONDS}"
  done
}
run_diag() {
  local name="$1"
  local run_dir="$2"
  local mode="$3"
  local out_dir="${REPORT_DIR}/diagnostics/${name}_${mode}"
  mkdir -p "${out_dir}"
  log "diagnostics ${name} mode=${mode} run_dir=${run_dir}"
  "${PYTHON}" -m overcooked_v2_experiments.acp.utils.zsc_diagnostics \
    --single_run_dir "${run_dir}" \
    --single_name "${name}_${mode}" \
    --single_backend acp \
    --single_eval_mode "${mode}" \
    --num_eval_seeds "${EVAL_SEEDS}" \
    --num_diag_episodes "${DIAG_EPISODES}" \
    --compatibility_sample_limit "${COMPAT_LIMIT}" \
    --output_dir "${out_dir}"
}

log "ACP diagnostics watcher run_id=${RUN_ID}"
wait_all_run_finals "${DISTILL_DIR}" "${NUM_SEEDS:-10}"
run_diag distill_only "${DISTILL_DIR}" state_adapt

if [[ -d "${NOHIST_DIR}" ]]; then
  wait_all_run_finals "${NOHIST_DIR}" "${NUM_SEEDS:-10}"
  run_diag no_history "${NOHIST_DIR}" state_adapt
fi

wait_path "${REPORT_DIR}/finetune_run_dir.txt"
FINETUNE_DIR="$(cat "${REPORT_DIR}/finetune_run_dir.txt")"
wait_all_run_finals "${FINETUNE_DIR}" "${NUM_SEEDS:-10}"
run_diag ppo_finetune "${FINETUNE_DIR}" memory_off
run_diag online_latent "${FINETUNE_DIR}" state_adapt
log "ACP diagnostics complete for ${RUN_ID}"
