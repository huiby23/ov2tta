#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/ppo_e3t_official_figure4_eval}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
NUM_SEEDS="${EVAL_SEEDS:-500}"
SEED="${SEED:-42}"
SCRIPT="experiments/overcooked_v2_experiments/ppo_e3t_official/utils/visualize_ppo.py"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

PREDICTED_DIR="runs/figure4_ppo_e3t_official_predicted_ce_full_20260428-ppo-e3t-ablation/20260428-124613_mz4qrn7j_counter_circuit_avs-full"
CONSTANT_DIR="runs/figure4_ppo_e3t_official_constant_ce_full_20260428-ppo-e3t-ablation/20260428-142809_ju78l6cx_counter_circuit_avs-full"
NOCE_DIR="runs/figure4_ppo_e3t_official_no_ce_full_20260428-ppo-e3t-ablation/20260428-155737_0foocnfr_counter_circuit_avs-full"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[ppo-e3t-fig4] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[ppo-e3t-fig4] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[ppo-e3t-fig4] not running"
    [[ -f "${PID_FILE}" ]] && echo "[ppo-e3t-fig4] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[ppo-e3t-fig4] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 80 "${log}"
  fi
}

run_eval() {
  local name="$1"
  local run_dir="$2"
  echo "[ppo-e3t-fig4] eval_start=$(date -Is) name=${name} run_dir=${run_dir} num_seeds=${NUM_SEEDS}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[ppo-e3t-fig4] missing run_dir=${run_dir}" >&2
    exit 2
  fi
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${SCRIPT}" \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${NUM_SEEDS}" \
    --seed "${SEED}" \
    --no_viz
  echo "[ppo-e3t-fig4] eval_done=$(date -Is) name=${name} output=${run_dir}/reward_summary_cross.csv"
}

pipeline() {
  print_repro_env
  echo "[ppo-e3t-fig4] launch_time=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  run_eval predicted_ce "${PREDICTED_DIR}"
  run_eval constant_ce "${CONSTANT_DIR}"
  run_eval no_ce "${NOCE_DIR}"
  echo "[ppo-e3t-fig4] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  local log="${JOB_ROOT}/ppo_e3t_figure4_eval_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" EVAL_SEEDS="${NUM_SEEDS}" SEED="${SEED}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[ppo-e3t-fig4] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop)
    if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[ppo-e3t-fig4] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
