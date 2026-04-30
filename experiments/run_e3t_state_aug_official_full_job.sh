#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/e3t_state_aug_official_full}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"

mkdir -p "${JOB_ROOT}"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[e3t-full] running pid=${pid}"
    ps -fp "${pid}" || true
  else
    echo "[e3t-full] not running"
    if [[ -f "${PID_FILE}" ]]; then
      echo "[e3t-full] stale pid=$(cat "${PID_FILE}")"
    fi
  fi

  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[e3t-full] latest_log=${log}"
    if [[ -f "${log}" ]]; then
      echo "[e3t-full] log_tail:"
      tail -n 40 "${log}"
    fi
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi

  local tag log
  tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  log="${JOB_ROOT}/e3t_state_aug_official_full_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"

  : "${LAYOUT:=counter_circuit}"
  : "${SEED:=42}"
  : "${NUM_SEEDS:=10}"
  : "${WANDB_MODE:=online}"
  : "${WANDB_PROJECT:=ov2-paper-repro}"
  : "${WANDB_ENTITY:=huiby_tsinghua23}"
  : "${CUDA_VISIBLE_DEVICES:=0,1}"

  {
    echo "[e3t-full] launch_time=$(date -Is)"
    echo "[e3t-full] host=$(hostname)"
    echo "[e3t-full] root=${ROOT}"
    echo "[e3t-full] python=${PYTHON}"
    echo "[e3t-full] layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS}"
    echo "[e3t-full] wandb_entity=${WANDB_ENTITY} wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"
    echo "[e3t-full] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  } > "${log}"

  nohup env \
    PYTHON="${PYTHON}" \
    LAYOUT="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash "${ROOT}/experiments/run_e3t_state_aug_official_suite.sh" full \
    >> "${log}" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[e3t-full] started pid=${pid} log=${log}"
}

stop() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    echo "[e3t-full] sent TERM to pid=${pid}"
  else
    echo "[e3t-full] not running"
  fi
}

case "${1:-start}" in
  start)
    start
    ;;
  status)
    status
    ;;
  tail)
    if [[ ! -f "${LATEST_LOG}" ]]; then
      echo "[e3t-full] no latest log" >&2
      exit 1
    fi
    tail -f "$(cat "${LATEST_LOG}")"
    ;;
  stop)
    stop
    ;;
  *)
    echo "Usage: $0 [start|status|tail|stop]" >&2
    exit 2
    ;;
esac
