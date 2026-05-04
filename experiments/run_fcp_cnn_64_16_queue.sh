#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/fcp_cnn_64_16_queue}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

E3T_PID_FILE="${E3T_PID_FILE:-logs/e3t_ppo_rnn_64_16_standard_state_aug/current.pid}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
POPULATION_DIR="${POPULATION_DIR:-fcp_populations/ppo_cnn_standard_64_16_counter_circuit_8policies_x10}"
PREFIX="${PREFIX:-figure4_fcp_cnn_64_16}"
LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
POPULATION_GROUPS="${POPULATION_GROUPS:-10}"
POPULATION_SIZE="${POPULATION_SIZE:-8}"
WAIT_FOR_E3T="${WAIT_FOR_E3T:-true}"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

e3t_running() {
  [[ -f "${E3T_PID_FILE}" ]] && kill -0 "$(cat "${E3T_PID_FILE}")" >/dev/null 2>&1
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[fcp-cnn] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[fcp-cnn] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[fcp-cnn] not running"
    [[ -f "${PID_FILE}" ]] && echo "[fcp-cnn] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[fcp-cnn] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
}

wait_for_e3t() {
  if [[ "${WAIT_FOR_E3T}" != "true" ]]; then
    return
  fi
  while e3t_running; do
    echo "[fcp-cnn] waiting_for_e3t pid=$(cat "${E3T_PID_FILE}") time=$(date -Is)"
    sleep 300
  done
}

prepare_population() {
  if [[ ! -d "${SOURCE_RUN_DIR}" ]]; then
    echo "[fcp-cnn] missing SOURCE_RUN_DIR=${SOURCE_RUN_DIR}" >&2
    exit 1
  fi

  rm -rf "${POPULATION_DIR}"
  mkdir -p "${POPULATION_DIR}"
  for group in $(seq 0 $((POPULATION_GROUPS - 1))); do
    local fcp_dir="${POPULATION_DIR}/fcp_${group}"
    mkdir -p "${fcp_dir}"
    for policy in $(seq 0 $((POPULATION_SIZE - 1))); do
      local src="${ROOT}/${SOURCE_RUN_DIR}/run_${policy}"
      local dst="${fcp_dir}/run_${policy}"
      if [[ ! -d "${src}/ckpt_final" ]]; then
        echo "[fcp-cnn] missing checkpoint ${src}/ckpt_final" >&2
        exit 1
      fi
      ln -s "${src}" "${dst}"
    done
  done
  echo "[fcp-cnn] population_ready dir=${POPULATION_DIR} groups=${POPULATION_GROUPS} size=${POPULATION_SIZE} source=${SOURCE_RUN_DIR}"
}

summarize_cross() {
  local csv="$1"
  "${PYTHON}" - "${csv}" <<PY
import csv
import re
import sys
from statistics import mean
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
sp = []
xp = []
for r in rows:
    label = r.get("policy_labels") or r.get("checkpoint") or ""
    m = re.match(r"cross-(\d+)_(\d+)$", label)
    if not m:
        continue
    value = float(r["total_reward"])
    if m.group(1) == m.group(2):
        sp.append(value)
    else:
        xp.append(value)
print(f"[summary] csv={path}")
print(f"[summary] SP={mean(sp) if sp else float(nan):.3f} XP={mean(xp) if xp else float(nan):.3f} n_sp={len(sp)} n_xp={len(xp)}")
PY
}

evaluate_run() {
  local run_dir="$1"
  echo "[fcp-cnn] eval_start=$(date -Is) run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
      --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${SEED}" --no_viz
  summarize_cross "${run_dir}/reward_summary_cross.csv"
  echo "[fcp-cnn] eval_done=$(date -Is) run_dir=${run_dir}"
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  echo "[fcp-cnn] queue_start=$(date -Is) host=$(hostname) wait_for_e3t=${WAIT_FOR_E3T} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[fcp-cnn] source=${SOURCE_RUN_DIR} population=${POPULATION_DIR} prefix=${PREFIX} layout=${LAYOUT} seed=${SEED}"
  wait_for_e3t
  prepare_population
  echo "[fcp-cnn] train_start=$(date -Is)"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
      +experiment=cnn \
      +env=original \
      env.ENV_KWARGS.layout="${LAYOUT}" \
      +FCP="${POPULATION_DIR}" \
      +OPTIONAL_PREFIX="${PREFIX}" \
      SEED="${SEED}" \
      NUM_SEEDS=1 \
      NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
      VISUALIZE=False \
      wandb.ENTITY="${WANDB_ENTITY}" \
      wandb.PROJECT="${WANDB_PROJECT}" \
      wandb.WANDB_MODE="${WANDB_MODE}" \
      model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
      model.NUM_ENVS="${NUM_ENVS}" \
      model.NUM_STEPS="${NUM_STEPS}" \
      model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
      model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  local run_dir
  run_dir="$(latest_run_dir "${PREFIX}")"
  echo "[fcp-cnn] train_done=$(date -Is) run_dir=${run_dir}"
  evaluate_run "${run_dir}"
  echo "[fcp-cnn] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  local log="${JOB_ROOT}/fcp_cnn_64_16_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${tag}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[fcp-cnn] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop)
    if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[fcp-cnn] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
