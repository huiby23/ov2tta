#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/e3t_ppo_rnn_64_16_standard_state_aug}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[e3t-ppo-rnn] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[e3t-ppo-rnn] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[e3t-ppo-rnn] not running"
    [[ -f "${PID_FILE}" ]] && echo "[e3t-ppo-rnn] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[e3t-ppo-rnn] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
}

set_scale() {
  if [[ "${MODE}" == "smoke" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-256}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-128}"
    NUM_ENVS="${NUM_ENVS:-4}"
    NUM_STEPS="${NUM_STEPS:-4}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-1}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-1}"
    NUM_SEEDS="${NUM_SEEDS:-2}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
    EVAL_SEEDS="${EVAL_SEEDS:-2}"
  elif [[ "${MODE}" == "full" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
    NUM_ENVS="${NUM_ENVS:-64}"
    NUM_STEPS="${NUM_STEPS:-256}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
    EVAL_SEEDS="${EVAL_SEEDS:-500}"
  else
    echo "[e3t-ppo-rnn] unknown MODE=${MODE}; use smoke or full" >&2
    exit 2
  fi
}

base_args() {
  E3T_ARGS=(
    +env=default
    +env.ENV_KWARGS.layout="${LAYOUT}"
    SEED="${SEED}"
    NUM_SEEDS="${NUM_SEEDS}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}"
    VISUALIZE=False
    wandb.ENTITY="${WANDB_ENTITY}"
    wandb.PROJECT="${WANDB_PROJECT}"
    wandb.WANDB_MODE="${WANDB_MODE}"
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}"
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}"
    model.NUM_ENVS="${NUM_ENVS}"
    model.NUM_STEPS="${NUM_STEPS}"
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}"
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
    model.LR=0.00025
    model.ENT_COEF=0.01
    model.MAX_GRAD_NORM=0.25
    model.USE_HISTORY_CONTEXT=True
    model.MOA_TO_ACTOR=True
    model.MOA_TO_ACTOR_DETACH=False
    model.USE_MOA_AUX=True
    model.MOA_COEF=1.0
    model.SEPARATE_MOA_UPDATE=True
    +model.OFFICIAL_SEPARATE_PARAM_SPLIT=True
    model.USE_PARTNER_MIX=True
    model.PARTNER_MIX_MODE=ppo_consistent
    model.PARTNER_MIX_EPS=0.55
    +model.PARTNER_COPY_COEF=0.1
    model.CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS}"
  )
}

summarize_cross() {
  local csv="$1"
  "${PYTHON}" - "${csv}" <<'PY'
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
print(f"[summary] SP={mean(sp) if sp else float('nan'):.3f} XP={mean(xp) if xp else float('nan'):.3f} n_sp={len(sp)} n_xp={len(xp)}")
PY
}

evaluate_run() {
  local run_dir="$1"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[e3t-ppo-rnn] missing run_dir=${run_dir}" >&2
    exit 1
  fi
  echo "[e3t-ppo-rnn] eval_start=$(date -Is) run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" experiments/overcooked_v2_experiments/e3t_ppo/utils/visualize_ppo.py \
      --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${SEED}" --no_viz
  summarize_cross "${run_dir}/reward_summary_cross.csv"
  echo "[e3t-ppo-rnn] eval_done=$(date -Is) run_dir=${run_dir}"
}

run_train() {
  local stage="$1"
  local experiment="$2"
  local prefix="${PREFIX_BASE}_${stage}_${MODE}_${JOB_TAG}"
  base_args
  echo "[e3t-ppo-rnn] stage_start=$(date -Is) stage=${stage} experiment=${experiment} prefix=${prefix}"
  local -a args=(+experiment="${experiment}" +OPTIONAL_PREFIX="${prefix}" "${E3T_ARGS[@]}")
  if [[ "${stage}" == "state_aug" ]]; then
    args+=(NUM_ITERATIONS="${NUM_ITERATIONS}")
  fi
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" -m overcooked_v2_experiments.e3t_ppo.main "${args[@]}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[e3t-ppo-rnn] stage_done=$(date -Is) stage=${stage} run_dir=${run_dir}"
  evaluate_run "${run_dir}"
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  set_scale
  echo "[e3t-ppo-rnn] launch_time=$(date -Is) host=$(hostname) root=${ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[e3t-ppo-rnn] mode=${MODE} layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} stages=${STAGES} eval_seeds=${EVAL_SEEDS}"
  echo "[e3t-ppo-rnn] scale total=${TOTAL_TIMESTEPS} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} state_aug_iterations=${NUM_ITERATIONS}"
  for stage in ${STAGES}; do
    case "${stage}" in
      standard) run_train standard rnn ;;
      state_aug) run_train state_aug rnn-sa ;;
      *) echo "[e3t-ppo-rnn] unknown stage=${stage}" >&2; exit 2 ;;
    esac
  done
  echo "[e3t-ppo-rnn] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_e3t_ppo_rnn_64_16}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export STAGES="${STAGES:-standard state_aug}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="${JOB_ROOT}/e3t_ppo_rnn_64_16_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" MODE="${MODE}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" \
    SEED="${SEED}" STAGES="${STAGES}" WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[e3t-ppo-rnn] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline)
    : "${MODE:=full}" "${PREFIX_BASE:=figure4_e3t_ppo_rnn_64_16}" "${LAYOUT:=counter_circuit}" "${SEED:=42}"
    : "${STAGES:=standard state_aug}" "${WANDB_MODE:=online}" "${WANDB_PROJECT:=ov2-paper-repro}" "${WANDB_ENTITY:=huiby_tsinghua23}"
    pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop)
    if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[e3t-ppo-rnn] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
