#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/e3t_state_aug_official_diagnosis}"
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
    echo "[e3t-diagnosis] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[e3t-diagnosis] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[e3t-diagnosis] not running"
    if [[ -f "${PID_FILE}" ]]; then
      echo "[e3t-diagnosis] stale pid=$(cat "${PID_FILE}")"
    fi
  fi

  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[e3t-diagnosis] latest_log=${log}"
    if [[ -f "${log}" ]]; then
      echo "[e3t-diagnosis] log_tail:"
      tail -n 60 "${log}"
    fi
  fi
}

run_train() {
  local prefix="$1"
  local total_timesteps="$2"
  local rew_shaping_horizon="$3"
  local num_steps="$4"
  local num_envs="$5"
  local update_epochs="$6"
  local num_minibatches="$7"
  local lr="$8"
  local gae_lambda="$9"
  local vf_coef="${10}"
  local max_grad_norm="${11}"
  local anneal_lr="${12}"
  local lr_warmup="${13}"
  local clip_eps="${14}"
  local ent_coef="${15}"
  local rand="${16}"

  echo "[e3t-diagnosis] stage_start=$(date -Is) prefix=${prefix} rand=${rand}"
  "${PYTHON}" experiments/overcooked_v2_experiments/e3t_state_aug_official/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${total_timesteps}" \
    model.REW_SHAPING_HORIZON="${rew_shaping_horizon}" \
    model.NUM_STEPS="${num_steps}" \
    model.NUM_ENVS="${num_envs}" \
    model.UPDATE_EPOCHS="${update_epochs}" \
    model.CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}" \
    model.NUM_MINIBATCHES="${num_minibatches}" \
    model.LR="${lr}" \
    model.GAE_LAMBDA="${gae_lambda}" \
    model.VF_COEF="${vf_coef}" \
    model.MAX_GRAD_NORM="${max_grad_norm}" \
    model.ANNEAL_LR="${anneal_lr}" \
    model.LR_WARMUP="${lr_warmup}" \
    model.CLIP_EPS="${clip_eps}" \
    model.ENT_COEF="${ent_coef}" \
    model.RAND="${rand}"
  echo "[e3t-diagnosis] stage_done=$(date -Is) prefix=${prefix}"
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

  echo "[e3t-diagnosis] launch_time=$(date -Is)"
  echo "[e3t-diagnosis] host=$(hostname)"
  echo "[e3t-diagnosis] root=${ROOT}"
  echo "[e3t-diagnosis] python=${PYTHON}"
  echo "[e3t-diagnosis] layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS}"
  echo "[e3t-diagnosis] wandb_entity=${WANDB_ENTITY} wandb_project=${WANDB_PROJECT} wandb_mode=${WANDB_MODE}"
  echo "[e3t-diagnosis] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

  # Stage A: pure E3T single-run. This intentionally does not pass NUM_ITERATIONS.
  run_train \
    "figure4_e3t_diag_official_single_rand07_${JOB_TAG}" \
    "4.8e6" "2.5e6" "400" "30" "8" "6" \
    "1e-3" "0.98" "1.0" "0.1" "False" "0.0" "0.05" "0.1" "0.7"

  # Stage B: same E3T structure, but PPO-CNN baseline optimization/budget.
  for rand_cfg in 0.0 0.3 0.7; do
    rand_tag="${rand_cfg/./}"
    run_train \
      "figure4_e3t_diag_ppo_hparams_rand${rand_tag}_${JOB_TAG}" \
      "1e7" "5e6" "256" "64" "4" "16" \
      "4e-4" "0.95" "0.5" "0.5" "True" "0.05" "0.2" "0.04" "${rand_cfg}"
  done

  echo "[e3t-diagnosis] pipeline_done=$(date -Is)"
}

start() {
  if is_running; then
    status
    exit 0
  fi

  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export NUM_SEEDS="${NUM_SEEDS:-10}"
  export NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

  local log
  log="${JOB_ROOT}/e3t_state_aug_official_diagnosis_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"

  nohup env \
    JOB_TAG="${JOB_TAG}" \
    LAYOUT="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline \
    > "${log}" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[e3t-diagnosis] started pid=${pid} log=${log}"
}

stop() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    echo "[e3t-diagnosis] sent TERM to pid=${pid}"
  else
    echo "[e3t-diagnosis] not running"
  fi
}

case "${1:-start}" in
  start)
    start
    ;;
  pipeline)
    pipeline
    ;;
  status)
    status
    ;;
  tail)
    if [[ ! -f "${LATEST_LOG}" ]]; then
      echo "[e3t-diagnosis] no latest log" >&2
      exit 1
    fi
    tail -f "$(cat "${LATEST_LOG}")"
    ;;
  stop)
    stop
    ;;
  *)
    echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2
    exit 2
    ;;
esac
