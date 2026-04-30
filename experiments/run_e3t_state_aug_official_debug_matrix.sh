#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/e3t_state_aug_official_debug_matrix}"
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
    echo "[e3t-debug] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[e3t-debug] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[e3t-debug] not running"
    if [[ -f "${PID_FILE}" ]]; then
      echo "[e3t-debug] stale pid=$(cat "${PID_FILE}")"
    fi
  fi

  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[e3t-debug] latest_log=${log}"
    if [[ -f "${log}" ]]; then
      echo "[e3t-debug] log_tail:"
      tail -n 80 "${log}"
    fi
  fi
}

set_scale() {
  if [[ "${MODE}" == "smoke" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-128}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-64}"
    NUM_STEPS="${NUM_STEPS:-4}"
    NUM_ENVS="${NUM_ENVS:-4}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-1}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-1}"
    NUM_SEEDS="${NUM_SEEDS:-1}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  elif [[ "${MODE}" == "full" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5e6}"
    NUM_STEPS="${NUM_STEPS:-256}"
    NUM_ENVS="${NUM_ENVS:-64}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  else
    echo "[e3t-debug] unknown MODE=${MODE}; use smoke or full" >&2
    exit 2
  fi
}

run_train() {
  local stage="$1"
  local layout="$2"
  local partner_source="$3"
  local actor_condition="$4"
  local enable_ce="$5"
  local rand="$6"

  local prefix="${PREFIX_BASE}_${stage}_${MODE}_${JOB_TAG}"
  echo "[e3t-debug] stage_start=$(date -Is) stage=${stage} layout=${layout} partner_source=${partner_source} actor_condition=${actor_condition} enable_ce=${enable_ce} rand=${rand}"

  "${PYTHON}" experiments/overcooked_v2_experiments/e3t_state_aug_official/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${layout}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.LR="${LR}" \
    model.GAE_LAMBDA="${GAE_LAMBDA}" \
    model.VF_COEF="${VF_COEF}" \
    model.MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
    model.ANNEAL_LR="${ANNEAL_LR}" \
    model.LR_WARMUP="${LR_WARMUP}" \
    model.CLIP_EPS="${CLIP_EPS}" \
    model.ENT_COEF="${ENT_COEF}" \
    model.RAND="${rand}" \
    model.E3T_PARTNER_SOURCE="${partner_source}" \
    model.E3T_ACTOR_CONDITION="${actor_condition}" \
    model.E3T_ENABLE_CE="${enable_ce}"

  echo "[e3t-debug] stage_done=$(date -Is) stage=${stage}"
}

run_stage() {
  case "$1" in
    ppo_equiv)
      run_train "ppo_equiv" "${LAYOUT}" "main_policy" "constant" "False" "0.0"
      ;;
    main_policy_pred)
      run_train "main_policy_pred_ce" "${LAYOUT}" "main_policy" "predicted_partner" "True" "0.0"
      ;;
    human_constant)
      run_train "human_constant_ce_rand00" "${LAYOUT}" "human_branch" "constant" "True" "0.0"
      ;;
    full_rand00)
      run_train "full_e3t_rand00" "${LAYOUT}" "human_branch" "predicted_partner" "True" "0.0"
      ;;
    full_rand07)
      run_train "full_e3t_rand07" "${LAYOUT}" "human_branch" "predicted_partner" "True" "0.7"
      ;;
    simple_full_rand07)
      run_train "simple_full_e3t_rand07" "grounded_coord_simple" "human_branch" "predicted_partner" "True" "0.7"
      ;;
    *)
      echo "[e3t-debug] unknown stage=$1" >&2
      exit 2
      ;;
  esac
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/experiments/overcooked_v2_experiments/e3t_state_aug_official:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  set_scale

  echo "[e3t-debug] launch_time=$(date -Is)"
  echo "[e3t-debug] host=$(hostname) root=${ROOT}"
  echo "[e3t-debug] mode=${MODE} layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} stages=${STAGES}"
  echo "[e3t-debug] scale total=${TOTAL_TIMESTEPS} steps=${NUM_STEPS} envs=${NUM_ENVS} epochs=${UPDATE_EPOCHS} ce_epochs=${CONTEXT_UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES}"
  echo "[e3t-debug] wandb=${WANDB_ENTITY}/${WANDB_PROJECT} mode=${WANDB_MODE} cuda=${CUDA_VISIBLE_DEVICES:-unset}"

  for stage in ${STAGES}; do
    run_stage "${stage}"
  done

  echo "[e3t-debug] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi

  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-e3t_debug_matrix}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export STAGES="${STAGES:-ppo_equiv main_policy_pred human_constant full_rand00 full_rand07 simple_full_rand07}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export LR="${LR:-4e-4}"
  export GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
  export VF_COEF="${VF_COEF:-0.5}"
  export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
  export ANNEAL_LR="${ANNEAL_LR:-True}"
  export LR_WARMUP="${LR_WARMUP:-0.05}"
  export CLIP_EPS="${CLIP_EPS:-0.2}"
  export ENT_COEF="${ENT_COEF:-0.04}"

  local log
  log="${JOB_ROOT}/e3t_debug_matrix_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"

  nohup env \
    JOB_TAG="${JOB_TAG}" MODE="${MODE}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" \
    SEED="${SEED}" STAGES="${STAGES}" WANDB_MODE="${WANDB_MODE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_ENTITY="${WANDB_ENTITY}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    LR="${LR}" GAE_LAMBDA="${GAE_LAMBDA}" VF_COEF="${VF_COEF}" MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
    ANNEAL_LR="${ANNEAL_LR}" LR_WARMUP="${LR_WARMUP}" CLIP_EPS="${CLIP_EPS}" ENT_COEF="${ENT_COEF}" \
    bash "${BASH_SOURCE[0]}" pipeline \
    > "${log}" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[e3t-debug] started pid=${pid} log=${log}"
}

stop() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    echo "[e3t-debug] sent TERM to pid=${pid}"
  else
    echo "[e3t-debug] not running"
  fi
}

case "${1:-start}" in
  start)
    start
    ;;
  pipeline)
    : "${MODE:=full}" "${PREFIX_BASE:=e3t_debug_matrix}" "${LAYOUT:=counter_circuit}" "${SEED:=42}"
    : "${STAGES:=ppo_equiv main_policy_pred human_constant full_rand00 full_rand07 simple_full_rand07}"
    : "${WANDB_MODE:=online}" "${WANDB_PROJECT:=ov2-paper-repro}" "${WANDB_ENTITY:=huiby_tsinghua23}"
    : "${LR:=4e-4}" "${GAE_LAMBDA:=0.95}" "${VF_COEF:=0.5}" "${MAX_GRAD_NORM:=0.5}"
    : "${ANNEAL_LR:=True}" "${LR_WARMUP:=0.05}" "${CLIP_EPS:=0.2}" "${ENT_COEF:=0.04}"
    : "${JOB_TAG:=$(date +%Y%m%d-%H%M%S)}"
    pipeline
    ;;
  status)
    status
    ;;
  tail)
    if [[ ! -f "${LATEST_LOG}" ]]; then
      echo "[e3t-debug] no latest log" >&2
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
