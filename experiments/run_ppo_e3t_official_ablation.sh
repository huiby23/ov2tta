#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/ppo_e3t_official_ablation}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[ppo-e3t] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[ppo-e3t] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[ppo-e3t] not running"
    [[ -f "${PID_FILE}" ]] && echo "[ppo-e3t] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[ppo-e3t] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 80 "${log}"
  fi
}

set_scale() {
  if [[ "${MODE}" == "smoke" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-128}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-64}"
    NUM_ENVS="${NUM_ENVS:-4}"
    NUM_STEPS="${NUM_STEPS:-4}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-1}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-1}"
    NUM_SEEDS="${NUM_SEEDS:-2}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-0}"
  elif [[ "${MODE}" == "full" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5e6}"
    NUM_ENVS="${NUM_ENVS:-64}"
    NUM_STEPS="${NUM_STEPS:-256}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  else
    echo "[ppo-e3t] unknown MODE=${MODE}; use smoke or full" >&2
    exit 2
  fi
}

run_train() {
  local stage="$1"
  local actor_condition="$2"
  local enable_ce="$3"
  local condition_actor="$4"
  local prefix="${PREFIX_BASE}_${stage}_${MODE}_${JOB_TAG}"
  echo "[ppo-e3t] stage_start=$(date -Is) stage=${stage} actor_condition=${actor_condition} enable_ce=${enable_ce} condition_actor=${condition_actor}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo_e3t_official/main.py \
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
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.E3T_PARTNER_SOURCE="main_policy" \
    model.E3T_ACTOR_CONDITION="${actor_condition}" \
    model.E3T_ENABLE_CE="${enable_ce}" \
    model.E3T_CONDITION_ACTOR="${condition_actor}" \
    model.USE_OFFICIAL_E3T_PARTNER=False \
    model.RAND=0.0 \
    model.COPY=0.0
  echo "[ppo-e3t] stage_done=$(date -Is) stage=${stage}"
}

run_stage() {
  case "$1" in
    predicted_ce)
      run_train "predicted_ce" "predicted_partner" "True" "True"
      ;;
    constant_ce)
      run_train "constant_ce" "constant" "True" "True"
      ;;
    no_ce)
      run_train "no_ce" "predicted_partner" "False" "True"
      ;;
    no_actor_condition)
      run_train "no_actor_condition" "predicted_partner" "True" "False"
      ;;
    *)
      echo "[ppo-e3t] unknown stage=$1" >&2
      exit 2
      ;;
  esac
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  set_scale
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${SEED}" --num-seeds "${NUM_SEEDS}" --num-iterations 0 --out "${JOB_ROOT}/seed_manifest_${JOB_TAG}.json" || true
  echo "[ppo-e3t] launch_time=$(date -Is)"
  echo "[ppo-e3t] host=$(hostname) root=${ROOT}"
  echo "[ppo-e3t] mode=${MODE} layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} stages=${STAGES} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[ppo-e3t] scale total=${TOTAL_TIMESTEPS} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} ce_epochs=${CONTEXT_UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES}"
  for stage in ${STAGES}; do
    run_stage "${stage}"
  done
  echo "[ppo-e3t] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-ppo_e3t_official}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export STAGES="${STAGES:-predicted_ce constant_ce no_ce}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="${JOB_ROOT}/ppo_e3t_official_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" MODE="${MODE}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" \
    SEED="${SEED}" STAGES="${STAGES}" WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[ppo-e3t] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline)
    : "${MODE:=full}" "${PREFIX_BASE:=ppo_e3t_official}" "${LAYOUT:=counter_circuit}" "${SEED:=42}"
    : "${STAGES:=predicted_ce constant_ce no_ce}" "${WANDB_MODE:=online}" "${WANDB_PROJECT:=ov2-paper-repro}" "${WANDB_ENTITY:=huiby_tsinghua23}"
    pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop)
    if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[ppo-e3t] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
