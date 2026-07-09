#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-16}"
COMA_NUM_STEPS="${COMA_NUM_STEPS:-128}"
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PROJECT="${PROJECT:-ov2_classic_marl_partners}"
METHODS="${METHODS:-qplex,wqmix,coma}"
RUN_ROOT="${RUN_ROOT:-runs/classic_marl_ov2_partners_${TS}}"
LOG_DIR="${LOG_DIR:-logs/classic_marl_ov2_partners_${TS}}"
RUN_SPXP="${RUN_SPXP:-1}"
SPXP_EVAL_SEEDS="${SPXP_EVAL_SEEDS:-500}"
ACTION_MODE="${ACTION_MODE:-greedy}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
QUEUE_NAME="${QUEUE_NAME:-parallel_queue.log}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/JaxMARL:${ROOT}/JaxMARL/baselines/QLearning:${ROOT}/experiments:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

QUEUE="${LOG_DIR}/${QUEUE_NAME}"
touch "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

cat >> "${RUN_ROOT}/manifest.txt" <<EOF

[parallel_resume_$(date '+%F_%T')]
run_root=${RUN_ROOT}
log_dir=${LOG_DIR}
seed=${SEED}
num_seeds=${NUM_SEEDS}
total_timesteps=${TOTAL_TIMESTEPS}
num_envs=${NUM_ENVS}
num_steps=${NUM_STEPS}
coma_num_steps=${COMA_NUM_STEPS}
layout=${LAYOUT}
wandb_mode=${WANDB_MODE}
project=${PROJECT}
methods=${METHODS}
run_spxp=${RUN_SPXP}
spxp_eval_seeds=${SPXP_EVAL_SEEDS}
action_mode=${ACTION_MODE}
parallel_jobs=${PARALLEL_JOBS}
EOF

ckpt_path_for() {
  local method="$1"
  local i="$2"
  local prefix
  case "${method}" in
    qplex) prefix="qplex_cnn" ;;
    wqmix) prefix="wqmix_cnn" ;;
    coma) prefix="coma_cnn" ;;
    *) echo "unknown method ${method}" >&2; return 1 ;;
  esac
  echo "${RUN_ROOT}/${method}/overcooked_v2_${LAYOUT}/${prefix}_overcooked_v2_${LAYOUT}_seed${SEED}_vmap${i}.safetensors"
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= PARALLEL_JOBS )); do
    wait -n
  done
}

wait_all_jobs() {
  local status=0
  while (( $(jobs -rp | wc -l) > 0 )); do
    if ! wait -n; then
      status=1
    fi
  done
  return "${status}"
}

run_split() {
  local method="$1"
  local i="$2"
  local ckpt
  ckpt="$(ckpt_path_for "${method}" "${i}")"
  if [[ -s "${ckpt}" ]]; then
    log "SKIP ${method} split ${i}/${NUM_SEEDS}; checkpoint exists"
    return 0
  fi
  log "START ${method} split ${i}/${NUM_SEEDS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py \
    --method "${method}" \
    --save_path "${RUN_ROOT}" \
    --layout "${LAYOUT}" \
    --seed "${SEED}" \
    --split_seed_index "${i}" \
    --split_seed_count "${NUM_SEEDS}" \
    --output_vmap_index "${i}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --coma_num_steps "${COMA_NUM_STEPS}" \
    --wandb_mode "${WANDB_MODE}" \
    --project "${PROJECT}" \
    > "${LOG_DIR}/${method}_vmap${i}.parallel.log" 2>&1
  log "DONE ${method} split ${i}/${NUM_SEEDS}"
}

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  [[ -n "${method}" ]] || continue
  log "PARALLEL START ${method} layout=${LAYOUT} seed=${SEED} split_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS} jobs=${PARALLEL_JOBS}"
  for i in $(seq 0 $((NUM_SEEDS - 1))); do
    wait_for_slot
    run_split "${method}" "${i}" &
  done
  wait_all_jobs
  log "PARALLEL DONE ${method}"

  if [[ "${RUN_SPXP}" == "1" ]]; then
    out_dir="reports/classic_marl_ov2_partners_${method}_spxp_${TS}"
    log "START SP/XP ${method} output=${out_dir}"
    "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning_object_sp_xp.py \
      --run_root "${RUN_ROOT}" \
      --method "${method}" \
      --layout "${LAYOUT}" \
      --output_dir "${out_dir}" \
      --num_eval_seeds "${SPXP_EVAL_SEEDS}" \
      --action_mode "${ACTION_MODE}" \
      > "${LOG_DIR}/${method}_spxp.parallel.log" 2>&1
    log "DONE SP/XP ${method} output=${out_dir}"
  fi
done

find "${RUN_ROOT}" -maxdepth 4 -type f | sort > "${RUN_ROOT}/files.txt"
log "PARALLEL ALL_DONE run_root=${RUN_ROOT}"
