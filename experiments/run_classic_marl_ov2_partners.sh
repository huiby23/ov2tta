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

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/JaxMARL:${ROOT}/JaxMARL/baselines/QLearning:${ROOT}/experiments:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

cat > "${RUN_ROOT}/manifest.txt" <<EOF
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
EOF

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  [[ -n "${method}" ]] || continue
  log "START ${method} OV2 layout=${LAYOUT} seed=${SEED} split_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
  for i in $(seq 0 $((NUM_SEEDS - 1))); do
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
      > "${LOG_DIR}/${method}_vmap${i}.log" 2>&1
    log "DONE ${method} split ${i}/${NUM_SEEDS}"
  done
  log "DONE ${method}"

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
      > "${LOG_DIR}/${method}_spxp.log" 2>&1
    log "DONE SP/XP ${method} output=${out_dir}"
  fi
done

find "${RUN_ROOT}" -maxdepth 4 -type f | sort > "${RUN_ROOT}/files.txt"
log "ALL_DONE run_root=${RUN_ROOT}"
