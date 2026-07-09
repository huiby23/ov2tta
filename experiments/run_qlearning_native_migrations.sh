#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-16}"
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PROJECT="${PROJECT:-ov2_qlearning_native_migrations}"
METHODS="${METHODS:-pqn_wqmix,pqn_soft}"
RUN_ROOT="${RUN_ROOT:-runs/qlearning_native_migrations_${TS}}"
LOG_DIR="${LOG_DIR:-logs/qlearning_native_migrations_${TS}}"
RUN_SPXP="${RUN_SPXP:-1}"
SPXP_EVAL_SEEDS="${SPXP_EVAL_SEEDS:-100}"
ACTION_MODE="${ACTION_MODE:-greedy}"
WQMIX_ALPHA="${WQMIX_ALPHA:-0.5}"
SOFT_Q_TEMPERATURE="${SOFT_Q_TEMPERATURE:-0.25}"
SOFT_Q_CENTER="${SOFT_Q_CENTER:-1}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/JaxMARL:${ROOT}/JaxMARL/baselines/QLearning:${ROOT}/experiments:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

{
  echo "run_root=${RUN_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "seed=${SEED}"
  echo "num_seeds=${NUM_SEEDS}"
  echo "total_timesteps=${TOTAL_TIMESTEPS}"
  echo "num_envs=${NUM_ENVS}"
  echo "num_steps=${NUM_STEPS}"
  echo "layout=${LAYOUT}"
  echo "wandb_mode=${WANDB_MODE}"
  echo "project=${PROJECT}"
  echo "methods=${METHODS}"
  echo "run_spxp=${RUN_SPXP}"
  echo "spxp_eval_seeds=${SPXP_EVAL_SEEDS}"
  echo "action_mode=${ACTION_MODE}"
  echo "wqmix_alpha=${WQMIX_ALPHA}"
  echo "soft_q_temperature=${SOFT_Q_TEMPERATURE}"
  echo "soft_q_center=${SOFT_Q_CENTER}"
} > "${RUN_ROOT}/manifest.txt"

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  [[ -n "${method}" ]] || continue
  log "START ${method} OV2 layout=${LAYOUT} seed=${SEED} split_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
  for i in $(seq 0 $((NUM_SEEDS - 1))); do
    log "START ${method} split ${i}/${NUM_SEEDS}"
    "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/train_classic_partner_ov2.py       --method "${method}"       --save_path "${RUN_ROOT}"       --layout "${LAYOUT}"       --seed "${SEED}"       --split_seed_index "${i}"       --split_seed_count "${NUM_SEEDS}"       --output_vmap_index "${i}"       --total_timesteps "${TOTAL_TIMESTEPS}"       --num_envs "${NUM_ENVS}"       --num_steps "${NUM_STEPS}"       --num_minibatches 16       --pqn_hidden_size 512       --pqn_num_layers 2       --pqn_norm_type layer_norm       --pqn_norm_input 0       --pqn_eps_finish 0.2       --pqn_eps_decay 0.2       --pqn_max_grad_norm 10.0       --pqn_lambda 0.5       --lr 0.000075       --lr_linear_decay 1       --wqmix_alpha "${WQMIX_ALPHA}"       --soft_q_temperature "${SOFT_Q_TEMPERATURE}"       --soft_q_center "${SOFT_Q_CENTER}"       --wandb_mode "${WANDB_MODE}"       --project "${PROJECT}"       > "${LOG_DIR}/${method}_vmap${i}.log" 2>&1
    log "DONE ${method} split ${i}/${NUM_SEEDS}"
  done
  log "DONE ${method}"

  if [[ "${RUN_SPXP}" == "1" ]]; then
    out_dir="reports/qlearning_native_migrations_${method}_spxp_${TS}"
    log "START SP/XP ${method} output=${out_dir}"
    "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning_object_sp_xp.py       --run_root "${RUN_ROOT}"       --method "${method}"       --layout "${LAYOUT}"       --output_dir "${out_dir}"       --num_eval_seeds "${SPXP_EVAL_SEEDS}"       --action_mode "${ACTION_MODE}"       > "${LOG_DIR}/${method}_spxp.log" 2>&1
    log "DONE SP/XP ${method} output=${out_dir}"
  fi
done

find "${RUN_ROOT}" -maxdepth 4 -type f | sort > "${RUN_ROOT}/files.txt"
log "ALL_DONE run_root=${RUN_ROOT}"
