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
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT="${PROJECT:-ov2_qlearning_1zsc}"
RUN_ROOT="${RUN_ROOT:-runs/qlearning_ov2_1zsc_qmix_shaq_${TS}}"
LOG_DIR="${LOG_DIR:-logs/qlearning_ov2_1zsc_qmix_shaq_${TS}}"
EGO_RUN_DIR="${EGO_RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
BASE_Q_REPORT="${BASE_Q_REPORT:-reports/mixed_1zsc_100seed_20260630_111256}"
EVAL_SEEDS="${EVAL_SEEDS:-100}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/JaxMARL:${ROOT}/experiments:${PYTHONPATH:-}"
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
layout=${LAYOUT}
wandb_mode=${WANDB_MODE}
project=${PROJECT}
algorithms=qmix_rnn,shaq_ps
ego_run_dir=${EGO_RUN_DIR}
base_q_report=${BASE_Q_REPORT}
eval_seeds=${EVAL_SEEDS}
EOF

log "START QMIX RNN OV2 layout=${LAYOUT} seed=${SEED} split_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
for i in $(seq 0 $((NUM_SEEDS - 1))); do
  log "START QMIX split ${i}/${NUM_SEEDS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/train_qmix_ov2.py \
    --save_path "${RUN_ROOT}/qmix_rnn" \
    --layout "${LAYOUT}" \
    --seed "${SEED}" \
    --num_seeds 1 \
    --split_seed_index "${i}" \
    --split_seed_count "${NUM_SEEDS}" \
    --output_vmap_index "${i}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --test_during_training 0 \
    --wandb_mode "${WANDB_MODE}" \
    --project "${PROJECT}" \
    > "${LOG_DIR}/qmix_rnn_vmap${i}.log" 2>&1
  log "DONE QMIX split ${i}/${NUM_SEEDS}"
done
log "DONE QMIX RNN"

log "START SHAQ OV2 layout=${LAYOUT} seed=${SEED} split_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
for i in $(seq 0 $((NUM_SEEDS - 1))); do
  log "START SHAQ split ${i}/${NUM_SEEDS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/train_shaq_ov2.py \
    --save_path "${RUN_ROOT}/shaq_ps" \
    --layout "${LAYOUT}" \
    --seed "${SEED}" \
    --num_seeds 1 \
    --split_seed_index "${i}" \
    --split_seed_count "${NUM_SEEDS}" \
    --output_vmap_index "${i}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --test_interval 1000000000 \
    --wandb_mode "${WANDB_MODE}" \
    --project "${PROJECT}" \
    > "${LOG_DIR}/shaq_ps_vmap${i}.log" 2>&1
  log "DONE SHAQ split ${i}/${NUM_SEEDS}"
done
log "DONE SHAQ"

EXT_REPORT="reports/mixed_1zsc_qmix_shaq_100seed_${TS}"
log "START MIXED 1-ZSC EVAL qmix_rnn,shaq_ps output=${EXT_REPORT}"
"${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/evaluate_mixed_1zsc.py \
  --ego_run_dir "${EGO_RUN_DIR}" \
  --q_run_root "${RUN_ROOT}" \
  --methods qmix_rnn,shaq_ps \
  --ego_modes base_no_test_adapt,ttac_v5_8_logit_bias \
  --max_ego_policies 10 \
  --max_partner_policies 10 \
  --num_eval_seeds "${EVAL_SEEDS}" \
  --output_dir "${EXT_REPORT}" \
  > "${LOG_DIR}/mixed_eval_qmix_shaq.log" 2>&1
log "DONE MIXED 1-ZSC EVAL qmix_rnn,shaq_ps"

COMBINED_REPORT="reports/mixed_1zsc_5q_100seed_${TS}"
log "START COMBINE base_q=${BASE_Q_REPORT} ext=${EXT_REPORT} output=${COMBINED_REPORT}"
"${PYTHON}" experiments/overcooked_v2_experiments/qlearning/utils/combine_mixed_1zsc_reports.py \
  --input_dirs "${BASE_Q_REPORT}" "${EXT_REPORT}" \
  --output_dir "${COMBINED_REPORT}" \
  > "${LOG_DIR}/combine_5q.log" 2>&1
log "DONE COMBINE output=${COMBINED_REPORT}"

find "${RUN_ROOT}" -maxdepth 4 -type f | sort > "${RUN_ROOT}/files.txt"
log "ALL_DONE run_root=${RUN_ROOT} report=${COMBINED_REPORT}"
