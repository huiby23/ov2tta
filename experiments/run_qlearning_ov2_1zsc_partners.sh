#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-offline}"
PROJECT="${PROJECT:-ov2_qlearning_1zsc}"
RUN_ROOT="${RUN_ROOT:-runs/qlearning_ov2_1zsc_${TS}}"
LOG_DIR="${LOG_DIR:-logs/qlearning_ov2_1zsc_${TS}}"
cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/JaxMARL:${ROOT}/experiments:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

run_iql() {
  log "START IQL CNN OV2 layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
  "${PYTHON}" JaxMARL/baselines/QLearning/iql_cnn_overcooked.py \
    +alg=ql_cnn_overcooked \
    NUM_SEEDS="${NUM_SEEDS}" SEED="${SEED}" WANDB_MODE="${WANDB_MODE}" PROJECT="${PROJECT}" \
    SAVE_PATH="${RUN_ROOT}/iql" \
    alg.ENV_NAME=overcooked_v2 alg.ENV_KWARGS.layout="${LAYOUT}" \
    alg.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" alg.NUM_ENVS=64 alg.NUM_STEPS=1 \
    alg.BUFFER_SIZE=100000 alg.BUFFER_BATCH_SIZE=128 alg.LEARNING_STARTS=1000 \
    alg.NUM_EPOCHS=4 alg.TEST_INTERVAL=0.05 alg.TEST_NUM_ENVS=256 \
    > "${LOG_DIR}/iql.log" 2>&1
  log "DONE IQL"
}

run_vdn() {
  log "START VDN CNN OV2 layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
  "${PYTHON}" JaxMARL/baselines/QLearning/vdn_cnn_overcooked.py \
    +alg=ql_cnn_overcooked \
    NUM_SEEDS="${NUM_SEEDS}" SEED="${SEED}" WANDB_MODE="${WANDB_MODE}" PROJECT="${PROJECT}" \
    SAVE_PATH="${RUN_ROOT}/vdn" \
    alg.ENV_NAME=overcooked_v2 alg.ENV_KWARGS.layout="${LAYOUT}" \
    alg.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" alg.NUM_ENVS=64 alg.NUM_STEPS=1 \
    alg.BUFFER_SIZE=100000 alg.BUFFER_BATCH_SIZE=128 alg.LEARNING_STARTS=1000 \
    alg.NUM_EPOCHS=4 alg.TEST_INTERVAL=0.05 alg.TEST_NUM_ENVS=256 \
    > "${LOG_DIR}/vdn.log" 2>&1
  log "DONE VDN"
}

run_pqn() {
  log "START PQN-VDN CNN OV2 layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS}"
  "${PYTHON}" JaxMARL/baselines/QLearning/pqn_vdn_cnn_overcooked.py \
    +alg=pqn_vdn_cnn_overcooked \
    NUM_SEEDS="${NUM_SEEDS}" SEED="${SEED}" WANDB_MODE="${WANDB_MODE}" PROJECT="${PROJECT}" \
    SAVE_PATH="${RUN_ROOT}/pqn_vdn" \
    alg.ENV_NAME=overcooked_v2 alg.ENV_KWARGS.layout="${LAYOUT}" \
    alg.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" alg.NUM_ENVS=64 alg.NUM_STEPS=16 \
    alg.NUM_MINIBATCHES=16 alg.NUM_EPOCHS=4 alg.TEST_INTERVAL=0.05 alg.TEST_NUM_ENVS=256 \
    > "${LOG_DIR}/pqn_vdn.log" 2>&1
  log "DONE PQN-VDN"
}

cat > "${RUN_ROOT}/manifest.txt" <<EOF
run_root=${RUN_ROOT}
log_dir=${LOG_DIR}
seed=${SEED}
num_seeds=${NUM_SEEDS}
total_timesteps=${TOTAL_TIMESTEPS}
layout=${LAYOUT}
wandb_mode=${WANDB_MODE}
project=${PROJECT}
algorithms=iql,vdn,pqn_vdn
EOF

run_iql
run_vdn
run_pqn
log "ALL_DONE run_root=${RUN_ROOT}"
find "${RUN_ROOT}" -maxdepth 4 -type f | sort > "${RUN_ROOT}/files.txt"
