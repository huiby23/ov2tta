#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/logs/ttac_v2_remaining_${TS}"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"
export MODE="${MODE:-full}"
export WANDB_MODE="${WANDB_MODE:-online}"
export EVAL_BATCHES="${EVAL_BATCHES:-10}"
export EVAL_SEEDS="${EVAL_SEEDS:-500}"
export NUM_ENVS="${NUM_ENVS:-64}"
export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
export NUM_SEEDS="${NUM_SEEDS:-10}"
export SEED="${SEED:-42}"
log(){ echo "[$(date "+%F %T")] $*" | tee -a "${LOG_DIR}/queue.log"; }
run_stage(){ local name="$1"; shift; log "START ${name}"; "$@" > "${LOG_DIR}/${name}.log" 2>&1; log "DONE ${name}"; }
log "SKIP remaining selfplay wrong/off eval; continue downstream TTACv2 frameworks"
run_stage ttac_v2_frozen_population experiments/run_ttac_v2_frozen_population_64_16.sh
run_stage ttac_v2_sync_population experiments/run_ttac_v2_sync_population_64_16.sh
run_stage gamma_ttac experiments/run_gamma_ttac_mappo_state_aug_full.sh
log "ALL_DONE"
