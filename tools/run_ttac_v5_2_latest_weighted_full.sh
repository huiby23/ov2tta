#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
cd "${ROOT}"

export MODE="${MODE:-ttac_v5_2_latest}"
export OUTPUT_TAG="${OUTPUT_TAG:-v5_2_latest_weighted_estimator_full_500_seedchunk100_20260623}"
export REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623}"
export LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_latest_weighted_full_500_seedchunk100_20260623}"

export EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
export SEED_CHUNKS="${SEED_CHUNKS:-5}"
export SEED_CHUNK_STRIDE="${SEED_CHUNK_STRIDE:-1000}"
export TOTAL_PAIRINGS="${TOTAL_PAIRINGS:-100}"
export SHARD_SIZE="${SHARD_SIZE:-10}"
export PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
export EVAL_BATCHES="${EVAL_BATCHES:-10}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RESET_QUEUE="${RESET_QUEUE:-0}"

export ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"

bash tools/run_ttac_v5_4_full_eval_sharded.sh
