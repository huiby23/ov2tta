#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-online}"
STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_e3t_standard}"

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_standard() {
  PYTHONPATH=experiments python experiments/overcooked_v2_experiments/e3t/main.py \
    +experiment=rnn-sp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +OPTIONAL_PREFIX="${STANDARD_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}"
}

evaluate_cross_play() {
  local run_dir="$1"
  PYTHONPATH=experiments python experiments/overcooked_v2_experiments/e3t/utils/visualize.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-E3T] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4-E3T] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"
echo "[Figure4-E3T] Done"
