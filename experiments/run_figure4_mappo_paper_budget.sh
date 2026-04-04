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
WANDB_MODE="${WANDB_MODE:-offline}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"

STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_mappo_paper_budget}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
MODEL_NUM_ENVS="${MODEL_NUM_ENVS:-64}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-256}"
MODEL_NUM_MINIBATCHES="${MODEL_NUM_MINIBATCHES:-16}"
MODEL_UPDATE_EPOCHS="${MODEL_UPDATE_EPOCHS:-4}"

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_standard() {
  PYTHONUNBUFFERED=1 MAPPO_SKIP_WANDB_FINISH=1 MAPPO_FORCE_OS_EXIT=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=experiments "$PYTHON_BIN" experiments/overcooked_v2_experiments/mappo/main.py \
    +experiment=rnn-sp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +OPTIONAL_PREFIX="${STANDARD_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${MODEL_NUM_ENVS}" \
    model.NUM_STEPS="${MODEL_NUM_STEPS}" \
    model.NUM_MINIBATCHES="${MODEL_NUM_MINIBATCHES}" \
    model.UPDATE_EPOCHS="${MODEL_UPDATE_EPOCHS}"
}

evaluate_cross_play() {
  local run_dir="$1"
  MAPPO_SKIP_WANDB_FINISH=1 MAPPO_FORCE_OS_EXIT=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH=experiments "$PYTHON_BIN" experiments/overcooked_v2_experiments/mappo/utils/visualize.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-MAPPO-PaperBudget] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4-MAPPO-PaperBudget] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"

echo "[Figure4-MAPPO-PaperBudget] Done"
echo "[Figure4-MAPPO-PaperBudget] Standard matrix: ${STANDARD_RUN_DIR}/reward_summary_cross_plot.png"
