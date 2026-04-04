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

STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_mappo_standard}"

TRAIN_OVERRIDES=()
if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}")
fi
if [[ -n "${REW_SHAPING_HORIZON:-}" ]]; then
  TRAIN_OVERRIDES+=("model.REW_SHAPING_HORIZON=${REW_SHAPING_HORIZON}")
fi
if [[ -n "${MODEL_NUM_ENVS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_ENVS=${MODEL_NUM_ENVS}")
fi
if [[ -n "${MODEL_NUM_STEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_STEPS=${MODEL_NUM_STEPS}")
fi
if [[ -n "${MODEL_NUM_MINIBATCHES:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_MINIBATCHES=${MODEL_NUM_MINIBATCHES}")
fi
if [[ -n "${MODEL_UPDATE_EPOCHS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.UPDATE_EPOCHS=${MODEL_UPDATE_EPOCHS}")
fi

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_standard() {
  PYTHONPATH=experiments python experiments/overcooked_v2_experiments/ppo/mappo_main.py \
    +experiment=rnn-sp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +OPTIONAL_PREFIX="${STANDARD_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    "${TRAIN_OVERRIDES[@]}"
}

evaluate_cross_play() {
  local run_dir="$1"
  PYTHONPATH=experiments python experiments/overcooked_v2_experiments/ppo/utils/visualize_mappo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-MAPPO] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4-MAPPO] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"

echo "[Figure4-MAPPO] Done"
echo "[Figure4-MAPPO] Standard matrix: ${STANDARD_RUN_DIR}/reward_summary_cross_plot.png"
