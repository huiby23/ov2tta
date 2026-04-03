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

STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_standard}"
STATE_AUG_PREFIX="${STATE_AUG_PREFIX:-figure4_state_aug}"
STATE_AUG_ITERATIONS="${STATE_AUG_ITERATIONS:-10}"

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
  python experiments/overcooked_v2_experiments/ppo/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +OPTIONAL_PREFIX="${STANDARD_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

train_state_aug() {
  python experiments/overcooked_v2_experiments/ppo/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +NUM_ITERATIONS="${STATE_AUG_ITERATIONS}" \
    +OPTIONAL_PREFIX="${STATE_AUG_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

evaluate_cross_play() {
  local run_dir="$1"
  python experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"

echo "[Figure4] Training state-augmented run"
train_state_aug
STATE_AUG_RUN_DIR="$(latest_run_dir "${STATE_AUG_PREFIX}")"
echo "[Figure4] State-augmented run dir: ${STATE_AUG_RUN_DIR}"
evaluate_cross_play "${STATE_AUG_RUN_DIR}"

echo "[Figure4] Done"
echo "[Figure4] Standard matrix: ${STANDARD_RUN_DIR}/reward_summary_cross_plot.png"
echo "[Figure4] State-augmented matrix: ${STATE_AUG_RUN_DIR}/reward_summary_cross_plot.png"
