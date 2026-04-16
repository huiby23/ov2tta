#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
LAYOUT="${LAYOUT:-counter_circuit}"
EXPERIMENT="${EXPERIMENT:-cnn}"
ENV_CONFIG="${ENV_CONFIG:-default}"
PREFIX="${PREFIX:-figure4_e3t_ppo_auto}"

PARTNER_MIX_MODE="${PARTNER_MIX_MODE:-ppo_consistent}"
PARTNER_MIX_EPS="${PARTNER_MIX_EPS:-0.55}"
PARTNER_COPY_COEF="${PARTNER_COPY_COEF:-0.1}"
USE_PARTNER_MIX="${USE_PARTNER_MIX:-True}"
USE_HISTORY_CONTEXT="${USE_HISTORY_CONTEXT:-True}"
SEPARATE_MOA_UPDATE="${SEPARATE_MOA_UPDATE:-True}"
OFFICIAL_SEPARATE_PARAM_SPLIT="${OFFICIAL_SEPARATE_PARAM_SPLIT:-True}"
CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-}"

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_run() {
  local -a args=(
    +experiment="${EXPERIMENT}"
    +env="${ENV_CONFIG}"
    +OPTIONAL_PREFIX="${PREFIX}"
    wandb.ENTITY="${ENTITY}"
    wandb.PROJECT="${PROJECT}"
    wandb.WANDB_MODE="${WANDB_MODE}"
    +env.ENV_KWARGS.layout="${LAYOUT}"
    SEED=42
    NUM_SEEDS="${NUM_SEEDS}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}"
    model.USE_PARTNER_MIX="${USE_PARTNER_MIX}"
    model.PARTNER_MIX_MODE="${PARTNER_MIX_MODE}"
    model.PARTNER_MIX_EPS="${PARTNER_MIX_EPS}"
    model.PARTNER_COPY_COEF="${PARTNER_COPY_COEF}"
    model.USE_HISTORY_CONTEXT="${USE_HISTORY_CONTEXT}"
    model.SEPARATE_MOA_UPDATE="${SEPARATE_MOA_UPDATE}"
    model.OFFICIAL_SEPARATE_PARAM_SPLIT="${OFFICIAL_SEPARATE_PARAM_SPLIT}"
  )

  if [[ -n "${CONTEXT_UPDATE_EPOCHS}" ]]; then
    args+=("model.CONTEXT_UPDATE_EPOCHS=${CONTEXT_UPDATE_EPOCHS}")
  fi

  PYTHONPATH=experiments python -m overcooked_v2_experiments.e3t_ppo.main "${args[@]}"
}

evaluate_cross_play() {
  local run_dir="$1"
  python experiments/eval/cross_play_from_dir.py \
    --run_dir "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-E3T-PPO] Training run"
train_run
RUN_DIR="$(latest_run_dir "${PREFIX}")"
echo "[Figure4-E3T-PPO] Run dir: ${RUN_DIR}"
echo "[Figure4-E3T-PPO] Starting figure4 cross-play evaluation"
evaluate_cross_play "${RUN_DIR}"
echo "[Figure4-E3T-PPO] Done"
echo "[Figure4-E3T-PPO] Matrix: ${RUN_DIR}/reward_summary_cross_plot.png"
