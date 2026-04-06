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
STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_e3t_cec_budget}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000000000}"
MODEL_NUM_ENVS="${MODEL_NUM_ENVS:-512}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-100}"
MODEL_NUM_MINIBATCHES="${MODEL_NUM_MINIBATCHES:-2}"
MODEL_UPDATE_EPOCHS="${MODEL_UPDATE_EPOCHS:-4}"

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_standard() {
  local -a env_prefix=(PYTHONUNBUFFERED=1 E3T_CEC_SKIP_WANDB_FINISH=1 E3T_CEC_FORCE_OS_EXIT=1 PYTHONPATH=experiments)
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    env_prefix+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi
  env "${env_prefix[@]}" "$PYTHON_BIN" experiments/overcooked_v2_experiments/e3t_cec/main.py \
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
    model.MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    model.NUM_ENVS="${MODEL_NUM_ENVS}" \
    model.NUM_STEPS="${MODEL_NUM_STEPS}" \
    model.NUM_MINIBATCHES="${MODEL_NUM_MINIBATCHES}" \
    model.UPDATE_EPOCHS="${MODEL_UPDATE_EPOCHS}"
}

evaluate_cross_play() {
  local run_dir="$1"
  local -a env_prefix=(E3T_CEC_SKIP_WANDB_FINISH=1 E3T_CEC_FORCE_OS_EXIT=1 PYTHONPATH=experiments)
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    env_prefix+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi
  env "${env_prefix[@]}" "$PYTHON_BIN" experiments/overcooked_v2_experiments/e3t_cec/utils/visualize.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-E3T-CECBudget] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4-E3T-CECBudget] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"
echo "[Figure4-E3T-CECBudget] Done"
