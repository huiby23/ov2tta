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
AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
GPU_BUSY_THRESHOLD_MB="${GPU_BUSY_THRESHOLD_MB:-1024}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
STANDARD_PREFIX="${STANDARD_PREFIX:-figure4_e3t_paper_budget}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
MODEL_NUM_ENVS="${MODEL_NUM_ENVS:-64}"
MODEL_NUM_STEPS="${MODEL_NUM_STEPS:-256}"
MODEL_NUM_MINIBATCHES="${MODEL_NUM_MINIBATCHES:-16}"
MODEL_UPDATE_EPOCHS="${MODEL_UPDATE_EPOCHS:-4}"

select_available_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local line idx used
  local -a free_gpus=()
  local least_used_idx=""
  local least_used_mem=999999

  while IFS=, read -r idx used; do
    idx="${idx// /}"
    used="${used// /}"
    [[ -z "${idx}" || -z "${used}" ]] && continue
    if (( used <= GPU_BUSY_THRESHOLD_MB )); then
      free_gpus+=("${idx}")
    fi
    if (( used < least_used_mem )); then
      least_used_mem="${used}"
      least_used_idx="${idx}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if (( ${#free_gpus[@]} > 0 )); then
    local IFS=,
    echo "${free_gpus[*]}"
    return 0
  fi

  if [[ -n "${least_used_idx}" ]]; then
    echo "${least_used_idx}"
    return 0
  fi

  return 1
}

if [[ -z "${CUDA_VISIBLE_DEVICES}" && "${AUTO_SELECT_GPUS}" == "1" ]]; then
  if CUDA_VISIBLE_DEVICES="$(select_available_gpus)"; then
    export CUDA_VISIBLE_DEVICES
    echo "[Figure4-E3T-PaperBudget] Auto-selected GPUs: ${CUDA_VISIBLE_DEVICES}"
  fi
fi

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_standard() {
  local -a env_prefix=(PYTHONUNBUFFERED=1 E3T_SKIP_WANDB_FINISH=1 E3T_FORCE_OS_EXIT=1 PYTHONPATH=experiments)
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    env_prefix+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi
  env "${env_prefix[@]}" "$PYTHON_BIN" experiments/overcooked_v2_experiments/e3t/main.py \
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
  local -a env_prefix=(E3T_SKIP_WANDB_FINISH=1 E3T_FORCE_OS_EXIT=1 PYTHONPATH=experiments)
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    env_prefix+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi
  env "${env_prefix[@]}" "$PYTHON_BIN" experiments/overcooked_v2_experiments/e3t/utils/visualize.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
}

echo "[Figure4-E3T-PaperBudget] Training standard self-play run"
train_standard
STANDARD_RUN_DIR="$(latest_run_dir "${STANDARD_PREFIX}")"
echo "[Figure4-E3T-PaperBudget] Standard run dir: ${STANDARD_RUN_DIR}"
evaluate_cross_play "${STANDARD_RUN_DIR}"
echo "[Figure4-E3T-PaperBudget] Done"
