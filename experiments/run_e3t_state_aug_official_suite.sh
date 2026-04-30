#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-overcooked_v2_tta}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

run_train() {
  local prefix="$1"
  local total_timesteps="$2"
  local num_steps="$3"
  local num_envs="$4"
  local num_iterations="$5"

  "${PYTHON}" experiments/overcooked_v2_experiments/e3t_state_aug_official/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS=1 \
    VISUALIZE=False \
    +NUM_ITERATIONS="${num_iterations}" \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${total_timesteps}" \
    model.NUM_STEPS="${num_steps}" \
    model.NUM_ENVS="${num_envs}"
}

case "${1:-full}" in
  smoke)
    run_train "smoke_e3t_state_aug_official" 76800 64 12 1
    ;;
  full)
    run_train "figure4_e3t_state_aug_official_seed${SEED}" 4.8e6 400 30 10
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac
