#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
source "${ROOT}/experiments/repro_env.sh"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${ROOT}/experiments/overcooked_v2_experiments/ppo:${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"

SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-runs/gamma_mappo_state_aug_full_20260513-154316/vae/gamma_vae.pkl}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT}/logs/gamma"
mkdir -p "${LOG_DIR}"

print_repro_env

echo "[GAMMA-MIX] VAE_CHECKPOINT=${VAE_CHECKPOINT}"

declare -a MIX_PROBS=("0.25" "0.50")
for MIX_PROB in "${MIX_PROBS[@]}"; do
  MIX_TAG="mix${MIX_PROB/./}"
  PREFIX="gamma_${MIX_TAG}_sp_anchor_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_${TS}"
  echo "[GAMMA-MIX] Training ${MIX_TAG} with POPULATION_MIX_PROB=${MIX_PROB}"
  "${PYTHON}" -m overcooked_v2_experiments.gamma.main \
    env=counter_circuit \
    SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" NUM_CHECKPOINTS=3 VISUALIZE=False \
    OPTIONAL_PREFIX="${PREFIX}" \
    wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE="${WANDB_MODE}" \
    GAMMA.VAE_CHECKPOINT="${VAE_CHECKPOINT}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    +POPULATION_MIX_PROB="${MIX_PROB}"

  RUN_DIR="$(find runs -maxdepth 2 -type d -name "${PREFIX}_gamma_CNN_ov2_counter_circuit_avs-full" | sort | tail -n 1)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[GAMMA-MIX] Could not locate run dir for ${PREFIX}" >&2
    exit 1
  fi
  echo "[GAMMA-MIX] Evaluating ${RUN_DIR}"
  "${PYTHON}" -m overcooked_v2_experiments.ppo.utils.visualize_ppo \
    --d "${RUN_DIR}" --seed "${SEED}" --num_seeds "${EVAL_SEEDS}" --all --no_viz
  echo "[GAMMA-MIX] Finished ${MIX_TAG}. RUN_DIR=${RUN_DIR}"
done

echo "[GAMMA-MIX] All experiments finished. TS=${TS}"
