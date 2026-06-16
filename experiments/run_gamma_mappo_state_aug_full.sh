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

SOURCE_BACKEND="${SOURCE_BACKEND:-mappo}"
SOURCE_RUN_FILE="${SOURCE_RUN_FILE:-runs_by_config/counter_circuit_full_obs_64env_16mb_10M/mappo/rnn_state_aug/mappo_rnn_state_aug_64_16/RUN_PATH.txt}"
SOURCE_RUN="${SOURCE_RUN:-}"
if [[ -z "${SOURCE_RUN}" ]]; then
  SOURCE_RUN="$(cat "${SOURCE_RUN_FILE}")"
fi
if [[ ! -d "${SOURCE_RUN}" ]]; then
  echo "Missing SOURCE_RUN directory: ${SOURCE_RUN}" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/gamma_mappo_state_aug_full_${TS}}"
DATASET="${RUN_ROOT}/dataset/gamma_mappo_state_aug_population.npz"
VAE_DIR="${RUN_ROOT}/vae"
DIAG_DIR="${RUN_ROOT}/diagnostics"
LOG_DIR="${ROOT}/logs/gamma"
mkdir -p "${RUN_ROOT}/dataset" "${VAE_DIR}" "${DIAG_DIR}" "${LOG_DIR}"

print_repro_env

echo "[GAMMA-MAPPO-SA] Source backend: ${SOURCE_BACKEND}"
echo "[GAMMA-MAPPO-SA] Source run: ${SOURCE_RUN}"
echo "[GAMMA-MAPPO-SA] Stage 1: collect high-quality MAPPO state-aug trajectories"
"${PYTHON}" -m overcooked_v2_experiments.gamma.collect_dataset \
  --backend "${SOURCE_BACKEND}" \
  --run-dir "${SOURCE_RUN}" \
  --output "${DATASET}" \
  --seed "${SEED}" \
  --num-episodes "${GAMMA_DATA_EPISODES:-1024}" \
  --max-pairs "${GAMMA_DATA_MAX_PAIRS:-100}" \
  --cross \
  --validation-ratio 0.1

"${PYTHON}" - <<PY
import json, re, numpy as np, pathlib
log_path = pathlib.Path('${LOG_DIR}/gamma_mappo_state_aug_collect_${TS}.tmp')
print('[GAMMA-MAPPO-SA] Dataset saved:', '${DATASET}')
PY

echo "[GAMMA-MAPPO-SA] Stage 2: train GAMMA VAE on MAPPO state-aug trajectories"
"${PYTHON}" -m overcooked_v2_experiments.gamma.train_vae \
  --dataset "${DATASET}" \
  --output-dir "${VAE_DIR}" \
  --seed "${SEED}" \
  --z-dim "${GAMMA_Z_DIM:-32}" \
  --hidden-dim "${GAMMA_HIDDEN_DIM:-128}" \
  --chunk-length "${GAMMA_CHUNK_LENGTH:-100}" \
  --batch-size "${GAMMA_VAE_BATCH_SIZE:-64}" \
  --epochs "${GAMMA_VAE_EPOCHS:-80}" \
  --steps-per-epoch "${GAMMA_VAE_STEPS_PER_EPOCH:-100}" \
  --lr "${GAMMA_VAE_LR:-0.001}" \
  --kl-coef "${GAMMA_VAE_KL_COEF:-0.1}"

echo "[GAMMA-MAPPO-SA] Stage 2.5: generated partner diagnostic against source MAPPO"
"${PYTHON}" -m overcooked_v2_experiments.gamma.eval_generated_partner \
  --backend "${SOURCE_BACKEND}" \
  --run-dir "${SOURCE_RUN}" \
  --vae-checkpoint "${VAE_DIR}/gamma_vae.pkl" \
  --seed "${SEED}" \
  --num-seeds "${GAMMA_DIAG_NUM_SEEDS:-10}" \
  --max-runs "${GAMMA_DIAG_MAX_RUNS:-10}" \
  --output "${DIAG_DIR}/generated_partner_vs_source.json"

echo "[GAMMA-MAPPO-SA] Stage 3: train PPO coordinator against MAPPO-trained GAMMA generated partners"
"${PYTHON}" -m overcooked_v2_experiments.gamma.main \
  env=counter_circuit \
  SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" NUM_CHECKPOINTS=3 VISUALIZE=False \
  OPTIONAL_PREFIX="gamma_mappo_state_aug_data_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_${TS}" \
  wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE="${WANDB_MODE}" \
  GAMMA.VAE_CHECKPOINT="${VAE_DIR}/gamma_vae.pkl" \
  model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  model.NUM_ENVS="${NUM_ENVS}" \
  model.NUM_STEPS="${NUM_STEPS}" \
  model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
  model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"

echo "[GAMMA-MAPPO-SA] Pipeline finished. RUN_ROOT=${RUN_ROOT}"
