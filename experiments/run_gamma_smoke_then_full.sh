#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
source "${ROOT}/experiments/repro_env.sh"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${ROOT}/experiments/overcooked_v2_experiments/ppo:${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"

LAYOUT="${LAYOUT:-counter_circuit}"
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

SOURCE_RUN_FILE="${SOURCE_RUN_FILE:-runs_by_config/counter_circuit_full_obs_64env_16mb_10M/ppo/cnn_standard/ppo_cnn_standard_64_16/RUN_PATH.txt}"
SOURCE_RUN="${SOURCE_RUN:-}"
if [[ -z "${SOURCE_RUN}" ]]; then
  SOURCE_RUN="$(cat "${SOURCE_RUN_FILE}")"
fi
if [[ ! -d "${SOURCE_RUN}" ]]; then
  echo "Missing SOURCE_RUN directory: ${SOURCE_RUN}" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/gamma_full_${TS}}"
SMOKE_DIR="${RUN_ROOT}/smoke"
DATASET="${RUN_ROOT}/dataset/gamma_ov2_population.npz"
VAE_DIR="${RUN_ROOT}/vae"
LOG_DIR="${ROOT}/logs/gamma"
mkdir -p "${SMOKE_DIR}" "${RUN_ROOT}/dataset" "${VAE_DIR}" "${LOG_DIR}"

print_repro_env

echo "[GAMMA] Stage 0: smoke dataset + VAE"
"${PYTHON}" -m overcooked_v2_experiments.gamma.make_dummy_dataset \
  --output "${SMOKE_DIR}/dummy_counter_circuit.npz" \
  --episodes 8 --steps 32 --height 5 --width 8 --channels 30 --seed "${SEED}"
"${PYTHON}" -m overcooked_v2_experiments.gamma.train_vae \
  --dataset "${SMOKE_DIR}/dummy_counter_circuit.npz" \
  --output-dir "${SMOKE_DIR}/vae" \
  --epochs 1 --steps-per-epoch 1 --batch-size 2 --chunk-length 4 \
  --hidden-dim 16 --z-dim 8 --seed "${SEED}"

echo "[GAMMA] Stage 0: smoke PPO training with generated partner"
"${PYTHON}" -m overcooked_v2_experiments.gamma.main \
  env=counter_circuit \
  SEED="${SEED}" NUM_SEEDS=1 NUM_CHECKPOINTS=1 VISUALIZE=False \
  OPTIONAL_PREFIX="gamma_smoke_${TS}" \
  wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE=offline \
  GAMMA.VAE_CHECKPOINT="${SMOKE_DIR}/vae/gamma_vae.pkl" \
  model.TOTAL_TIMESTEPS=512 model.REW_SHAPING_HORIZON=256 \
  model.NUM_ENVS=2 model.NUM_STEPS=16 model.UPDATE_EPOCHS=1 model.NUM_MINIBATCHES=1 model.FC_DIM_SIZE=16

echo "[GAMMA] Stage 1: collect real OV2 population trajectories from ${SOURCE_RUN}"
"${PYTHON}" -m overcooked_v2_experiments.gamma.collect_dataset \
  --run-dir "${SOURCE_RUN}" \
  --output "${DATASET}" \
  --seed "${SEED}" \
  --num-episodes "${GAMMA_DATA_EPISODES:-256}" \
  --max-pairs "${GAMMA_DATA_MAX_PAIRS:-64}" \
  --cross \
  --validation-ratio 0.1

echo "[GAMMA] Stage 2: train full GAMMA VAE"
"${PYTHON}" -m overcooked_v2_experiments.gamma.train_vae \
  --dataset "${DATASET}" \
  --output-dir "${VAE_DIR}" \
  --seed "${SEED}" \
  --z-dim "${GAMMA_Z_DIM:-16}" \
  --hidden-dim "${GAMMA_HIDDEN_DIM:-64}" \
  --chunk-length "${GAMMA_CHUNK_LENGTH:-100}" \
  --batch-size "${GAMMA_VAE_BATCH_SIZE:-64}" \
  --epochs "${GAMMA_VAE_EPOCHS:-50}" \
  --steps-per-epoch "${GAMMA_VAE_STEPS_PER_EPOCH:-100}" \
  --lr "${GAMMA_VAE_LR:-0.001}" \
  --kl-coef "${GAMMA_VAE_KL_COEF:-0.1}"

echo "[GAMMA] Stage 3: full PPO coordinator training against GAMMA generated partners"
"${PYTHON}" -m overcooked_v2_experiments.gamma.main \
  env=counter_circuit \
  SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" NUM_CHECKPOINTS=3 VISUALIZE=False \
  OPTIONAL_PREFIX="gamma_ppo_cnn_standard_data_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_${TS}" \
  wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE="${WANDB_MODE}" \
  GAMMA.VAE_CHECKPOINT="${VAE_DIR}/gamma_vae.pkl" \
  model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  model.NUM_ENVS="${NUM_ENVS}" \
  model.NUM_STEPS="${NUM_STEPS}" \
  model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
  model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"

echo "[GAMMA] Pipeline finished. RUN_ROOT=${RUN_ROOT}"
