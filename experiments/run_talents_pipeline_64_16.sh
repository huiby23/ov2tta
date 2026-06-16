#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
MODE="${MODE:-smoke}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

if [ -f "${ROOT}/experiments/repro_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/experiments/repro_env.sh"
fi
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-10000000}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_MODE="${WANDB_MODE:-online}"
SOURCE_BACKEND="${SOURCE_BACKEND:-ppo}"
TALENTS_DATA_MIN_RETURN="${TALENTS_DATA_MIN_RETURN:-}"
TALENTS_DATA_TOP_EPISODE_FRACTION="${TALENTS_DATA_TOP_EPISODE_FRACTION:-1.0}"
TALENTS_DATA_MAX_ATTEMPTS="${TALENTS_DATA_MAX_ATTEMPTS:-}"
TALENTS_Z_SAMPLE_SCALE="${TALENTS_Z_SAMPLE_SCALE:-1.0}"
POPULATION_MIX_PROB="${POPULATION_MIX_PROB:-}"

DEFAULT_SOURCE_RUN_FILE="${ROOT}/runs_by_config/counter_circuit_full_obs_64env_16mb_10M/ppo/cnn_standard/ppo_cnn_standard_64_16/RUN_PATH.txt"
if [ -z "${SOURCE_RUN:-}" ] && [ -f "${DEFAULT_SOURCE_RUN_FILE}" ]; then
  SOURCE_RUN="$(cat "${DEFAULT_SOURCE_RUN_FILE}")"
fi

RUN_ROOT="${ROOT}/runs/talents_${MODE}_${TS}"
mkdir -p "${RUN_ROOT}"

echo "[TALENTS] mode=${MODE}"
echo "[TALENTS] run_root=${RUN_ROOT}"
echo "[TALENTS] python=${PYTHON}"

if [ "${MODE}" = "smoke" ]; then
  SMOKE_DIR="${RUN_ROOT}/smoke"
  mkdir -p "${SMOKE_DIR}"
  "${PYTHON}" -m overcooked_v2_experiments.gamma.make_dummy_dataset \
    --output "${SMOKE_DIR}/dummy.npz" \
    --episodes 8 --steps 60 --height 5 --width 5 --channels 26 --seed "${SEED}"
  "${PYTHON}" -m overcooked_v2_experiments.talents.train_vae \
    --dataset "${SMOKE_DIR}/dummy.npz" \
    --output-dir "${SMOKE_DIR}/vae" \
    --epochs 1 --steps-per-epoch 1 --batch-size 4 --hidden-dim 16 \
    --chunk-length 20 --z-dim 4 --seed "${SEED}"
  "${PYTHON}" -m overcooked_v2_experiments.talents.cluster_latents \
    --dataset "${SMOKE_DIR}/dummy.npz" \
    --vae-checkpoint "${SMOKE_DIR}/vae/gamma_vae.pkl" \
    --output-dir "${SMOKE_DIR}/clusters" \
    --num-clusters 2 --num-samples 16 --batch-size 4 --chunk-length 20 --seed "${SEED}"
  echo "[TALENTS] smoke artifacts: ${SMOKE_DIR}"
  exit 0
fi

if [ "${MODE}" != "full" ]; then
  echo "Unsupported MODE=${MODE}; use smoke or full" >&2
  exit 2
fi

if [ -z "${SOURCE_RUN:-}" ]; then
  echo "SOURCE_RUN is required for full mode, or create ${DEFAULT_SOURCE_RUN_FILE}" >&2
  exit 2
fi
if [ ! -d "${SOURCE_RUN}" ]; then
  echo "SOURCE_RUN does not exist: ${SOURCE_RUN}" >&2
  exit 2
fi

DATASET="${RUN_ROOT}/talents_teacher_dataset.npz"
VAE_DIR="${RUN_ROOT}/vae"
CLUSTER_DIR="${RUN_ROOT}/clusters"
mkdir -p "${VAE_DIR}" "${CLUSTER_DIR}"

echo "[TALENTS] source_run=${SOURCE_RUN}"
echo "[TALENTS] source_backend=${SOURCE_BACKEND}"
COLLECT_FILTER_ARGS=()
if [ -n "${TALENTS_DATA_MIN_RETURN}" ]; then
  COLLECT_FILTER_ARGS+=(--min-return "${TALENTS_DATA_MIN_RETURN}")
fi
if [ "${TALENTS_DATA_TOP_EPISODE_FRACTION}" != "1.0" ]; then
  COLLECT_FILTER_ARGS+=(--top-episode-fraction "${TALENTS_DATA_TOP_EPISODE_FRACTION}")
fi
if [ -n "${TALENTS_DATA_MAX_ATTEMPTS}" ]; then
  COLLECT_FILTER_ARGS+=(--max-attempts "${TALENTS_DATA_MAX_ATTEMPTS}")
fi
"${PYTHON}" -m overcooked_v2_experiments.talents.collect_dataset \
  --run-dir "${SOURCE_RUN}" \
  --output "${DATASET}" \
  --backend "${SOURCE_BACKEND}" \
  --seed "${SEED}" \
  --num-episodes "${TALENTS_DATA_EPISODES:-512}" \
  --max-pairs "${TALENTS_DATA_MAX_PAIRS:-90}" \
  --cross --validation-ratio 0.1 --greedy \
  "${COLLECT_FILTER_ARGS[@]}"

"${PYTHON}" -m overcooked_v2_experiments.talents.train_vae \
  --dataset "${DATASET}" \
  --output-dir "${VAE_DIR}" \
  --z-dim "${TALENTS_Z_DIM:-8}" \
  --hidden-dim "${TALENTS_HIDDEN_DIM:-256}" \
  --chunk-length "${TALENTS_CHUNK_LENGTH:-50}" \
  --batch-size "${TALENTS_VAE_BATCH_SIZE:-512}" \
  --epochs "${TALENTS_VAE_EPOCHS:-100}" \
  --steps-per-epoch "${TALENTS_VAE_STEPS_PER_EPOCH:-100}" \
  --lr "${TALENTS_VAE_LR:-0.0005}" \
  --kl-coef "${TALENTS_VAE_KL_COEF:-0.05}" \
  --seed "${SEED}"

"${PYTHON}" -m overcooked_v2_experiments.talents.cluster_latents \
  --dataset "${DATASET}" \
  --vae-checkpoint "${VAE_DIR}/gamma_vae.pkl" \
  --output-dir "${CLUSTER_DIR}" \
  --chunk-length "${TALENTS_CHUNK_LENGTH:-50}" \
  --num-samples "${TALENTS_CLUSTER_SAMPLES:-4096}" \
  --batch-size "${TALENTS_CLUSTER_BATCH_SIZE:-256}" \
  --num-clusters "${TALENTS_NUM_CLUSTERS:-0}" \
  --min-k "${TALENTS_MIN_K:-2}" \
  --max-k "${TALENTS_MAX_K:-8}" \
  --seed "${SEED}"

K="$("${PYTHON}" -c 'import pickle, sys; print(pickle.load(open(sys.argv[1], "rb"))["num_clusters"])' "${CLUSTER_DIR}/talents_clusters.pkl")"
echo "[TALENTS] selected_num_clusters=${K}"
TRAIN_EXTRA_ARGS=()
if [ -n "${POPULATION_MIX_PROB}" ]; then
  TRAIN_EXTRA_ARGS+=(+POPULATION_MIX_PROB="${POPULATION_MIX_PROB}")
fi

"${PYTHON}" -m overcooked_v2_experiments.talents.main \
  +experiment=cnn +env=original \
  env.ENV_KWARGS.layout="${LAYOUT}" \
  SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" NUM_CHECKPOINTS=3 VISUALIZE=False \
  +OPTIONAL_PREFIX="talents_cnn_standard_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_${TS}" \
  wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE="${WANDB_MODE}" \
  TALENTS.VAE_CHECKPOINT="${VAE_DIR}/gamma_vae.pkl" \
  TALENTS.CLUSTER_CHECKPOINT="${CLUSTER_DIR}/talents_clusters.pkl" \
  TALENTS.NUM_CLUSTERS="${K}" \
  TALENTS.FIXED_SHARE_ALPHA="${TALENTS_FIXED_SHARE_ALPHA:-0.4}" \
  TALENTS.FIXED_SHARE_ETA="${TALENTS_FIXED_SHARE_ETA:-0.2}" \
  TALENTS.REGRET_CLIP="${TALENTS_REGRET_CLIP:-1.0}" \
  TALENTS.Z_SAMPLE_SCALE="${TALENTS_Z_SAMPLE_SCALE}" \
  model.NUM_STRATEGY_CLUSTERS="${K}" \
  model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  model.NUM_ENVS="${NUM_ENVS}" \
  model.NUM_STEPS="${NUM_STEPS}" \
  model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
  model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
  "${TRAIN_EXTRA_ARGS[@]}"

echo "[TALENTS] full pipeline complete: ${RUN_ROOT}"
