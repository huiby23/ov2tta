#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
MODE="${MODE:-smoke}"  # smoke or full
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT}"
if [ -f "${ROOT}/experiments/repro_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/experiments/repro_env.sh"
fi
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_MODE="${WANDB_MODE:-}"

if [ "${MODE}" = "smoke" ]; then
  NUM_SEEDS="${NUM_SEEDS:-1}"
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-262144}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-131072}"
  NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
  EVAL_SEEDS="${EVAL_SEEDS:-4}"
  EVAL_BATCHES="${EVAL_BATCHES:-1}"
  WANDB_MODE="${WANDB_MODE:-disabled}"
else
  WANDB_MODE="${WANDB_MODE:-online}"
  NUM_SEEDS="${NUM_SEEDS:-10}"
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
  EVAL_SEEDS="${EVAL_SEEDS:-500}"
  EVAL_BATCHES="${EVAL_BATCHES:-10}"
fi

NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF:-0.05}"
TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF:-0.01}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-1}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.001}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"
DIAG_MAX_PAIRS="${DIAG_MAX_PAIRS:-}"
PREFIX="${PREFIX:-ttac_joint_adapter_state_aug_64_16_10M_seed42_10seeds_${TS}}"
REPORT_DIR="reports/ttac_repro_${TS}"
mkdir -p "${REPORT_DIR}"

log() { echo "[$(date +%F' '%T)] $*"; }

log "TTAC pipeline mode=${MODE} prefix=${PREFIX} layout=${LAYOUT}"
"${PYTHON}" -m overcooked_v2_experiments.ttac.main \
  +experiment=cnn \
  +env=original \
  env.ENV_KWARGS.layout="${LAYOUT}" \
  SEED="${SEED}" \
  NUM_SEEDS="${NUM_SEEDS}" \
  NUM_CHECKPOINTS=3 \
  +NUM_ITERATIONS="${NUM_ITERATIONS}" \
  VISUALIZE=False \
  +OPTIONAL_PREFIX="${PREFIX}" \
  wandb.ENTITY="${WANDB_ENTITY}" \
  wandb.PROJECT="${WANDB_PROJECT}" \
  wandb.WANDB_MODE="${WANDB_MODE}" \
  model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  model.NUM_ENVS="${NUM_ENVS}" \
  model.NUM_STEPS="${NUM_STEPS}" \
  model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
  model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
  model.TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF}" \
  model.TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF}" \
  model.TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS}" \
  model.TTAC_TEST_LR="${TTAC_TEST_LR}" \
  model.TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN}"

RUN_DIR="$(find "runs/${PREFIX}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
log "Training run_dir=${RUN_DIR}"

EVAL_MODES=(
  base_no_test_adapt
  ttac_true_history
  ttac_wrong_history
  ttac_random_history
  ttac_delayed_history
  ttac_no_kl
  ttac_adapter_off
)

for mode in "${EVAL_MODES[@]}"; do
  log "Eval mode=${mode}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac.utils.visualize_ppo \
    --d "${RUN_DIR}" --all --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" \
    --ttac_mode "${mode}" --output_tag "${mode}" --eval_batches "${EVAL_BATCHES}"
done

if [ "${RUN_DIAGNOSTICS}" = "1" ]; then
  for mode in "${EVAL_MODES[@]}"; do
    log "Diagnostics mode=${mode}"
    DIAG_ARGS=(
      --single_run_dir "${RUN_DIR}"
      --single_name "ttac_${mode}"
      --single_backend ttac
      --single_eval_mode "${mode}"
      --layout "${LAYOUT}"
      --eval_seed "${SEED}"
      --num_eval_seeds "${EVAL_SEEDS}"
      --num_diag_episodes "${DIAG_EPISODES}"
      --output_dir "${REPORT_DIR}/diagnostics_${mode}"
    )
    if [ -n "${DIAG_MAX_PAIRS}" ]; then
      DIAG_ARGS+=(--max_pairs "${DIAG_MAX_PAIRS}")
    fi
    "${PYTHON}" -m overcooked_v2_experiments.ttac.utils.zsc_diagnostics "${DIAG_ARGS[@]}"
  done
fi

log "TTAC pipeline complete report_dir=${REPORT_DIR} run_dir=${RUN_DIR}"
