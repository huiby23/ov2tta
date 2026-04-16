#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate myconda
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
  conda activate myconda
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate myconda
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MASTER_TAG="${MASTER_TAG:-xp_plan_$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-runs/${MASTER_TAG}_logs}"
mkdir -p "${LOG_DIR}"

RUN_STAGE_V3_CAUSAL="${RUN_STAGE_V3_CAUSAL:-1}"
RUN_STAGE_V4_SMOKE="${RUN_STAGE_V4_SMOKE:-1}"
RUN_STAGE_V4_FULL="${RUN_STAGE_V4_FULL:-1}"

V3_BASE_PREFIX="${V3_BASE_PREFIX:-figure4_ttappo_v3_memory_424344_20260414-213624}"
V3_RUN_DIRS_CSV="${V3_RUN_DIRS_CSV:-}"
CAUSAL_EVAL_SEEDS="${CAUSAL_EVAL_SEEDS:-500}"
CAUSAL_DIAG_EPISODES="${CAUSAL_DIAG_EPISODES:-100}"
CAUSAL_SEED="${CAUSAL_SEED:-42}"
CAUSAL_RUN_DIAGNOSTICS="${CAUSAL_RUN_DIAGNOSTICS:-1}"
CAUSAL_RUN_THRESHOLD_SCAN="${CAUSAL_RUN_THRESHOLD_SCAN:-1}"
CAUSAL_THRESHOLDS_CSV="${CAUSAL_THRESHOLDS_CSV:-0.2,0.5,0.8,1.2}"
CAUSAL_MODES_CSV="${CAUSAL_MODES_CSV:-memory_off,state_adapt,state_adapt_gated,state_adapt_wrong_partner,state_adapt_random_partner,state_adapt_delayed_partner,state_readout_off}"

BASE_SEEDS_CSV="${BASE_SEEDS_CSV:-42,43,44}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-3}"
LAYOUT="${LAYOUT:-counter_circuit}"
FULL_EVAL_SEEDS="${FULL_EVAL_SEEDS:-500}"
FULL_DIAG_EPISODES="${FULL_DIAG_EPISODES:-100}"
FULL_DIAG_WORKERS="${FULL_DIAG_WORKERS:-2}"
RUN_GATED_FULL="${RUN_GATED_FULL:-0}"

SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-262144}"
SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON:-131072}"
SMOKE_EVAL_SEEDS="${SMOKE_EVAL_SEEDS:-32}"
SMOKE_DIAG_EPISODES="${SMOKE_DIAG_EPISODES:-20}"

TRAIN_TOTAL_TIMESTEPS="${TRAIN_TOTAL_TIMESTEPS:-}"
TRAIN_REW_SHAPING_HORIZON="${TRAIN_REW_SHAPING_HORIZON:-}"
TRAIN_MODEL_NUM_ENVS="${TRAIN_MODEL_NUM_ENVS:-}"
TRAIN_MODEL_NUM_STEPS="${TRAIN_MODEL_NUM_STEPS:-}"
TRAIN_MODEL_NUM_MINIBATCHES="${TRAIN_MODEL_NUM_MINIBATCHES:-}"
TRAIN_MODEL_UPDATE_EPOCHS="${TRAIN_MODEL_UPDATE_EPOCHS:-}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

latest_child_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

resolve_v3_run_dirs() {
  if [[ -n "${V3_RUN_DIRS_CSV}" ]]; then
    IFS=',' read -r -a RUNS <<< "${V3_RUN_DIRS_CSV}"
    printf '%s\n' "${RUNS[@]}"
    return
  fi

  find runs -maxdepth 2 -type d \
    | grep "runs/${V3_BASE_PREFIX}_r[0-9]\\+/[^/]*$" \
    | sort
}

run_logged() {
  local stage_name="$1"
  shift
  local stage_log="${LOG_DIR}/${stage_name}.log"
  log "START ${stage_name}"
  {
    printf '[stage] %s\n' "${stage_name}"
    printf '[cmd] '
    printf '%q ' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${stage_log}"
  log "DONE ${stage_name}"
}

write_manifest_line() {
  printf '%s\n' "$*" | tee -a "${LOG_DIR}/manifest.txt" >/dev/null
}

log "MASTER_TAG=${MASTER_TAG}"
write_manifest_line "MASTER_TAG=${MASTER_TAG}"
write_manifest_line "LOG_DIR=${LOG_DIR}"

if [[ "${RUN_STAGE_V3_CAUSAL}" == "1" ]]; then
  mapfile -t V3_RUN_DIRS < <(resolve_v3_run_dirs)
  if [[ "${#V3_RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No v3 run dirs found for ${V3_BASE_PREFIX}" >&2
    exit 1
  fi
  idx=1
  for run_dir in "${V3_RUN_DIRS[@]}"; do
    run_logged \
      "v3_causal_r${idx}" \
      env \
      RUN_DIR="${ROOT_DIR}/${run_dir}" \
      EVAL_SEEDS="${CAUSAL_EVAL_SEEDS}" \
      DIAG_EPISODES="${CAUSAL_DIAG_EPISODES}" \
      SEED="${CAUSAL_SEED}" \
      RUN_DIAGNOSTICS="${CAUSAL_RUN_DIAGNOSTICS}" \
      RUN_THRESHOLD_SCAN="${CAUSAL_RUN_THRESHOLD_SCAN}" \
      THRESHOLDS_CSV="${CAUSAL_THRESHOLDS_CSV}" \
      CAUSAL_MODES_CSV="${CAUSAL_MODES_CSV}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash experiments/run_ttappo_v3_causal_suite.sh
    write_manifest_line "V3_CAUSAL_RUN_${idx}=${ROOT_DIR}/${run_dir}"
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE_V4_SMOKE}" == "1" ]]; then
  SMOKE_PIPELINE_TAG="${MASTER_TAG}_smoke"
  run_logged \
    "v4_smoke" \
    env \
    RUN_STAGE0=0 \
    RUN_STAGE1=1 \
    RUN_STAGE2=0 \
    RUN_STAGE3=0 \
    RUN_GATED=0 \
    PIPELINE_TAG="${SMOKE_PIPELINE_TAG}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    LAYOUT="${LAYOUT}" \
    SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS}" \
    SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON}" \
    SMOKE_EVAL_SEEDS="${SMOKE_EVAL_SEEDS}" \
    DIAG_EPISODES="${SMOKE_DIAG_EPISODES}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash experiments/run_ttappo_v4_actor_adapter_suite.sh
  SMOKE_PREFIX="figure4_ttappo_v4_actor_adapter_${SMOKE_PIPELINE_TAG}_smoke"
  SMOKE_RUN_DIR="$(latest_child_dir "${SMOKE_PREFIX}")"
  write_manifest_line "V4_SMOKE_RUN_DIR=${SMOKE_RUN_DIR}"
fi

if [[ "${RUN_STAGE_V4_FULL}" == "1" ]]; then
  FULL_PIPELINE_TAG="${MASTER_TAG}_full"
  FULL_ENV=(
    env
    RUN_STAGE0=0
    RUN_STAGE1=0
    RUN_STAGE2=1
    RUN_STAGE3=1
    RUN_GATED="${RUN_GATED_FULL}"
    PIPELINE_TAG="${FULL_PIPELINE_TAG}"
    BASE_SEEDS_CSV="${BASE_SEEDS_CSV}"
    NUM_SEEDS="${NUM_SEEDS}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}"
    LAYOUT="${LAYOUT}"
    EVAL_SEEDS="${FULL_EVAL_SEEDS}"
    DIAG_EPISODES="${FULL_DIAG_EPISODES}"
    DIAG_WORKERS="${FULL_DIAG_WORKERS}"
    PYTHON_BIN="${PYTHON_BIN}"
  )

  if [[ -n "${TRAIN_TOTAL_TIMESTEPS}" ]]; then
    FULL_ENV+=(TOTAL_TIMESTEPS="${TRAIN_TOTAL_TIMESTEPS}")
  fi
  if [[ -n "${TRAIN_REW_SHAPING_HORIZON}" ]]; then
    FULL_ENV+=(REW_SHAPING_HORIZON="${TRAIN_REW_SHAPING_HORIZON}")
  fi
  if [[ -n "${TRAIN_MODEL_NUM_ENVS}" ]]; then
    FULL_ENV+=(MODEL_NUM_ENVS="${TRAIN_MODEL_NUM_ENVS}")
  fi
  if [[ -n "${TRAIN_MODEL_NUM_STEPS}" ]]; then
    FULL_ENV+=(MODEL_NUM_STEPS="${TRAIN_MODEL_NUM_STEPS}")
  fi
  if [[ -n "${TRAIN_MODEL_NUM_MINIBATCHES}" ]]; then
    FULL_ENV+=(MODEL_NUM_MINIBATCHES="${TRAIN_MODEL_NUM_MINIBATCHES}")
  fi
  if [[ -n "${TRAIN_MODEL_UPDATE_EPOCHS}" ]]; then
    FULL_ENV+=(MODEL_UPDATE_EPOCHS="${TRAIN_MODEL_UPDATE_EPOCHS}")
  fi

  run_logged \
    "v4_full" \
    "${FULL_ENV[@]}" \
    bash experiments/run_ttappo_v4_actor_adapter_suite.sh

  FULL_PREFIX="figure4_ttappo_v4_actor_adapter_${FULL_PIPELINE_TAG}"
  FULL_SUMMARY_DIR="runs/${FULL_PREFIX}_summary"
  write_manifest_line "V4_FULL_PREFIX=${FULL_PREFIX}"
  write_manifest_line "V4_FULL_SUMMARY_DIR=${FULL_SUMMARY_DIR}"
fi

log "ALL_DONE"
write_manifest_line "STATUS=ALL_DONE"
