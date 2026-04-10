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

ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-3}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"
LAYOUT="${LAYOUT:-counter_circuit}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
BASE_SEED="${BASE_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)}"
V2_PREFIX_BASE="${V2_PREFIX_BASE:-figure4_ttappo_v2_noadapt_repeat_${PIPELINE_TAG}}"
V3_PACKAGE="${V3_PACKAGE:-}"
V3_PREFIX="${V3_PREFIX:-figure4_ttappo_v3_${PIPELINE_TAG}}"

TRAIN_OVERRIDES=()
if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}")
fi
if [[ -n "${REW_SHAPING_HORIZON:-}" ]]; then
  TRAIN_OVERRIDES+=("model.REW_SHAPING_HORIZON=${REW_SHAPING_HORIZON}")
fi
if [[ -n "${MODEL_NUM_ENVS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_ENVS=${MODEL_NUM_ENVS}")
fi
if [[ -n "${MODEL_NUM_STEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_STEPS=${MODEL_NUM_STEPS}")
fi
if [[ -n "${MODEL_NUM_MINIBATCHES:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_MINIBATCHES=${MODEL_NUM_MINIBATCHES}")
fi
if [[ -n "${MODEL_UPDATE_EPOCHS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.UPDATE_EPOCHS=${MODEL_UPDATE_EPOCHS}")
fi

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

train_package() {
  local package="$1"
  local prefix="$2"
  local seed="$3"
  echo "[pipeline] train package=${package} prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" "experiments/overcooked_v2_experiments/${package}/main.py" \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

run_noadapt_eval() {
  local package="$1"
  local run_dir="$2"
  local tag="$3"
  echo "[pipeline] no_adapt eval package=${package} run_dir=${run_dir} tag=${tag}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    "experiments/overcooked_v2_experiments/${package}/utils/visualize_ppo.py" \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz \
    --seed "${BASE_SEED}" \
    --ttt_mode no_adapt \
    --output_tag "${tag}"
}

run_partner_diagnostics() {
  local run_dir="$1"
  echo "[pipeline] partner diagnostics run_dir=${run_dir} episodes=${DIAG_EPISODES}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    "experiments/overcooked_v2_experiments/ttappo_v2_temporal/utils/partner_diagnostics.py" \
    --d "${run_dir}" \
    --num_episodes "${DIAG_EPISODES}" \
    --seed "${BASE_SEED}"
}

echo "[pipeline] Stage 1: v2 temporal no_adapt repeats"
for ((i=1; i<=REPEAT_COUNT; i++)); do
  prefix="${V2_PREFIX_BASE}_r${i}"
  seed=$((BASE_SEED + i - 1))
  train_package ttappo_v2_temporal "${prefix}" "${seed}"
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[pipeline] Stage 1 run dir ${run_dir}"
  run_noadapt_eval ttappo_v2_temporal "${run_dir}" "no_adapt"
  run_partner_diagnostics "${run_dir}"
done

echo "[pipeline] Stage 2: current v2 online adaptation variants are intentionally paused"
echo "[pipeline] Stage 2: skipping ce_only/full/gated on ttappo_v2_temporal by design"

echo "[pipeline] Stage 3: v3 hook"
if [[ -z "${V3_PACKAGE}" ]]; then
  echo "[pipeline] Stage 3 skipped: V3_PACKAGE is empty; implement isolated-update v3 first"
else
  train_package "${V3_PACKAGE}" "${V3_PREFIX}" "${BASE_SEED}"
  V3_RUN_DIR="$(latest_run_dir "${V3_PREFIX}")"
  echo "[pipeline] Stage 3 v3 run dir ${V3_RUN_DIR}"
  run_noadapt_eval "${V3_PACKAGE}" "${V3_RUN_DIR}" "stage3_no_adapt"
  if [[ -x "experiments/run_${V3_PACKAGE}_eval_suite.sh" ]]; then
    bash "experiments/run_${V3_PACKAGE}_eval_suite.sh" "${V3_RUN_DIR}" "${EVAL_SEEDS}"
  else
    echo "[pipeline] No eval suite found for ${V3_PACKAGE}; only stage3_no_adapt was run"
  fi
fi

echo "[pipeline] done"
