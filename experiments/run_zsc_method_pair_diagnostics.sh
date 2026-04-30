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

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-}"
COMPARISON_RUN_DIR="${COMPARISON_RUN_DIR:-}"
BASELINE_NAME="${BASELINE_NAME:-baseline}"
COMPARISON_NAME="${COMPARISON_NAME:-comparison}"
BASELINE_BACKEND="${BASELINE_BACKEND:-ppo}"
COMPARISON_BACKEND="${COMPARISON_BACKEND:-ppo}"
BASELINE_EVAL_MODE="${BASELINE_EVAL_MODE:-memory_off}"
COMPARISON_EVAL_MODE="${COMPARISON_EVAL_MODE:-memory_off}"
LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-100}"
TOP_K="${TOP_K:-10}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-512}"
MAX_PAIRS="${MAX_PAIRS:-}"
FORCE_EVAL="${FORCE_EVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/zsc_pair_diagnostics_$(date +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${BASELINE_RUN_DIR}" || -z "${COMPARISON_RUN_DIR}" ]]; then
  echo "BASELINE_RUN_DIR and COMPARISON_RUN_DIR are required"
  exit 1
fi

ARGS=(
  --baseline_run_dir "$BASELINE_RUN_DIR"
  --comparison_run_dir "$COMPARISON_RUN_DIR"
  --baseline_name "$BASELINE_NAME"
  --comparison_name "$COMPARISON_NAME"
  --baseline_backend "$BASELINE_BACKEND"
  --comparison_backend "$COMPARISON_BACKEND"
  --baseline_eval_mode "$BASELINE_EVAL_MODE"
  --comparison_eval_mode "$COMPARISON_EVAL_MODE"
  --layout "$LAYOUT"
  --eval_seed "$EVAL_SEED"
  --num_eval_seeds "$NUM_EVAL_SEEDS"
  --num_diag_episodes "$NUM_DIAG_EPISODES"
  --top_k "$TOP_K"
  --compatibility_sample_limit "$COMPATIBILITY_SAMPLE_LIMIT"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_PAIRS" ]]; then
  ARGS+=(--max_pairs "$MAX_PAIRS")
fi

if [[ "$FORCE_EVAL" == "1" ]]; then
  ARGS+=(--force_eval)
fi

echo "[zsc-pair] baseline=${BASELINE_NAME} backend=${BASELINE_BACKEND} eval_mode=${BASELINE_EVAL_MODE}"
echo "[zsc-pair] baseline_run_dir=${BASELINE_RUN_DIR}"
echo "[zsc-pair] comparison=${COMPARISON_NAME} backend=${COMPARISON_BACKEND} eval_mode=${COMPARISON_EVAL_MODE}"
echo "[zsc-pair] comparison_run_dir=${COMPARISON_RUN_DIR}"
echo "[zsc-pair] output_dir=${OUTPUT_DIR}"

PYTHONPATH=experiments "$PYTHON_BIN" \
  experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py \
  "${ARGS[@]}"

echo "[zsc-pair] done"
echo "[zsc-pair] reward summary: $OUTPUT_DIR/reward_summary.csv"
echo "[zsc-pair] coverage summary: $OUTPUT_DIR/coverage_summary.csv"
echo "[zsc-pair] mismatch summary: $OUTPUT_DIR/shared_state_mismatch.csv"
echo "[zsc-pair] complementarity summary: $OUTPUT_DIR/complementarity_summary.csv"
echo "[zsc-pair] bottleneck states: $OUTPUT_DIR/bottleneck_states.csv"
echo "[zsc-pair] report: $OUTPUT_DIR/report.md"
