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

STANDARD_RUN_DIR="${STANDARD_RUN_DIR:-/teams/ius_1663576043/hby/rl/ov2/runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
STATE_AUG_RUN_DIR="${STATE_AUG_RUN_DIR:-/teams/ius_1663576043/hby/rl/ov2/runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full}"
LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-100}"
TOP_K="${TOP_K:-10}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-512}"
MAX_PAIRS="${MAX_PAIRS:-}"
FORCE_EVAL="${FORCE_EVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ppo_zsc_diagnostics_$(date +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --standard_run_dir "$STANDARD_RUN_DIR"
  --state_aug_run_dir "$STATE_AUG_RUN_DIR"
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

echo "[ppo-zsc] standard_run_dir=$STANDARD_RUN_DIR"
echo "[ppo-zsc] state_aug_run_dir=$STATE_AUG_RUN_DIR"
echo "[ppo-zsc] output_dir=$OUTPUT_DIR"

PYTHONPATH=experiments "$PYTHON_BIN" \
  experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py \
  "${ARGS[@]}"

echo "[ppo-zsc] done"
echo "[ppo-zsc] reward summary: $OUTPUT_DIR/reward_summary.csv"
echo "[ppo-zsc] coverage summary: $OUTPUT_DIR/coverage_summary.csv"
echo "[ppo-zsc] mismatch summary: $OUTPUT_DIR/shared_state_mismatch.csv"
echo "[ppo-zsc] complementarity summary: $OUTPUT_DIR/complementarity_summary.csv"
echo "[ppo-zsc] bottleneck states: $OUTPUT_DIR/bottleneck_states.csv"
echo "[ppo-zsc] report: $OUTPUT_DIR/report.md"
