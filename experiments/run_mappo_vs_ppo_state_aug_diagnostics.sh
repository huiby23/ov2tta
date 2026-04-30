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

PPO_STATE_AUG_RUN_DIR="${PPO_STATE_AUG_RUN_DIR:-runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full}"
MAPPO_STATE_AUG_RUN_DIR="${MAPPO_STATE_AUG_RUN_DIR:-runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full}"
LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
TOP_K="${TOP_K:-10}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-512}"
COMPATIBILITY_VALUE_MODE="${COMPATIBILITY_VALUE_MODE:-reward_only}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-full}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"

case "$MODE" in
  smoke)
    NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-5}"
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-2}"
    MAX_PAIRS="${MAX_PAIRS:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-runs/mappo_vs_ppo_state_aug_diag_smoke_${STAMP}}"
    ;;
  full)
    NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-100}"
    MAX_PAIRS="${MAX_PAIRS:-}"
    OUTPUT_DIR="${OUTPUT_DIR:-runs/mappo_vs_ppo_state_aug_diagnostics_${STAMP}}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

ARGS=(
  --baseline_run_dir "$PPO_STATE_AUG_RUN_DIR"
  --comparison_run_dir "$MAPPO_STATE_AUG_RUN_DIR"
  --baseline_name ppo_cnn_state_aug
  --comparison_name mappo_state_aug
  --baseline_backend ppo
  --comparison_backend mappo
  --layout "$LAYOUT"
  --eval_seed "$EVAL_SEED"
  --num_eval_seeds "$NUM_EVAL_SEEDS"
  --num_diag_episodes "$NUM_DIAG_EPISODES"
  --top_k "$TOP_K"
  --compatibility_sample_limit "$COMPATIBILITY_SAMPLE_LIMIT"
  --compatibility_value_mode "$COMPATIBILITY_VALUE_MODE"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_PAIRS" ]]; then
  ARGS+=(--max_pairs "$MAX_PAIRS")
fi
if [[ "$FORCE_EVAL" == "1" ]]; then
  ARGS+=(--force_eval)
fi

echo "[mappo-vs-ppo-diag] mode=$MODE layout=$LAYOUT output=$OUTPUT_DIR"
echo "[mappo-vs-ppo-diag] ppo=$PPO_STATE_AUG_RUN_DIR"
echo "[mappo-vs-ppo-diag] mappo=$MAPPO_STATE_AUG_RUN_DIR"
echo "[mappo-vs-ppo-diag] eval_seeds=$NUM_EVAL_SEEDS diag_episodes=$NUM_DIAG_EPISODES max_pairs=${MAX_PAIRS:-all} value_mode=$COMPATIBILITY_VALUE_MODE"

PYTHONUNBUFFERED=1 PYTHONPATH=experiments "$PYTHON_BIN" \
  experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py \
  "${ARGS[@]}"

echo "[mappo-vs-ppo-diag] done"
echo "[mappo-vs-ppo-diag] report: $OUTPUT_DIR/report.md"
