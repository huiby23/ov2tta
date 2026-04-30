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

PPO_RNN_RUN_DIR="${PPO_RNN_RUN_DIR:-runs/figure4_ppo_rnn_state_aug_paper_budget_a40/20260406-171800_18yjvl1i_counter_circuit_avs-full}"
MAPPO_RNN_RUN_DIR="${MAPPO_RNN_RUN_DIR:-runs/figure4_mappo_state_aug_ippo_budget_match_multigpu/20260404-093657_hktgsy82_counter_circuit_avs-full}"
LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-full}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"

case "$MODE" in
  smoke)
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-2}"
    MAX_PAIRS="${MAX_PAIRS:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-runs/hidden_state_mismatch_smoke_${STAMP}}"
    ;;
  full)
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-100}"
    MAX_PAIRS="${MAX_PAIRS:-}"
    OUTPUT_DIR="${OUTPUT_DIR:-runs/hidden_state_mismatch_full_${STAMP}}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

ARGS=(
  --ppo_rnn_run_dir "$PPO_RNN_RUN_DIR"
  --mappo_rnn_run_dir "$MAPPO_RNN_RUN_DIR"
  --layout "$LAYOUT"
  --eval_seed "$EVAL_SEED"
  --num_diag_episodes "$NUM_DIAG_EPISODES"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_PAIRS" ]]; then
  ARGS+=(--max_pairs "$MAX_PAIRS")
fi

echo "[hidden-state-diag] mode=$MODE output=$OUTPUT_DIR"
echo "[hidden-state-diag] ppo_rnn=$PPO_RNN_RUN_DIR"
echo "[hidden-state-diag] mappo_rnn=$MAPPO_RNN_RUN_DIR"
echo "[hidden-state-diag] diag_episodes=$NUM_DIAG_EPISODES max_pairs=${MAX_PAIRS:-all}"

PYTHONUNBUFFERED=1 PYTHONPATH=experiments "$PYTHON_BIN" \
  experiments/overcooked_v2_experiments/ppo/utils/hidden_state_mismatch_diagnostics.py \
  "${ARGS[@]}"

echo "[hidden-state-diag] done"
echo "[hidden-state-diag] report: $OUTPUT_DIR/comparison_report.md"
