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
RUN_DIR="${RUN_DIR:-}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"
SEED="${SEED:-42}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
RUN_THRESHOLD_SCAN="${RUN_THRESHOLD_SCAN:-0}"
THRESHOLDS_CSV="${THRESHOLDS_CSV:-0.2,0.5,0.8,1.2}"
CAUSAL_MODES_CSV="${CAUSAL_MODES_CSV:-memory_off,state_adapt,state_adapt_wrong_partner,state_adapt_random_partner,state_adapt_delayed_partner,state_readout_off}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR is required"
  exit 1
fi

eval_mode() {
  local mode="$1"
  echo "[eval] mode=${mode}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz \
    --seed "${SEED}" \
    --ttt_mode "${mode}" \
    --output_tag "${mode}"
}

diag_mode() {
  local mode="$1"
  echo "[diag] mode=${mode}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/partner_diagnostics.py \
    --d "${RUN_DIR}" \
    --num_episodes "${DIAG_EPISODES}" \
    --seed "${SEED}" \
    --mode "${mode}" \
    --output_suffix "${mode}"
}

IFS=',' read -r -a CAUSAL_MODES <<< "${CAUSAL_MODES_CSV}"
for mode in "${CAUSAL_MODES[@]}"; do
  eval_mode "${mode}"
  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    diag_mode "${mode}"
  fi
done

if [[ "${RUN_THRESHOLD_SCAN}" == "1" ]]; then
  IFS=',' read -r -a THRESHOLDS <<< "${THRESHOLDS_CSV}"
  for threshold in "${THRESHOLDS[@]}"; do
    echo "[eval] gated threshold=${threshold}"
    PYTHONPATH=experiments "$PYTHON_BIN" \
      experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/visualize_ppo.py \
      --d "${RUN_DIR}" \
      --cross \
      --num_seeds "${EVAL_SEEDS}" \
      --no_viz \
      --seed "${SEED}" \
      --ttt_mode state_adapt_gated \
      --output_tag "state_adapt_gated_th${threshold}" \
      --ce_threshold_override "${threshold}"
  done
fi

echo "[causal-suite] done"
