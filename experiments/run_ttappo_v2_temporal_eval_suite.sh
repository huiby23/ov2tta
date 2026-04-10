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

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: bash experiments/run_ttappo_v2_temporal_eval_suite.sh <run_dir> [num_seeds] [pairing_policy]" >&2
  exit 1
fi

NUM_SEEDS="${2:-500}"
PAIRING_POLICY="${3:-}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="experiments/overcooked_v2_experiments/ttappo_v2_temporal/utils/visualize_ppo.py"

run_eval() {
  local mode="$1"
  local tag="$2"
  echo "[TTAPPO-V2 eval suite] mode=${mode} tag=${tag} run_dir=${RUN_DIR} num_seeds=${NUM_SEEDS}"
  local args=(--d "$RUN_DIR" --cross --num_seeds "$NUM_SEEDS" --no_viz --seed "$SEED" --ttt_mode "$mode" --output_tag "$tag")
  if [[ -n "$PAIRING_POLICY" ]]; then
    args+=(--pairing_policy "$PAIRING_POLICY")
  fi
  PYTHONPATH=experiments "$PYTHON_BIN" "$SCRIPT" "${args[@]}"
}

run_eval no_adapt no_adapt
run_eval ce_only ce_only
run_eval gated gated

echo "[TTAPPO-V2 eval suite] done"
