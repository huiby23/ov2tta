#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
cd "${ROOT}"
REPORT_ROOT="${REPORT_ROOT:-reports/ttac_v5_2_state_selection_${TS}}"
LOG_ROOT="${LOG_ROOT:-logs/ttac_v5_2_state_selection_${TS}}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-5}"
  export EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-2}"
  export EVAL_BATCHES="${EVAL_BATCHES:-2}"
  export ATTR_NUM_SEEDS="${ATTR_NUM_SEEDS:-5}"
  export ATTR_MAX_PAIRINGS="${ATTR_MAX_PAIRINGS:-2}"
  export TV_THRESHOLDS="${TV_THRESHOLDS:-0.05}"
  export MAX_SAMPLES="${MAX_SAMPLES:-5000}"
else
  export EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
  export EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
  export EVAL_BATCHES="${EVAL_BATCHES:-10}"
  export ATTR_NUM_SEEDS="${ATTR_NUM_SEEDS:-20}"
  export ATTR_MAX_PAIRINGS="${ATTR_MAX_PAIRINGS:-20}"
fi

export REPORT_DIR="${REPORT_ROOT}/stage1_same_script"
export LOG_DIR="${LOG_ROOT}/stage1_same_script"
bash experiments/run_ttac_v5_2_same_script_eval.sh

export REPORT_DIR="${REPORT_ROOT}/stage2_paired_attribution"
export LOG_DIR="${LOG_ROOT}/stage2_paired_attribution"
bash experiments/run_ttac_v5_2_paired_attribution.sh

export REPORT_DIR="${REPORT_ROOT}/stage3_gate_sweep"
export LOG_DIR="${LOG_ROOT}/stage3_gate_sweep"
bash experiments/run_ttac_v5_2_gate_sweep.sh

echo "TTAC v5.2 pipeline done: ${REPORT_ROOT}"
