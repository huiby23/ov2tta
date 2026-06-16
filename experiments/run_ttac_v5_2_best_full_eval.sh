#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
ESTIMATOR="${ESTIMATOR:-reports/ttac_v5_agreement_estimator_20260609_180926/agreement_estimator/agreement_estimator.npz}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_state_selection_${TS}/stage4_best_full_eval}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_state_selection_${TS}/stage4_best_full_eval}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-0}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-1.0}"
BEST_MODE="${BEST_MODE:?Set BEST_MODE, e.g. ttac_v5_2_tv_gate}"
BEST_THRESHOLD="${BEST_THRESHOLD:-0.05}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${ESTIMATOR}" > "${REPORT_DIR}/estimator.txt"
echo "${BEST_MODE}" > "${REPORT_DIR}/best_mode.txt"
echo "${BEST_THRESHOLD}" > "${REPORT_DIR}/best_threshold.txt"

run_eval() {
  local label="$1" mode="$2"
  local tag="v5_2_full_${label}_${mode}"
  local max_pair_args=()
  if [[ "${EVAL_MAX_PAIRINGS}" != "0" ]]; then
    max_pair_args=(--max_pairings "${EVAL_MAX_PAIRINGS}")
  fi
  log "EVAL_START label=${label} mode=${mode} threshold=${BEST_THRESHOLD} seeds=${EVAL_NUM_SEEDS} max_pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    "${max_pair_args[@]}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef 0.01 \
    --ttac_test_cur_kl_coef 0.01 \
    --ttac_test_hist_kl_coef 0.0 \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_2_tv_threshold "${BEST_THRESHOLD}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE label=${label} mode=${mode} threshold=${BEST_THRESHOLD}"
}

run_eval base base_no_test_adapt
run_eval best_true "${BEST_MODE}"
run_eval best_wrong "${BEST_MODE}_wrong_history"
run_eval best_random "${BEST_MODE}_random_history"
run_eval best_delayed "${BEST_MODE}_delayed_history"

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
run = Path(os.environ['RUN_DIR'])
report = Path(os.environ['REPORT_DIR'])
rows = []
for p in sorted(run.glob('reward_summary_cross_v5_2_full_*.csv')):
    name = p.name.removeprefix('reward_summary_cross_v5_2_full_').removesuffix('.csv')
    data = defaultdict(list)
    with p.open(newline='') as f:
        for r in csv.DictReader(f):
            data[r['policy_labels']].append(float(r['total_reward']))
    vals = [float(np.mean(v)) for v in data.values()]
    rows.append({'name': name, 'mean': float(np.mean(vals)), 'std_pair': float(np.std(vals)), 'num_pairs': len(vals), 'csv': str(p)})
rows = sorted(rows, key=lambda r: r['name'])
with (report / 'full_eval_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name','mean','std_pair','num_pairs','csv'])
    writer.writeheader(); writer.writerows(rows)
md = ['# TTAC v5.2 best full eval summary', '']
md.append('| name | mean | std_pair | num_pairs |')
md.append('|---|---:|---:|---:|')
for r in rows:
    md.append(f"| {r['name']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report / 'full_eval_summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
PY
log "DONE report=${REPORT_DIR}"
