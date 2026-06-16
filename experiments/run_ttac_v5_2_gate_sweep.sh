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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_state_selection_${TS}/stage3_gate_sweep}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_state_selection_${TS}/stage3_gate_sweep}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-1.0}"
TV_THRESHOLDS="${TV_THRESHOLDS:-0.03 0.05 0.08 0.12}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${ESTIMATOR}" > "${REPORT_DIR}/estimator.txt"

run_eval() {
  local label="$1" mode="$2" threshold="$3"
  local tag="v5_2_${label}_${mode}"
  log "EVAL_START label=${label} mode=${mode} threshold=${threshold} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --max_pairings "${EVAL_MAX_PAIRINGS}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef 0.01 \
    --ttac_test_cur_kl_coef 0.01 \
    --ttac_test_hist_kl_coef 0.0 \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_2_tv_threshold "${threshold}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE label=${label} mode=${mode} threshold=${threshold}"
}

# Baselines and ungated latest controls.
run_eval base base_no_test_adapt 0.05
for mode in \
  ttac_v5_2_latest \
  ttac_v5_2_latest_wrong_history \
  ttac_v5_2_latest_random_history \
  ttac_v5_2_latest_delayed_history; do
  run_eval latest "${mode}" 0.05
done

for threshold in ${TV_THRESHOLDS}; do
  label="tv${threshold//./p}"
  for base_mode in ttac_v5_2_tv_gate ttac_v5_2_value_tv_gate ttac_v5_2_change_tv_gate; do
    for suffix in "" _wrong_history _random_history _delayed_history; do
      run_eval "${label}" "${base_mode}${suffix}" "${threshold}"
    done
  done
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
run = Path(os.environ['RUN_DIR'])
report = Path(os.environ['REPORT_DIR'])
rows = []
for p in sorted(run.glob('reward_summary_cross_v5_2_*.csv')):
    name = p.name.removeprefix('reward_summary_cross_v5_2_').removesuffix('.csv')
    if not (name.startswith('base_') or name.startswith('latest_') or name.startswith('tv')):
        continue
    data = defaultdict(list)
    with p.open(newline='') as f:
        for r in csv.DictReader(f):
            data[r['policy_labels']].append(float(r['total_reward']))
    vals = [float(np.mean(v)) for v in data.values()]
    rows.append({'name': name, 'mean': float(np.mean(vals)), 'std_pair': float(np.std(vals)), 'num_pairs': len(vals), 'csv': str(p)})
rows = sorted(rows, key=lambda r: r['name'])
with (report / 'gate_sweep_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name','mean','std_pair','num_pairs','csv'])
    writer.writeheader(); writer.writerows(rows)
md = ['# TTAC v5.2 gate sweep reward summary', '']
md.append('| name | mean | std_pair | num_pairs |')
md.append('|---|---:|---:|---:|')
for r in rows:
    md.append(f"| {r['name']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report / 'gate_sweep_summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
PY
log "DONE report=${REPORT_DIR}"
