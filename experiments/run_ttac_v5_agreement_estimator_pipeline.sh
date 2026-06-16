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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_agreement_estimator_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_agreement_estimator_${TS}}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

if [ "${SMOKE}" = "1" ]; then
  MAX_PAIRINGS="${MAX_PAIRINGS:-2}"
  NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-5}"
  MAX_TRANSITIONS="${MAX_TRANSITIONS:-4000}"
  TRANSITION_STRIDE="${TRANSITION_STRIDE:-8}"
  TRAIN_STEPS="${TRAIN_STEPS:-100}"
  BATCH_SIZE="${BATCH_SIZE:-256}"
  EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-2}"
  EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-5}"
  FORCE_EVAL="${FORCE_EVAL:-1}"
else
  MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
  NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-100}"
  MAX_TRANSITIONS="${MAX_TRANSITIONS:-250000}"
  TRANSITION_STRIDE="${TRANSITION_STRIDE:-4}"
  TRAIN_STEPS="${TRAIN_STEPS:-2000}"
  BATCH_SIZE="${BATCH_SIZE:-1024}"
  EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
  EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
fi
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-1.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"

DATASET_DIR="${REPORT_DIR}/dataset"
ESTIMATOR_DIR="${REPORT_DIR}/agreement_estimator"
DATASET="${DATASET_DIR}/agreement_dataset.npz"
ESTIMATOR="${ESTIMATOR_DIR}/agreement_estimator.npz"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"

log "STAGE1_DATASET_START pairs=${MAX_PAIRINGS} seeds=${NUM_EVAL_SEEDS} max_transitions=${MAX_TRANSITIONS}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_agreement_estimator/utils/collect_agreement_dataset.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${DATASET_DIR}" \
  --seed "${SEED}" \
  --num_eval_seeds "${NUM_EVAL_SEEDS}" \
  --max_pairings "${MAX_PAIRINGS}" \
  --transition_stride "${TRANSITION_STRIDE}" \
  --max_transitions "${MAX_TRANSITIONS}" \
  --history_len "${HISTORY_LEN}" \
  > "${LOG_DIR}/collect_agreement_dataset.log" 2>&1
log "STAGE1_DATASET_DONE dataset=${DATASET}"

log "STAGE2_ESTIMATOR_TRAIN_START steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_agreement_estimator/utils/train_agreement_estimator.py \
  --dataset "${DATASET}" \
  --output_dir "${ESTIMATOR_DIR}" \
  --seed "${SEED}" \
  --steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  > "${LOG_DIR}/train_agreement_estimator.log" 2>&1
log "STAGE2_ESTIMATOR_TRAIN_DONE estimator=${ESTIMATOR}"

PASS_HISTORY="$(${PYTHON} - <<PY
import csv
from pathlib import Path
p=Path('${ESTIMATOR_DIR}')/'agreement_validation.csv'
with p.open(newline='') as f:
    row=next(csv.DictReader(f))
print(str(row.get('pass_history_gate','False')).lower())
PY
)"
log "STAGE2_HISTORY_GATE pass=${PASS_HISTORY}"
if [ "${PASS_HISTORY}" != "true" ] && [ "${FORCE_EVAL}" != "1" ]; then
  echo "# TTAC v5 decision" > "${REPORT_DIR}/v5_decision.md"
  echo >> "${REPORT_DIR}/v5_decision.md"
  echo "Stopped before reward eval because true-history estimator gate failed." >> "${REPORT_DIR}/v5_decision.md"
  log "STOP_HISTORY_GATE_FAILED"
  exit 0
fi

run_eval() {
  local mode="$1"
  local tag="v5_agreement_${mode}"
  log "EVAL_START mode=${mode}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_agreement_estimator/utils/visualize_ppo.py \
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
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode}"
}

log "STAGE3_PAIR20_REWARD_START"
for mode in \
  base_no_test_adapt \
  ttac_v5_support_aw \
  ttac_v5_agreement \
  ttac_v5_agreement_wrong_history \
  ttac_v5_agreement_random_history \
  ttac_v5_agreement_delayed_history \
  ttac_v5_agreement_support \
  ttac_v5_agreement_support_wrong_history \
  ttac_v5_agreement_support_random_history \
  ttac_v5_agreement_support_delayed_history; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
run=Path(os.environ['RUN_DIR'])
report=Path(os.environ['REPORT_DIR'])
rows=[]
for p in sorted(run.glob('reward_summary_cross_v5_agreement_*.csv')):
    mode=p.name.removeprefix('reward_summary_cross_v5_agreement_').removesuffix('.csv')
    data=defaultdict(list)
    with p.open(newline='') as f:
        for r in csv.DictReader(f):
            data[r['policy_labels']].append(float(r['total_reward']))
    vals=[float(np.mean(v)) for v in data.values()]
    rows.append({'mode':mode,'mean':float(np.mean(vals)),'std_pair':float(np.std(vals)),'num_pairs':len(vals),'csv':str(p)})
out=report/'pair20_reward_summary.csv'
with out.open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=['mode','mean','std_pair','num_pairs','csv'])
    writer.writeheader(); writer.writerows(rows)
md=['# TTAC v5 agreement estimator pair20 reward summary','']
md.append('| mode | mean | std_pair | num_pairs |')
md.append('|---|---:|---:|---:|')
for r in rows:
    md.append(f"| {r['mode']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report/'pair20_reward_summary.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))
PY
log "STAGE3_PAIR20_REWARD_DONE"
log "PIPELINE_DONE report=${REPORT_DIR}"
