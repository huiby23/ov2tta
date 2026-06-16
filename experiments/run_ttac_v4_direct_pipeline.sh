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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v4_direct_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v4_direct_${TS}}"
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
  TARGET_MAX_PAIRS="${TARGET_MAX_PAIRS:-2}"
  TARGET_NUM_EPISODES="${TARGET_NUM_EPISODES:-1}"
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
  TARGET_MAX_PAIRS="${TARGET_MAX_PAIRS:-12}"
  TARGET_NUM_EPISODES="${TARGET_NUM_EPISODES:-2}"
fi
EVAL_BATCHES="${EVAL_BATCHES:-10}"

DATASET_DIR="${REPORT_DIR}/dataset"
SURROGATE_DIR="${REPORT_DIR}/surrogate"
DATASET="${DATASET_DIR}/surrogate_dataset.npz"
SURROGATE="${SURROGATE_DIR}/surrogate_heads.npz"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"

log "STAGE1_DATASET_START pairs=${MAX_PAIRINGS} seeds=${NUM_EVAL_SEEDS} max_transitions=${MAX_TRANSITIONS}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v4_direct/utils/collect_surrogate_dataset.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${DATASET_DIR}" \
  --seed "${SEED}" \
  --num_eval_seeds "${NUM_EVAL_SEEDS}" \
  --max_pairings "${MAX_PAIRINGS}" \
  --transition_stride "${TRANSITION_STRIDE}" \
  --max_transitions "${MAX_TRANSITIONS}" \
  > "${LOG_DIR}/collect_dataset.log" 2>&1
log "STAGE1_DATASET_DONE dataset=${DATASET}"

log "STAGE2_SURROGATE_TRAIN_START steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v4_direct/utils/train_surrogate_heads.py \
  --dataset "${DATASET}" \
  --output_dir "${SURROGATE_DIR}" \
  --seed "${SEED}" \
  --steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  > "${LOG_DIR}/train_surrogate.log" 2>&1
log "STAGE2_SURROGATE_TRAIN_DONE surrogate=${SURROGATE}"

PASS_Q="$(${PYTHON} - <<PY
import pandas as pd
from pathlib import Path
p=Path('${SURROGATE_DIR}')/'surrogate_validation.csv'
df=pd.read_csv(p)
print(str(bool(df.iloc[0]['pass_q_history_gate'])).lower())
PY
)"
log "STAGE2_Q_GATE pass=${PASS_Q}"
if [ "${PASS_Q}" != "true" ] && [ "${FORCE_EVAL}" != "1" ]; then
  echo "direct surrogate stopped: q_eta true-history gate failed" > "${REPORT_DIR}/v4_decision.md"
  log "STOP_Q_HISTORY_GATE_FAILED"
  exit 0
fi

log "STAGE3_TARGET_AUDIT_START"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v4_direct/utils/target_effect_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${REPORT_DIR}/target_effect" \
  --seed "${SEED}" \
  --max_pairs "${TARGET_MAX_PAIRS}" \
  --num_episodes "${TARGET_NUM_EPISODES}" \
  --prefix_steps 80 \
  --eval_suffix_start 80 \
  --compatibility_sample_limit 128 \
  --modes base_no_update,v4_support_aw,v4_direct_q,v4_direct_q_support \
  --ttac_v4_surrogate_path "${SURROGATE}" \
  > "${LOG_DIR}/target_effect.log" 2>&1
log "STAGE3_TARGET_AUDIT_DONE"

run_eval() {
  local mode="$1"
  local tag="v4_direct_${mode}"
  log "EVAL_START mode=${mode}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v4_direct/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --max_pairings "${EVAL_MAX_PAIRINGS}" \
    --ttac_v4_surrogate_path "${SURROGATE}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode}"
}

log "STAGE4_PAIR_REWARD_START"
for mode in \
  base_no_test_adapt \
  ttac_v4_support_aw \
  ttac_v4_direct_q_support \
  ttac_v4_direct_q_support_wrong_history \
  ttac_v4_direct_q_support_random_history \
  ttac_v4_direct_q_support_delayed_history; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
import pandas as pd
run=Path(os.environ['RUN_DIR'])
report=Path(os.environ['REPORT_DIR'])
rows=[]
for p in sorted(run.glob('reward_summary_cross_v4_direct_*.csv')):
    mode=p.name.removeprefix('reward_summary_cross_v4_direct_').removesuffix('.csv')
    data=defaultdict(list)
    with p.open(newline='') as f:
        for r in csv.DictReader(f):
            data[r['policy_labels']].append(float(r['total_reward']))
    vals=[float(np.mean(v)) for v in data.values()]
    rows.append({'mode':mode,'mean':float(np.mean(vals)),'std_pair':float(np.std(vals)),'num_pairs':len(vals),'csv':str(p)})
df=pd.DataFrame(rows).sort_values('mode')
df.to_csv(report/'pair20_reward_summary.csv', index=False)
try:
    table=df.to_markdown(index=False)
except Exception:
    table=df.to_csv(index=False)
(report/'pair20_reward_summary.md').write_text('# TTAC v4-direct pair reward summary\n\n'+table+'\n')
print(df.to_string(index=False))
PY
log "STAGE4_PAIR_REWARD_DONE"
log "PIPELINE_DONE report=${REPORT_DIR}"
