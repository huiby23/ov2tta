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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_1_agreement_amp_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_1_agreement_amp_${TS}}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-50}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-8}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
HISTORY_LEN="${HISTORY_LEN:-50}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${ESTIMATOR}" > "${REPORT_DIR}/estimator.txt"

run_eval() {
  local label="$1" mode="$2" coef="$3" lr="$4" steps="$5" ego_kl="$6" cur_kl="$7" ent_gate="$8" tv_gate="$9"
  local tag="v5_1_${label}_${mode}"
  log "EVAL_START label=${label} mode=${mode} coef=${coef} lr=${lr} steps=${steps} ego_kl=${ego_kl} cur_kl=${cur_kl} ent_gate=${ent_gate} tv_gate=${tv_gate}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_1_agreement_amp/utils/visualize_ppo.py \
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
    --ttac_test_lr "${lr}" \
    --ttac_test_update_steps "${steps}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef "${ego_kl}" \
    --ttac_test_cur_kl_coef "${cur_kl}" \
    --ttac_test_hist_kl_coef 0.0 \
    --ttac_v5_agreement_coef "${coef}" \
    --ttac_v5_support_coef 1.0 \
    --ttac_v5_conf_max_entropy "${ent_gate}" \
    --ttac_v5_conf_min_target_base_tv "${tv_gate}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE label=${label} mode=${mode}"
}

# Baselines once.
run_eval base base_no_test_adapt 1.0 0.003 3 0.01 0.01 1.25 0.03
run_eval base ttac_v5_support_aw 1.0 0.003 3 0.01 0.01 1.25 0.03

# label coef lr steps ego_kl cur_kl entropy_gate tv_gate
CONFIGS=(
  "mild 2.0 0.003 3 0.01 0.01 1.25 0.03"
  "amp 5.0 0.006 5 0.005 0.005 1.25 0.03"
  "conf 5.0 0.006 5 0.005 0.005 1.00 0.06"
)
MODES=(
  ttac_v5_1_multiquery
  ttac_v5_1_multiquery_wrong_history
  ttac_v5_1_multiquery_random_history
  ttac_v5_1_multiquery_delayed_history
  ttac_v5_1_multiquery_support
  ttac_v5_1_confident
  ttac_v5_1_confident_wrong_history
  ttac_v5_1_confident_random_history
  ttac_v5_1_confident_delayed_history
  ttac_v5_1_confident_support
)
for cfg in "${CONFIGS[@]}"; do
  read -r label coef lr steps ego_kl cur_kl ent_gate tv_gate <<<"${cfg}"
  for mode in "${MODES[@]}"; do
    run_eval "${label}" "${mode}" "${coef}" "${lr}" "${steps}" "${ego_kl}" "${cur_kl}" "${ent_gate}" "${tv_gate}"
  done
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
run=Path(os.environ['RUN_DIR'])
report=Path(os.environ['REPORT_DIR'])
rows=[]
for p in sorted(run.glob('reward_summary_cross_v5_1_*.csv')):
    name=p.name.removeprefix('reward_summary_cross_v5_1_').removesuffix('.csv')
    data=defaultdict(list)
    with p.open(newline='') as f:
        for r in csv.DictReader(f):
            data[r['policy_labels']].append(float(r['total_reward']))
    vals=[float(np.mean(v)) for v in data.values()]
    parts=name.split('_ttac_',1)
    label=parts[0]
    mode='ttac_'+parts[1] if len(parts)>1 else name
    rows.append({'label':label,'mode':mode,'mean':float(np.mean(vals)),'std_pair':float(np.std(vals)),'num_pairs':len(vals),'csv':str(p)})
rows=sorted(rows, key=lambda r:(r['label'], r['mode']))
with (report/'reward_sweep_summary.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=['label','mode','mean','std_pair','num_pairs','csv'])
    writer.writeheader(); writer.writerows(rows)
md=['# TTAC v5.1 agreement amplification reward sweep','']
md.append('| label | mode | mean | std_pair | num_pairs |')
md.append('|---|---|---:|---:|---:|')
for r in rows:
    md.append(f"| {r['label']} | {r['mode']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report/'reward_sweep_summary.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))
PY
log "PIPELINE_DONE report=${REPORT_DIR}"
