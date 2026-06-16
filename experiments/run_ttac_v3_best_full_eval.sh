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
SWEEP_REPORT_DIR="${SWEEP_REPORT_DIR:-}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v3_best_full_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v3_best_full_eval_${TS}}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA:-2.0}"
TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}"
TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER:-1.0}"
TTAC_TEST_CONTRAST_BETA="${TTAC_TEST_CONTRAST_BETA:-1.0}"
TTAC_TEST_CONTRAST_FLOOR="${TTAC_TEST_CONTRAST_FLOOR:-0.0}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

if [ -z "${SWEEP_REPORT_DIR}" ]; then
  SWEEP_REPORT_DIR="$(find reports -maxdepth 1 -type d -name 'ttac_v3_surrogate_sweep_*' | sort | tail -n 1)"
fi
if [ -z "${SWEEP_REPORT_DIR}" ] || [ ! -f "${SWEEP_REPORT_DIR}/lambda_sweep_summary.csv" ]; then
  log "ERROR no lambda_sweep_summary.csv found; run semantic lambda sweep first."
  exit 2
fi

set +e
BEST_LAMBDA="${BEST_LAMBDA:-$(SWEEP_REPORT_DIR="${SWEEP_REPORT_DIR}" "${PYTHON}" - <<'PY'
from pathlib import Path
import os
import pandas as pd
summary = pd.read_csv(Path(os.environ["SWEEP_REPORT_DIR"]) / "lambda_sweep_summary.csv")
passed = summary[(summary["complete"] == True) & (summary["pass_success_criterion"] == True)]
if passed.empty:
    raise SystemExit(2)
best = passed.sort_values(["true_minus_max_wrong_delayed", "winrate_true_vs_max_wrong_delayed"], ascending=False).iloc[0]
print(best["lambda"])
PY
)}"
status=$?
set -e
if [ "${status}" -ne 0 ] || [ -z "${BEST_LAMBDA}" ]; then
  log "NO_PASSING_LAMBDA: full validation skipped by design."
  exit 2
fi

log "FULL_EVAL_START lambda=${BEST_LAMBDA} sweep_report=${SWEEP_REPORT_DIR}"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${BEST_LAMBDA}" > "${REPORT_DIR}/best_lambda.txt"

run_eval() {
  local mode="$1"
  local lambda_tag="${BEST_LAMBDA//./p}"
  local tag="semantic_best_lam${lambda_tag}_${mode}"
  log "EVAL_START lambda=${BEST_LAMBDA} mode=${mode} seeds=${NUM_SEEDS} full_pairings=1"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v3/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" \
    --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" \
    --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --ttac_test_contrast_beta "${TTAC_TEST_CONTRAST_BETA}" \
    --ttac_test_contrast_floor "${TTAC_TEST_CONTRAST_FLOOR}" \
    --ttac_test_semantic_lambda "${BEST_LAMBDA}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE lambda=${BEST_LAMBDA} mode=${mode}"
}

for mode in base_no_test_adapt ttac_ego_advantage_weighted ttac_semantic_ego_aw ttac_semantic_ego_aw_wrong_history ttac_semantic_ego_aw_random_history ttac_semantic_ego_aw_delayed_history; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" BEST_LAMBDA="${BEST_LAMBDA}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
import pandas as pd
run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
lam = os.environ["BEST_LAMBDA"].replace(".", "p")
rows=[]
for path in sorted(run.glob(f"reward_summary_cross_semantic_best_lam{lam}_*.csv")):
    mode = path.name.removeprefix(f"reward_summary_cross_semantic_best_lam{lam}_").removesuffix(".csv")
    data=defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            data[row["policy_labels"]].append(float(row["total_reward"]))
    per_pair=[float(np.mean(v)) for v in data.values()]
    rows.append({"mode":mode,"mean":float(np.mean(per_pair)),"std_pair":float(np.std(per_pair)),"num_pairs":len(per_pair),"csv":str(path)})
df=pd.DataFrame(rows).sort_values("mode")
df.to_csv(report/"full_eval_summary.csv", index=False)
try:
    table=df.to_markdown(index=False)
except Exception:
    table=df.to_csv(index=False)
(report/"full_eval_summary.md").write_text(f"# TTAC Semantic Best Full Eval\n\nbest_lambda: `{os.environ['BEST_LAMBDA']}`\nrun_dir: `{run}`\n\n"+table+"\n")
print(df.to_string(index=False))
PY

DIAG_DIR="${REPORT_DIR}/diagnostics"
log "DIAGNOSTICS_START output=${DIAG_DIR}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v3/utils/target_effect_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${DIAG_DIR}" \
  --seed "${SEED}" \
  --max_pairs 12 \
  --num_episodes 2 \
  --prefix_steps 80 \
  --eval_suffix_start 80 \
  --compatibility_sample_limit 128 \
  --modes base_no_update,semantic_ego_aw,semantic_ego_aw_wrong_history,semantic_ego_aw_random_history,semantic_ego_aw_delayed_history \
  --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" \
  --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
  --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
  --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
  --ttac_test_lr "${TTAC_TEST_LR}" \
  --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
  --ttac_history_len "${TTAC_HISTORY_LEN}" \
  --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" \
  --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
  --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" \
  --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
  --ttac_test_contrast_beta "${TTAC_TEST_CONTRAST_BETA}" \
  --ttac_test_contrast_floor "${TTAC_TEST_CONTRAST_FLOOR}" \
  --ttac_test_semantic_lambda "${BEST_LAMBDA}" \
  > "${LOG_DIR}/diagnostics.log" 2>&1
log "DIAGNOSTICS_DONE"
log "FULL_EVAL_DONE report=${REPORT_DIR}"
