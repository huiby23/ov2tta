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
SWEEP_REPORT_DIR="${SWEEP_REPORT_DIR:-$(find reports -maxdepth 1 -type d -name 'ttac_v3_alignment_sweep_*' | sort | tail -n 1)}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v3_alignment_best_full_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v3_alignment_best_full_eval_${TS}}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"; : > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }
tag_float() { echo "$1" | sed 's/-/m/g; s/\./p/g'; }

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
TTAC_TEST_SEMANTIC_LAMBDA="${TTAC_TEST_SEMANTIC_LAMBDA:-0.25}"
TTAC_V3_SEMANTIC_COEF="${TTAC_V3_SEMANTIC_COEF:-0.25}"
TTAC_V3_MARGIN="${TTAC_V3_MARGIN:-0.1}"
TTAC_V3_USE_CHANGE_GATE="${TTAC_V3_USE_CHANGE_GATE:-1}"
TTAC_V3_CHANGE_GATE_FLOOR="${TTAC_V3_CHANGE_GATE_FLOOR:-0.1}"

if [ -z "${SWEEP_REPORT_DIR}" ] || [ ! -f "${SWEEP_REPORT_DIR}/pair20_reward_summary.csv" ]; then
  log "ERROR no pair20_reward_summary.csv found. Run experiments/run_ttac_v3_alignment_pair20_sweep.sh first."
  exit 2
fi

set +e
BEST_JSON="$(SWEEP_REPORT_DIR="${SWEEP_REPORT_DIR}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import pandas as pd
p = Path(os.environ["SWEEP_REPORT_DIR"]) / "pair20_reward_summary.csv"
df = pd.read_csv(p)
passed = df[df["pass_success_criterion"].astype(str).str.lower().isin(["true", "1"])]
if passed.empty:
    raise SystemExit(2)
best = passed.sort_values(["true_minus_max_wrong_delayed", "true_mean"], ascending=False).iloc[0]
print(json.dumps(best.to_dict(), ensure_ascii=False))
PY
)"
status=$?
set -e
if [ "${status}" -ne 0 ] || [ -z "${BEST_JSON}" ]; then
  log "NO_PASSING_V3_TARGET: full validation skipped by design."
  exit 2
fi

echo "${BEST_JSON}" > "${REPORT_DIR}/best_pair20_target.json"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${SWEEP_REPORT_DIR}" > "${REPORT_DIR}/sweep_report_dir.txt"

read -r GROUP MARGIN_COEF TAU TRUE_MODE WRONG_MODE RANDOM_MODE DELAYED_MODE AUDIT_MODE < <(BEST_JSON="${BEST_JSON}" "${PYTHON}" - <<'PY'
import json, os
best = json.loads(os.environ["BEST_JSON"])
group = str(best["group"])
margin = str(best["margin_coef"])
tau = str(best["tau"])
base = {
    "support": "ttac_v3_support_aw",
    "gated_semantic": "ttac_v3_gated_semantic",
    "gated_margin": "ttac_v3_gated_margin",
    "margin_only": "ttac_v3_margin_only",
}[group]
audit = {
    "support": "v3_support_aw",
    "gated_semantic": "v3_gated_semantic",
    "gated_margin": "v3_gated_margin",
    "margin_only": "v3_margin_only",
}[group]
print(group, margin, tau, base, base + "_wrong_history", base + "_random_history", base + "_delayed_history", audit)
PY
)

log "FULL_EVAL_START group=${GROUP} mode=${TRUE_MODE} margin_coef=${MARGIN_COEF} tau=${TAU} num_seeds=${NUM_SEEDS}"

run_eval() {
  local mode="$1"
  local tag="v3_best_${GROUP}_mc$(tag_float "${MARGIN_COEF}")_tau$(tag_float "${TAU}")_${mode}"
  log "EVAL_START mode=${mode} tag=${tag}"
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
    --ttac_test_semantic_lambda "${TTAC_TEST_SEMANTIC_LAMBDA}" \
    --ttac_v3_semantic_coef "${TTAC_V3_SEMANTIC_COEF}" \
    --ttac_v3_margin_coef "${MARGIN_COEF}" \
    --ttac_v3_margin "${TTAC_V3_MARGIN}" \
    --ttac_v3_recency_tau "${TAU}" \
    --ttac_v3_use_change_gate "${TTAC_V3_USE_CHANGE_GATE}" \
    --ttac_v3_change_gate_floor "${TTAC_V3_CHANGE_GATE_FLOOR}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode}"
}

for mode in base_no_test_adapt ttac_v3_support_aw "${TRUE_MODE}" "${WRONG_MODE}" "${RANDOM_MODE}" "${DELAYED_MODE}"; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" GROUP="${GROUP}" MARGIN_COEF="${MARGIN_COEF}" TAU="${TAU}" TRUE_MODE="${TRUE_MODE}" WRONG_MODE="${WRONG_MODE}" RANDOM_MODE="${RANDOM_MODE}" DELAYED_MODE="${DELAYED_MODE}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os
import numpy as np
import pandas as pd
run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
group = os.environ["GROUP"]
mc = os.environ["MARGIN_COEF"].replace(".", "p")
tau = os.environ["TAU"].replace(".", "p")
prefix = f"reward_summary_cross_v3_best_{group}_mc{mc}_tau{tau}_"
rows = []
per_pair_frames = []
for path in sorted(run.glob(prefix + "*.csv")):
    mode = path.name.removeprefix(prefix).removesuffix(".csv")
    data = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            data[row["policy_labels"]].append(float(row["total_reward"]))
    per_pair = {k: float(np.mean(v)) for k, v in data.items()}
    vals = list(per_pair.values())
    rows.append({"mode": mode, "mean": float(np.mean(vals)), "std_pair": float(np.std(vals)), "num_pairs": len(vals), "csv": str(path)})
    for pair, val in per_pair.items():
        per_pair_frames.append({"mode": mode, "policy_labels": pair, "mean_reward": val})
df = pd.DataFrame(rows).sort_values("mode")
df.to_csv(report / "full_eval_summary.csv", index=False)
pair_df = pd.DataFrame(per_pair_frames)
pair_df.to_csv(report / "full_eval_per_pair.csv", index=False)
try:
    table = df.to_markdown(index=False)
except Exception:
    table = df.to_csv(index=False)
(report / "full_eval_summary.md").write_text(
    "# TTAC v3 Best Full Eval\n\n"
    + f"group: `{group}`\nmargin_coef: `{os.environ['MARGIN_COEF']}`\ntau: `{os.environ['TAU']}`\n"
    + f"true_mode: `{os.environ['TRUE_MODE']}`\nwrong_mode: `{os.environ['WRONG_MODE']}`\n"
    + f"random_mode: `{os.environ['RANDOM_MODE']}`\ndelayed_mode: `{os.environ['DELAYED_MODE']}`\n\n"
    + table + "\n"
)
print(df.to_string(index=False))
PY

DIAG_DIR="${REPORT_DIR}/target_effect_diagnostics"
log "TARGET_EFFECT_DIAGNOSTICS_START output=${DIAG_DIR} audit_mode=${AUDIT_MODE}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v3/utils/target_effect_audit.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${DIAG_DIR}" \
  --seed "${SEED}" \
  --max_pairs 12 \
  --num_episodes 2 \
  --prefix_steps 80 \
  --eval_suffix_start 80 \
  --compatibility_sample_limit 128 \
  --modes "base_no_update,v3_support_aw,${AUDIT_MODE}" \
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
  --ttac_test_semantic_lambda "${TTAC_TEST_SEMANTIC_LAMBDA}" \
  --ttac_v3_semantic_coef "${TTAC_V3_SEMANTIC_COEF}" \
  --ttac_v3_margin_coef "${MARGIN_COEF}" \
  --ttac_v3_margin "${TTAC_V3_MARGIN}" \
  --ttac_v3_recency_tau "${TAU}" \
  --ttac_v3_use_change_gate "${TTAC_V3_USE_CHANGE_GATE}" \
  --ttac_v3_change_gate_floor "${TTAC_V3_CHANGE_GATE_FLOOR}" \
  > "${LOG_DIR}/target_effect_diagnostics.log" 2>&1
log "TARGET_EFFECT_DIAGNOSTICS_DONE"
log "FULL_EVAL_DONE report=${REPORT_DIR}"
