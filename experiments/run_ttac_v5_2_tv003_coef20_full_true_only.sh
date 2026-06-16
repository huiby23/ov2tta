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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_tv003_coef20_full_true_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_tv003_coef20_full_true_${TS}}"
OUTPUT_TAG="${OUTPUT_TAG:-v5_2_tv003_coef20_full_true}"

SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-0}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TV_THRESHOLD="${TV_THRESHOLD:-0.03}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

cat > "${REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
OUTPUT_TAG=${OUTPUT_TAG}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
EVAL_BATCHES=${EVAL_BATCHES}
EVAL_MAX_PAIRINGS=${EVAL_MAX_PAIRINGS}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TV_THRESHOLD=${TV_THRESHOLD}
EOF

max_pair_args=()
if [[ "${EVAL_MAX_PAIRINGS}" != "0" ]]; then
  max_pair_args=(--max_pairings "${EVAL_MAX_PAIRINGS}")
fi

log "EVAL_START mode=ttac_v5_2_tv_gate threshold=${TV_THRESHOLD} coef=${TTAC_V5_AGREEMENT_COEF} seeds=${EVAL_NUM_SEEDS} max_pairings=${EVAL_MAX_PAIRINGS}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/visualize_ppo.py \
  --d "${RUN_DIR}" \
  --seed "${SEED}" \
  --num_seeds "${EVAL_NUM_SEEDS}" \
  --cross \
  --no_viz \
  --ttac_mode ttac_v5_2_tv_gate \
  --output_tag "${OUTPUT_TAG}" \
  --eval_batches "${EVAL_BATCHES}" \
  "${max_pair_args[@]}" \
  --ttac_v5_estimator_path "${ESTIMATOR}" \
  --ttac_test_lr "${TTAC_TEST_LR}" \
  --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
  --ttac_history_len "${HISTORY_LEN}" \
  --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
  --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
  --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
  --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
  --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
  --ttac_v5_2_tv_threshold "${TV_THRESHOLD}" \
  > "${LOG_DIR}/eval_${OUTPUT_TAG}.log" 2>&1
log "EVAL_DONE mode=ttac_v5_2_tv_gate output_tag=${OUTPUT_TAG}"

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" OUTPUT_TAG="${OUTPUT_TAG}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import statistics

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
tag = os.environ["OUTPUT_TAG"]
csv_path = run / f"reward_summary_cross_{tag}.csv"

def label_kind(label):
    m = re.search(r"cross-(\d+)_(\d+)$", label)
    if not m:
        return "unknown"
    return "sp" if m.group(1) == m.group(2) else "xp"

pair_values = defaultdict(list)
with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        pair_values[row["policy_labels"]].append(float(row["total_reward"]))

per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
sp_vals = [v for k, v in per_pair.items() if label_kind(k) == "sp"]
xp_vals = [v for k, v in per_pair.items() if label_kind(k) == "xp"]
all_vals = list(per_pair.values())
summary = {
    "tag": tag,
    "xp_mean": statistics.fmean(xp_vals) if xp_vals else float("nan"),
    "sp_mean": statistics.fmean(sp_vals) if sp_vals else float("nan"),
    "all_mean": statistics.fmean(all_vals) if all_vals else float("nan"),
    "xp_pairs": len(xp_vals),
    "sp_pairs": len(sp_vals),
    "num_pairs": len(all_vals),
    "csv": str(csv_path),
}
with (report / "tv003_coef20_full_true_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)
md = [
    "# TTAC v5.2 TV-gate threshold=0.03 coef=20 full true eval",
    "",
    "| metric | value |",
    "|---|---:|",
    f"| xp_mean | {summary['xp_mean']:.3f} |",
    f"| sp_mean | {summary['sp_mean']:.3f} |",
    f"| all_mean | {summary['all_mean']:.3f} |",
    f"| xp_pairs | {summary['xp_pairs']} |",
    f"| sp_pairs | {summary['sp_pairs']} |",
    f"| num_pairs | {summary['num_pairs']} |",
    "",
    f"Source CSV: `{csv_path}`",
]
(report / "tv003_coef20_full_true_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY

log "DONE report=${REPORT_DIR}"
