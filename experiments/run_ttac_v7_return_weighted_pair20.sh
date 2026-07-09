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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v7_return_weighted_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v7_return_weighted_${TS}}"

SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_V5_2_TV_THRESHOLD="${TTAC_V5_2_TV_THRESHOLD:-0.03}"
TTAC_V7_TV_THRESHOLD="${TTAC_V7_TV_THRESHOLD:-0.03}"
TTAC_V7_TV_SLOPE="${TTAC_V7_TV_SLOPE:-25.0}"
TTAC_V7_VALUE_MARGIN="${TTAC_V7_VALUE_MARGIN:-0.0}"
TTAC_V7_VALUE_TEMP="${TTAC_V7_VALUE_TEMP:-5.0}"
TTAC_V7_CORRECTION_COEF="${TTAC_V7_CORRECTION_COEF:-1.0}"
TTAC_V7_SUPPORT_COEF="${TTAC_V7_SUPPORT_COEF:-0.0}"
TAG_PREFIX="${TAG_PREFIX:-v7_return_weighted_${TS}}"
MODES="${MODES:-base_no_test_adapt ttac_v5_2_latest ttac_v5_2_tv_gate ttac_v7_return_weighted ttac_v7_return_weighted_tv_gate ttac_v7_return_weighted_value_gate}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

cat > "${REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
EVAL_MAX_PAIRINGS=${EVAL_MAX_PAIRINGS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TTAC_V5_2_TV_THRESHOLD=${TTAC_V5_2_TV_THRESHOLD}
TTAC_V7_TV_THRESHOLD=${TTAC_V7_TV_THRESHOLD}
TTAC_V7_TV_SLOPE=${TTAC_V7_TV_SLOPE}
TTAC_V7_VALUE_MARGIN=${TTAC_V7_VALUE_MARGIN}
TTAC_V7_VALUE_TEMP=${TTAC_V7_VALUE_TEMP}
TTAC_V7_CORRECTION_COEF=${TTAC_V7_CORRECTION_COEF}
TTAC_V7_SUPPORT_COEF=${TTAC_V7_SUPPORT_COEF}
TAG_PREFIX=${TAG_PREFIX}
MODES=${MODES}
EOF

run_eval() {
  local mode="$1"
  local tag_mode="${mode//[^A-Za-z0-9]/_}"
  local tag="${TAG_PREFIX}_${tag_mode}"
  log "EVAL_START mode=${mode} tag=${tag} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v7_return_correction/utils/visualize_ppo.py \
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
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_2_tv_threshold "${TTAC_V5_2_TV_THRESHOLD}" \
    --ttac_v7_tv_threshold "${TTAC_V7_TV_THRESHOLD}" \
    --ttac_v7_tv_slope "${TTAC_V7_TV_SLOPE}" \
    --ttac_v7_value_margin "${TTAC_V7_VALUE_MARGIN}" \
    --ttac_v7_value_temp "${TTAC_V7_VALUE_TEMP}" \
    --ttac_v7_correction_coef "${TTAC_V7_CORRECTION_COEF}" \
    --ttac_v7_support_coef "${TTAC_V7_SUPPORT_COEF}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode} tag=${tag}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" TAG_PREFIX="${TAG_PREFIX}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import statistics

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
prefix = os.environ["TAG_PREFIX"]

def label_kind(label):
    m = re.search(r"cross-(\d+)_(\d+)$", label)
    if not m:
        return "unknown"
    return "sp" if m.group(1) == m.group(2) else "xp"

rows = []
for p in sorted(run.glob(f"reward_summary_cross_{prefix}_*.csv")):
    tag = p.name.removeprefix("reward_summary_cross_").removesuffix(".csv")
    mode = tag.removeprefix(prefix + "_")
    pair_values = defaultdict(list)
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            pair_values[r["policy_labels"]].append(float(r["total_reward"]))
    per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
    all_vals = list(per_pair.values())
    sp_vals = [v for k, v in per_pair.items() if label_kind(k) == "sp"]
    xp_vals = [v for k, v in per_pair.items() if label_kind(k) == "xp"]
    rows.append({
        "mode": mode,
        "xp_mean": statistics.fmean(xp_vals) if xp_vals else "",
        "sp_mean": statistics.fmean(sp_vals) if sp_vals else "",
        "all_mean": statistics.fmean(all_vals) if all_vals else "",
        "xp_pairs": len(xp_vals),
        "sp_pairs": len(sp_vals),
        "num_pairs": len(all_vals),
        "csv": str(p),
    })
rows.sort(key=lambda r: str(r["mode"]))
with (report / "v7_pair20_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["mode", "xp_mean", "sp_mean", "all_mean", "xp_pairs", "sp_pairs", "num_pairs", "csv"],
    )
    writer.writeheader()
    writer.writerows(rows)
md = ["# TTAC v7 return-weighted pair20", ""]
md.append("| mode | XP | SP | all | XP pairs | SP pairs |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    def fmt(x):
        return "" if x == "" else f"{float(x):.3f}"
    md.append(f"| {r['mode']} | {fmt(r['xp_mean'])} | {fmt(r['sp_mean'])} | {fmt(r['all_mean'])} | {r['xp_pairs']} | {r['sp_pairs']} |")
(report / "v7_pair20_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
}

log "START report=${REPORT_DIR}"
for mode in ${MODES}; do
  run_eval "${mode}"
  summarize | tee -a "${QUEUE}"
done
summarize | tee -a "${QUEUE}"
log "DONE report=${REPORT_DIR}"
