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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_xp_amplify_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_xp_amplify_${TS}}"

SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
COEFS="${COEFS:-0.25 0.5 1.0 2.0 5.0 10.0 20.0}"
TAG_PREFIX="${TAG_PREFIX:-v5_2_amp}"
RUN_BASE="${RUN_BASE:-1}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
echo "${ESTIMATOR}" > "${REPORT_DIR}/estimator.txt"
cat > "${REPORT_DIR}/config.env" <<EOF
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
EVAL_MAX_PAIRINGS=${EVAL_MAX_PAIRINGS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TAG_PREFIX=${TAG_PREFIX}
RUN_BASE=${RUN_BASE}
COEFS=${COEFS}
EOF

run_eval() {
  local mode="$1"
  local tag="$2"
  local coef="$3"
  log "EVAL_START mode=${mode} tag=${tag} coef=${coef} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
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
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_v5_agreement_coef "${coef}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    > "${LOG_DIR}/eval_${tag}_${mode}.log" 2>&1
  log "EVAL_DONE mode=${mode} tag=${tag} coef=${coef}"
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
rows = []

def label_kind(label):
    m = re.search(r"cross-(\d+)_(\d+)$", label)
    if not m:
        return "unknown"
    return "sp" if m.group(1) == m.group(2) else "xp"

prefix = os.environ.get("TAG_PREFIX", "v5_2_amp")
for p in sorted(run.glob(f"reward_summary_cross_{prefix}_*.csv")):
    stem = p.name.removeprefix("reward_summary_cross_").removesuffix(".csv")
    parts = stem.split("_ttac_", 1)
    tag = parts[0]
    mode = "ttac_" + parts[1] if len(parts) == 2 else stem
    pair_values = defaultdict(list)
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            pair_values[r["policy_labels"]].append(float(r["total_reward"]))
    per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
    all_vals = list(per_pair.values())
    sp_vals = [v for k, v in per_pair.items() if label_kind(k) == "sp"]
    xp_vals = [v for k, v in per_pair.items() if label_kind(k) == "xp"]
    coef_match = re.search(r"coef([0-9p]+)", tag)
    coef = coef_match.group(1).replace("p", ".") if coef_match else ""
    rows.append({
        "tag": tag,
        "mode": mode,
        "coef": coef,
        "all_mean": statistics.fmean(all_vals) if all_vals else float("nan"),
        "sp_mean": statistics.fmean(sp_vals) if sp_vals else "",
        "xp_mean": statistics.fmean(xp_vals) if xp_vals else "",
        "sp_pairs": len(sp_vals),
        "xp_pairs": len(xp_vals),
        "num_pairs": len(all_vals),
        "csv": str(p),
    })

rows.sort(key=lambda r: (r["mode"], float(r["coef"] or -1), r["tag"]))
with (report / "xp_amplify_sweep_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "tag",
            "mode",
            "coef",
            "all_mean",
            "sp_mean",
            "xp_mean",
            "sp_pairs",
            "xp_pairs",
            "num_pairs",
            "csv",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

md = ["# TTAC v5.2 XP amplification sweep", ""]
md.append("| tag | mode | coef | XP offdiag | SP diag | all mean | XP pairs | SP pairs |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    def fmt(x):
        return "" if x == "" else f"{float(x):.3f}"
    md.append(
        f"| {r['tag']} | {r['mode']} | {r['coef']} | {fmt(r['xp_mean'])} | "
        f"{fmt(r['sp_mean'])} | {fmt(r['all_mean'])} | {r['xp_pairs']} | {r['sp_pairs']} |"
    )
(report / "xp_amplify_sweep_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
}

log "START report=${REPORT_DIR}"

if [[ "${RUN_BASE}" != "0" ]]; then
  # Base is evaluated once. It is independent of the agreement coefficient.
  run_eval base_no_test_adapt "${TAG_PREFIX}_base" 1.0
  summarize | tee -a "${QUEUE}"
fi

for coef in ${COEFS}; do
  coef_tag="${coef//./p}"
  run_eval ttac_v5_2_latest "${TAG_PREFIX}_coef${coef_tag}" "${coef}"
  summarize | tee -a "${QUEUE}"
done

log "DONE report=${REPORT_DIR}"
