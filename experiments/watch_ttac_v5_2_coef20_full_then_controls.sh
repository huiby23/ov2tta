#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
TRUE_REPORT_DIR="${TRUE_REPORT_DIR:-reports/ttac_v5_2_xp_amplify_full_coef20_20260615_1615}"
TRUE_TAG="${TRUE_TAG:-v5_2_amp_full_coef20_coef20p0}"
OLD_COEF1_XP="${OLD_COEF1_XP:-164.209}"
POLL_SECONDS="${POLL_SECONDS:-300}"
CONTROL_REPORT_DIR="${CONTROL_REPORT_DIR:-reports/ttac_v5_2_xp_amplify_full_coef20_controls_20260615_auto}"
CONTROL_LOG_DIR="${CONTROL_LOG_DIR:-logs/ttac_v5_2_xp_amplify_full_coef20_controls_20260615_auto}"
CONTROL_SENTINEL="${CONTROL_SENTINEL:-${TRUE_REPORT_DIR}/controls_started.sentinel}"

cd "${ROOT}"
mkdir -p "${TRUE_REPORT_DIR}"
WATCH_LOG="${TRUE_REPORT_DIR}/watch_coef20_full_then_controls.log"
CSV="${RUN_DIR}/reward_summary_cross_${TRUE_TAG}.csv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

summarize_true() {
  TRUE_REPORT_DIR="${TRUE_REPORT_DIR}" CSV="${CSV}" OLD_COEF1_XP="${OLD_COEF1_XP}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import statistics

report = Path(os.environ["TRUE_REPORT_DIR"])
csv_path = Path(os.environ["CSV"])
old_coef1_xp = float(os.environ["OLD_COEF1_XP"])

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
    "csv": str(csv_path),
    "all_mean": statistics.fmean(all_vals),
    "sp_mean": statistics.fmean(sp_vals) if sp_vals else float("nan"),
    "xp_mean": statistics.fmean(xp_vals) if xp_vals else float("nan"),
    "sp_pairs": len(sp_vals),
    "xp_pairs": len(xp_vals),
    "num_pairs": len(all_vals),
    "old_coef1_xp": old_coef1_xp,
    "delta_vs_old_coef1_xp": (statistics.fmean(xp_vals) if xp_vals else float("nan")) - old_coef1_xp,
}

with (report / "coef20_full_true_split_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

md = ["# TTAC v5.2 coef=20 full true-history summary", ""]
md.append("| metric | value |")
md.append("|---|---:|")
for key in ["xp_mean", "sp_mean", "all_mean", "xp_pairs", "sp_pairs", "num_pairs", "old_coef1_xp", "delta_vs_old_coef1_xp"]:
    val = summary[key]
    if isinstance(val, float):
        md.append(f"| {key} | {val:.3f} |")
    else:
        md.append(f"| {key} | {val} |")
md.append("")
md.append(f"Source CSV: `{csv_path}`")
(report / "coef20_full_true_split_summary.md").write_text("\n".join(md) + "\n")
print(summary["xp_mean"])
PY
}

log "WATCH_START csv=${CSV}"
while [[ ! -s "${CSV}" ]]; do
  log "WAIT csv_missing poll_seconds=${POLL_SECONDS}"
  sleep "${POLL_SECONDS}"
done

log "CSV_FOUND ${CSV}"
# Give the evaluator a short moment to finish wrapper post-processing if needed.
sleep 10
XP="$(summarize_true | tail -1)"
log "TRUE_SUMMARY_DONE xp=${XP} old_coef1_xp=${OLD_COEF1_XP}"

if "${PYTHON}" - <<PY
xp = float("${XP}")
old = float("${OLD_COEF1_XP}")
raise SystemExit(0 if xp > old else 1)
PY
then
  if [[ -e "${CONTROL_SENTINEL}" ]]; then
    log "CONTROLS_ALREADY_STARTED sentinel=${CONTROL_SENTINEL}"
  else
    log "START_CONTROLS xp=${XP} > old_coef1_xp=${OLD_COEF1_XP}"
    touch "${CONTROL_SENTINEL}"
    REPORT_DIR="${CONTROL_REPORT_DIR}" \
      LOG_DIR="${CONTROL_LOG_DIR}" \
      TTAC_V5_AGREEMENT_COEF=20.0 \
      EVAL_NUM_SEEDS=500 \
      EVAL_MAX_PAIRINGS=0 \
      EVAL_BATCHES=10 \
      experiments/run_ttac_v5_2_full_coef20_controls.sh \
      > "${CONTROL_LOG_DIR}.nohup.log" 2>&1 &
    log "CONTROLS_LAUNCHED report=${CONTROL_REPORT_DIR} log=${CONTROL_LOG_DIR}.nohup.log pid=$!"
  fi
else
  log "SKIP_CONTROLS xp=${XP} <= old_coef1_xp=${OLD_COEF1_XP}"
fi
