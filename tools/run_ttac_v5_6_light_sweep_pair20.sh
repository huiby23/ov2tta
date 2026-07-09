#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

MASTER_REPORT_DIR="${MASTER_REPORT_DIR:-reports/ttac_v5_6_light_sweep_pair20_${TS}}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/ttac_v5_6_light_sweep_pair20_${TS}}"
mkdir -p "${MASTER_REPORT_DIR}" "${MASTER_LOG_DIR}"

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"

cat > "${MASTER_REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
MAX_PAIRINGS=${MAX_PAIRINGS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
EOF

run_variant() {
  local name="$1"
  local lr="$2"
  local steps="$3"
  local interval="$4"
  local report_dir="${MASTER_REPORT_DIR}/${name}"
  local log_dir="${MASTER_LOG_DIR}/${name}"
  echo "[$(date '+%F %T')] VARIANT_START ${name} lr=${lr} steps=${steps} interval=${interval}" | tee -a "${MASTER_LOG_DIR}/queue.log"
  RUN_DIR="${RUN_DIR}" \
  ESTIMATOR="${ESTIMATOR}" \
  REPORT_DIR="${report_dir}" \
  LOG_DIR="${log_dir}" \
  TS="${name}_${TS}" \
  MODES="ttac_v5_6_latest_light" \
  SEED="${SEED}" \
  EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS}" \
  MAX_PAIRINGS="${MAX_PAIRINGS}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  HISTORY_LEN="${HISTORY_LEN}" \
  TTAC_TEST_LR="${lr}" \
  TTAC_TEST_UPDATE_STEPS="${steps}" \
  TTAC_UPDATE_INTERVAL="${interval}" \
  TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF}" \
  TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF}" \
  TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF}" \
  TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF}" \
  TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF}" \
  bash tools/run_ttac_v5_6_light_pair20.sh > "${MASTER_LOG_DIR}/${name}.stdout.log" 2>&1
  echo "[$(date '+%F %T')] VARIANT_DONE ${name}" | tee -a "${MASTER_LOG_DIR}/queue.log"
}

run_variant "every_step_steps3_lr003" "0.003" "3" "1"
run_variant "interval2_steps3_lr003" "0.003" "3" "2"
run_variant "interval4_steps3_lr003" "0.003" "3" "4"
run_variant "every_step_steps1_lr009" "0.009" "1" "1"
run_variant "interval2_steps1_lr009" "0.009" "1" "2"

MASTER_REPORT_DIR="${MASTER_REPORT_DIR}" "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import math
import os

root = Path(os.environ["MASTER_REPORT_DIR"])
rows = []
for summary in sorted(root.glob("*/summary.csv")):
    variant = summary.parent.name
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            row["variant"] = variant
            rows.append(row)

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

base_xp = None
out_csv = root / "sweep_summary.csv"
with out_csv.open("w", newline="") as f:
    fieldnames = [
        "variant", "mode", "xp", "sp", "all", "seconds",
        "xp_delta_vs_light_baseline", "xp_per_second_gain_vs_base",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        xp = fnum(row.get("xp", "nan"))
        seconds = fnum(row.get("seconds", "nan"))
        if row["variant"] == "every_step_steps3_lr003":
            base_xp = xp
        delta = xp - base_xp if base_xp is not None and math.isfinite(xp) else float("nan")
        writer.writerow({
            "variant": row["variant"],
            "mode": row["mode"],
            "xp": f"{xp:.6f}" if math.isfinite(xp) else "nan",
            "sp": row.get("sp", "nan"),
            "all": row.get("all", "nan"),
            "seconds": row.get("seconds", "nan"),
            "xp_delta_vs_light_baseline": f"{delta:.6f}" if math.isfinite(delta) else "nan",
            "xp_per_second_gain_vs_base": f"{xp / seconds:.6f}" if math.isfinite(xp) and seconds > 0 else "nan",
        })

summary_md = root / "sweep_summary.md"
with out_csv.open(newline="") as f:
    data = list(csv.DictReader(f))
data.sort(key=lambda r: float(r["xp"]), reverse=True)
with summary_md.open("w") as f:
    f.write("# TTAC v5.6 Light Online Sweep Pair20\n\n")
    f.write("| variant | XP | seconds | delta vs every-step |\n")
    f.write("|---|---:|---:|---:|\n")
    for r in data:
        f.write(
            f"| {r['variant']} | {float(r['xp']):.3f} | "
            f"{float(r['seconds']):.0f} | {float(r['xp_delta_vs_light_baseline']):+.3f} |\n"
        )
    f.write("\n")
    f.write("Variants all use `ttac_v5_6_latest_light`, 20 pairings x 100 eval seeds.\n")
print(summary_md)
PY

echo "[$(date '+%F %T')] SWEEP_SUMMARY ${MASTER_REPORT_DIR}/sweep_summary.md" | tee -a "${MASTER_LOG_DIR}/queue.log"
