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

MASTER_REPORT_DIR="${MASTER_REPORT_DIR:-reports/ttac_v5_6_event_gate_sweep_pair20_${TS}}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/ttac_v5_6_event_gate_sweep_pair20_${TS}}"
mkdir -p "${MASTER_REPORT_DIR}" "${MASTER_LOG_DIR}"

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_UPDATE_INTERVAL="${TTAC_UPDATE_INTERVAL:-2}"
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
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_UPDATE_INTERVAL=${TTAC_UPDATE_INTERVAL}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
EOF

run_variant() {
  local name="$1"
  local gate="$2"
  local tv_threshold="$3"
  local value_margin="$4"
  local report_dir="${MASTER_REPORT_DIR}/${name}"
  local log_dir="${MASTER_LOG_DIR}/${name}"
  echo "[$(date '+%F %T')] VARIANT_START ${name} gate=${gate} tv=${tv_threshold} value_margin=${value_margin}" | tee -a "${MASTER_LOG_DIR}/queue.log"
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
  TTAC_TEST_LR="${TTAC_TEST_LR}" \
  TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS}" \
  TTAC_UPDATE_INTERVAL="${TTAC_UPDATE_INTERVAL}" \
  TTAC_UPDATE_GATE="${gate}" \
  TTAC_UPDATE_TV_THRESHOLD="${tv_threshold}" \
  TTAC_UPDATE_VALUE_MARGIN="${value_margin}" \
  TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF}" \
  TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF}" \
  TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF}" \
  TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF}" \
  TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF}" \
  bash tools/run_ttac_v5_6_light_pair20.sh > "${MASTER_LOG_DIR}/${name}.stdout.log" 2>&1
  echo "[$(date '+%F %T')] VARIANT_DONE ${name}" | tee -a "${MASTER_LOG_DIR}/queue.log"
}

run_variant "interval2_no_gate" "none" "0.03" "0.0"
run_variant "interval2_tv003" "tv" "0.03" "0.0"
run_variant "interval2_tv005" "tv" "0.05" "0.0"
run_variant "interval2_value" "value" "0.03" "0.0"
run_variant "interval2_tv_value003" "tv_value" "0.03" "0.0"
run_variant "interval2_change" "change" "0.03" "0.0"
run_variant "interval2_tv_change003" "tv_change" "0.03" "0.0"

MASTER_REPORT_DIR="${MASTER_REPORT_DIR}" "${PYTHON}" - <<'PY'
from pathlib import Path
import csv

root = Path(__import__("os").environ["MASTER_REPORT_DIR"])
rows = []
for p in sorted(root.glob("*/summary.csv")):
    variant = p.parent.name
    cfg = {}
    cfg_path = p.parent / "config.env"
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k] = v
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            r["variant"] = variant
            r["gate"] = cfg.get("TTAC_UPDATE_GATE", "")
            r["tv_threshold"] = cfg.get("TTAC_UPDATE_TV_THRESHOLD", "")
            r["value_margin"] = cfg.get("TTAC_UPDATE_VALUE_MARGIN", "")
            rows.append(r)

baseline_xp = None
for r in rows:
    if r["variant"] == "interval2_no_gate":
        baseline_xp = float(r["xp"])
        break

out = root / "event_gate_sweep_summary.csv"
with out.open("w", newline="") as f:
    fieldnames = [
        "variant", "gate", "tv_threshold", "value_margin",
        "xp", "seconds", "delta_vs_interval2_no_gate", "xp_per_second",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        xp = float(r["xp"])
        sec = float(r["seconds"])
        writer.writerow({
            "variant": r["variant"],
            "gate": r["gate"],
            "tv_threshold": r["tv_threshold"],
            "value_margin": r["value_margin"],
            "xp": f"{xp:.6f}",
            "seconds": f"{sec:.0f}",
            "delta_vs_interval2_no_gate": f"{xp - baseline_xp:.6f}" if baseline_xp is not None else "nan",
            "xp_per_second": f"{xp / sec:.6f}" if sec > 0 else "nan",
        })

with out.open(newline="") as f:
    data = list(csv.DictReader(f))
data.sort(key=lambda r: float(r["xp"]), reverse=True)
md = root / "event_gate_sweep_summary.md"
with md.open("w") as f:
    f.write("# TTAC v5.6 Event-Gate Sweep Pair20\n\n")
    f.write("Base schedule for all rows: `interval=2`, `update_steps=3`, `lr=0.003`, `20 pairings x 100 eval seeds`.\n\n")
    f.write("| variant | gate | tv | XP | delta vs no-gate | seconds |\n")
    f.write("|---|---|---:|---:|---:|---:|\n")
    for r in data:
        f.write(
            f"| {r['variant']} | {r['gate']} | {r['tv_threshold']} | "
            f"{float(r['xp']):.3f} | {float(r['delta_vs_interval2_no_gate']):+.3f} | "
            f"{float(r['seconds']):.0f} |\n"
        )
print(md)
PY

echo "[$(date '+%F %T')] SWEEP_SUMMARY ${MASTER_REPORT_DIR}/event_gate_sweep_summary.md" | tee -a "${MASTER_LOG_DIR}/queue.log"
