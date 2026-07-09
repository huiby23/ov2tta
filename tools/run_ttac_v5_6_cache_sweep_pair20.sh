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

MASTER_REPORT_DIR="${MASTER_REPORT_DIR:-reports/ttac_v5_6_cache_sweep_pair20_${TS}}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/ttac_v5_6_cache_sweep_pair20_${TS}}"
mkdir -p "${MASTER_REPORT_DIR}" "${MASTER_LOG_DIR}"

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"

run_variant() {
  local name="$1"
  local interval="$2"
  local cache="$3"
  local report_dir="${MASTER_REPORT_DIR}/${name}"
  local log_dir="${MASTER_LOG_DIR}/${name}"
  echo "[$(date '+%F %T')] VARIANT_START ${name} interval=${interval} cache=${cache}" | tee -a "${MASTER_LOG_DIR}/queue.log"
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
  TTAC_TEST_LR="0.003" \
  TTAC_TEST_UPDATE_STEPS="3" \
  TTAC_UPDATE_INTERVAL="${interval}" \
  TTAC_UPDATE_GATE="none" \
  TTAC_CACHE_ESTIMATOR_TARGET="${cache}" \
  TTAC_V5_AGREEMENT_COEF="20.0" \
  TTAC_V5_SUPPORT_COEF="1.0" \
  TTAC_TEST_EGO_KL_COEF="0.01" \
  TTAC_TEST_CUR_KL_COEF="0.01" \
  TTAC_TEST_HIST_KL_COEF="0.0" \
  bash tools/run_ttac_v5_6_light_pair20.sh > "${MASTER_LOG_DIR}/${name}.stdout.log" 2>&1
  echo "[$(date '+%F %T')] VARIANT_DONE ${name}" | tee -a "${MASTER_LOG_DIR}/queue.log"
}

run_variant "every_step_cache_off" "1" "0"
run_variant "every_step_cache_on" "1" "1"
run_variant "interval2_cache_off" "2" "0"
run_variant "interval2_cache_on" "2" "1"

MASTER_REPORT_DIR="${MASTER_REPORT_DIR}" "${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import os

root = Path(os.environ["MASTER_REPORT_DIR"])
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
            rows.append({
                "variant": variant,
                "interval": cfg.get("TTAC_UPDATE_INTERVAL", ""),
                "cache": cfg.get("TTAC_CACHE_ESTIMATOR_TARGET", ""),
                "xp": float(r["xp"]),
                "seconds": float(r["seconds"]),
            })

off_by_interval = {
    r["interval"]: r for r in rows if r["cache"] == "0"
}
out = root / "cache_sweep_summary.csv"
with out.open("w", newline="") as f:
    fieldnames = [
        "variant", "interval", "cache", "xp", "seconds",
        "delta_xp_vs_cache_off", "speedup_vs_cache_off",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        ref = off_by_interval.get(r["interval"])
        writer.writerow({
            "variant": r["variant"],
            "interval": r["interval"],
            "cache": r["cache"],
            "xp": f"{r['xp']:.6f}",
            "seconds": f"{r['seconds']:.0f}",
            "delta_xp_vs_cache_off": f"{r['xp'] - ref['xp']:.6f}" if ref else "nan",
            "speedup_vs_cache_off": f"{ref['seconds'] / r['seconds']:.6f}" if ref and r["seconds"] > 0 else "nan",
        })

with out.open(newline="") as f:
    data = list(csv.DictReader(f))
data.sort(key=lambda r: (int(r["interval"]), int(r["cache"])))
md = root / "cache_sweep_summary.md"
with md.open("w") as f:
    f.write("# TTAC v5.6 Estimator Target Cache Sweep Pair20\n\n")
    f.write("All rows use `ttac_v5_6_latest_light`, `update_steps=3`, `lr=0.003`, `20 pairings x 100 eval seeds`.\n\n")
    f.write("| variant | interval | cache | XP | seconds | delta XP vs cache-off | speedup vs cache-off |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in data:
        f.write(
            f"| {r['variant']} | {r['interval']} | {r['cache']} | "
            f"{float(r['xp']):.3f} | {float(r['seconds']):.0f} | "
            f"{float(r['delta_xp_vs_cache_off']):+.3f} | "
            f"{float(r['speedup_vs_cache_off']):.2f}x |\n"
        )
print(md)
PY

echo "[$(date '+%F %T')] SWEEP_SUMMARY ${MASTER_REPORT_DIR}/cache_sweep_summary.md" | tee -a "${MASTER_LOG_DIR}/queue.log"
