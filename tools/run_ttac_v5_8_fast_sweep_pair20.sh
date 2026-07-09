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

MASTER_REPORT_DIR="${MASTER_REPORT_DIR:-reports/ttac_v5_8_fast_sweep_pair20_${TS}}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-logs/ttac_v5_8_fast_sweep_pair20_${TS}}"
mkdir -p "${MASTER_REPORT_DIR}" "${MASTER_LOG_DIR}"

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"

run_variant() {
  local name="$1"
  local mode="$2"
  local interval="$3"
  local step_size="$4"
  local latent_dim="$5"
  local latent_scale="$6"
  local report_dir="${MASTER_REPORT_DIR}/${name}"
  local log_dir="${MASTER_LOG_DIR}/${name}"
  echo "[$(date '+%F %T')] VARIANT_START ${name} mode=${mode} interval=${interval} step=${step_size}" | tee -a "${MASTER_LOG_DIR}/queue.log"
  RUN_DIR="${RUN_DIR}" \
  ESTIMATOR="${ESTIMATOR}" \
  REPORT_DIR="${report_dir}" \
  LOG_DIR="${log_dir}" \
  TS="${name}_${TS}" \
  MODES="${mode}" \
  SEED="${SEED}" \
  EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS}" \
  MAX_PAIRINGS="${MAX_PAIRINGS}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  TTAC_TEST_LR="0.003" \
  TTAC_TEST_UPDATE_STEPS="3" \
  TTAC_UPDATE_INTERVAL="${interval}" \
  TTAC_UPDATE_GATE="none" \
  TTAC_CACHE_ESTIMATOR_TARGET="1" \
  TTAC_V5_AGREEMENT_COEF="20.0" \
  TTAC_V5_SUPPORT_COEF="1.0" \
  TTAC_TEST_EGO_KL_COEF="0.01" \
  TTAC_TEST_CUR_KL_COEF="0.01" \
  TTAC_TEST_HIST_KL_COEF="0.0" \
  TTAC_V5_8_LOGIT_STEP_SIZE="${step_size}" \
  TTAC_V5_8_BIAS_CLIP="2.0" \
  TTAC_V5_8_BIAS_DECAY="0.0" \
  TTAC_V5_8_LATENT_DIM="${latent_dim}" \
  TTAC_V5_8_LATENT_SCALE="${latent_scale}" \
  bash tools/run_ttac_v5_8_fast_pair20.sh > "${MASTER_LOG_DIR}/${name}.stdout.log" 2>&1
  echo "[$(date '+%F %T')] VARIANT_DONE ${name}" | tee -a "${MASTER_LOG_DIR}/queue.log"
}

run_variant "base_no_adapt" "base_no_test_adapt" "1" "0.003" "3" "1.0"
run_variant "v56_interval2_cache" "ttac_v5_6_latest_light" "2" "0.003" "3" "1.0"
run_variant "logit_step003_i1" "ttac_v5_8_logit_bias" "1" "0.003" "3" "1.0"
run_variant "logit_step006_i1" "ttac_v5_8_logit_bias" "1" "0.006" "3" "1.0"
run_variant "logit_step003_i2" "ttac_v5_8_logit_bias" "2" "0.003" "3" "1.0"
run_variant "latent3_step003_i1" "ttac_v5_8_latent_bias" "1" "0.003" "3" "1.0"
run_variant "latent6_step003_i1" "ttac_v5_8_latent_bias" "1" "0.003" "6" "1.0"

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
                "mode": r["mode"],
                "interval": cfg.get("TTAC_UPDATE_INTERVAL", ""),
                "step_size": cfg.get("TTAC_V5_8_LOGIT_STEP_SIZE", ""),
                "latent_dim": cfg.get("TTAC_V5_8_LATENT_DIM", ""),
                "xp": float(r["xp"]),
                "seconds": float(r["seconds"]),
            })

base_xp = next(r["xp"] for r in rows if r["variant"] == "base_no_adapt")
out = root / "fast_sweep_summary.csv"
with out.open("w", newline="") as f:
    fieldnames = [
        "variant", "mode", "interval", "step_size", "latent_dim",
        "xp", "gain_vs_base", "seconds", "xp_gain_per_second",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        gain = r["xp"] - base_xp
        writer.writerow({
            "variant": r["variant"],
            "mode": r["mode"],
            "interval": r["interval"],
            "step_size": r["step_size"],
            "latent_dim": r["latent_dim"],
            "xp": f"{r['xp']:.6f}",
            "gain_vs_base": f"{gain:.6f}",
            "seconds": f"{r['seconds']:.0f}",
            "xp_gain_per_second": f"{gain / r['seconds']:.6f}" if r["seconds"] > 0 else "nan",
        })

with out.open(newline="") as f:
    data = list(csv.DictReader(f))
data.sort(key=lambda r: float(r["xp"]), reverse=True)
md = root / "fast_sweep_summary.md"
with md.open("w") as f:
    f.write("# TTAC v5.8 Fast Online Sweep Pair20\n\n")
    f.write("All rows use 20 pairings x 100 eval seeds. v5.8 modes update online logit/latent state without adapter-parameter gradients.\n\n")
    f.write("| variant | mode | XP | gain vs base | seconds | interval | step |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for r in data:
        f.write(
            f"| {r['variant']} | {r['mode']} | {float(r['xp']):.3f} | "
            f"{float(r['gain_vs_base']):+.3f} | {float(r['seconds']):.0f} | "
            f"{r['interval']} | {r['step_size']} |\n"
        )
print(md)
PY

echo "[$(date '+%F %T')] SWEEP_SUMMARY ${MASTER_REPORT_DIR}/fast_sweep_summary.md" | tee -a "${MASTER_LOG_DIR}/queue.log"
