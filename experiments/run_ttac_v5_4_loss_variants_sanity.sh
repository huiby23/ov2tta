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
ESTIMATOR="${ESTIMATOR:-reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_4_loss_variants_sanity_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_4_loss_variants_sanity_${TS}}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-20}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-8}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
TAG_PREFIX="${TAG_PREFIX:-v5_4_sanity}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_V5_2_TV_THRESHOLD="${TTAC_V5_2_TV_THRESHOLD:-0.03}"
TTAC_V5_4_DELTA_MARGIN="${TTAC_V5_4_DELTA_MARGIN:-0.0}"
TTAC_V5_4_DELTA_SCALE="${TTAC_V5_4_DELTA_SCALE:-2.0}"
TTAC_V5_4_DELTA_MASS_THRESHOLD="${TTAC_V5_4_DELTA_MASS_THRESHOLD:-0.03}"
TTAC_V5_4_TV_WEIGHT_FLOOR="${TTAC_V5_4_TV_WEIGHT_FLOOR:-0.25}"
TTAC_V5_4_TV_WEIGHT_SCALE="${TTAC_V5_4_TV_WEIGHT_SCALE:-4.0}"
TTAC_V5_4_CHANGE_AMP="${TTAC_V5_4_CHANGE_AMP:-1.0}"
TTAC_V5_4_STICKY_ACTION="${TTAC_V5_4_STICKY_ACTION:-5}"
TTAC_V5_4_STICKY_SUPPRESS="${TTAC_V5_4_STICKY_SUPPRESS:-0.5}"
TTAC_V5_4_STAY_ACTION="${TTAC_V5_4_STAY_ACTION:-4}"
TTAC_V5_4_ANTI_STAY_COEF="${TTAC_V5_4_ANTI_STAY_COEF:-0.5}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

{
  echo "run_dir=${RUN_DIR}"
  echo "estimator=${ESTIMATOR}"
  echo "seed=${SEED}"
  echo "eval_num_seeds=${EVAL_NUM_SEEDS}"
  echo "eval_max_pairings=${EVAL_MAX_PAIRINGS}"
  echo "eval_batches=${EVAL_BATCHES}"
  echo "ttac_v5_agreement_coef=${TTAC_V5_AGREEMENT_COEF}"
  echo "ttac_v5_2_tv_threshold=${TTAC_V5_2_TV_THRESHOLD}"
  echo "ttac_v5_4_delta_margin=${TTAC_V5_4_DELTA_MARGIN}"
  echo "ttac_v5_4_delta_scale=${TTAC_V5_4_DELTA_SCALE}"
  echo "ttac_v5_4_delta_mass_threshold=${TTAC_V5_4_DELTA_MASS_THRESHOLD}"
  echo "ttac_v5_4_tv_weight_floor=${TTAC_V5_4_TV_WEIGHT_FLOOR}"
  echo "ttac_v5_4_tv_weight_scale=${TTAC_V5_4_TV_WEIGHT_SCALE}"
  echo "ttac_v5_4_change_amp=${TTAC_V5_4_CHANGE_AMP}"
  echo "ttac_v5_4_sticky_action=${TTAC_V5_4_STICKY_ACTION}"
  echo "ttac_v5_4_sticky_suppress=${TTAC_V5_4_STICKY_SUPPRESS}"
  echo "ttac_v5_4_stay_action=${TTAC_V5_4_STAY_ACTION}"
  echo "ttac_v5_4_anti_stay_coef=${TTAC_V5_4_ANTI_STAY_COEF}"
} > "${REPORT_DIR}/config.txt"

run_eval() {
  local mode="$1"
  local tag="${TAG_PREFIX}_${mode}"
  log "EVAL_START mode=${mode} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_4_loss_variants/utils/visualize_ppo.py \
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
    --ttac_test_ego_kl_coef 0.01 \
    --ttac_test_cur_kl_coef 0.01 \
    --ttac_test_hist_kl_coef 0.0 \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    --ttac_v5_2_tv_threshold "${TTAC_V5_2_TV_THRESHOLD}" \
    --ttac_v5_4_delta_margin "${TTAC_V5_4_DELTA_MARGIN}" \
    --ttac_v5_4_delta_scale "${TTAC_V5_4_DELTA_SCALE}" \
    --ttac_v5_4_delta_mass_threshold "${TTAC_V5_4_DELTA_MASS_THRESHOLD}" \
    --ttac_v5_4_tv_weight_floor "${TTAC_V5_4_TV_WEIGHT_FLOOR}" \
    --ttac_v5_4_tv_weight_scale "${TTAC_V5_4_TV_WEIGHT_SCALE}" \
    --ttac_v5_4_change_amp "${TTAC_V5_4_CHANGE_AMP}" \
    --ttac_v5_4_sticky_action "${TTAC_V5_4_STICKY_ACTION}" \
    --ttac_v5_4_sticky_suppress "${TTAC_V5_4_STICKY_SUPPRESS}" \
    --ttac_v5_4_stay_action "${TTAC_V5_4_STAY_ACTION}" \
    --ttac_v5_4_anti_stay_coef "${TTAC_V5_4_ANTI_STAY_COEF}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode}"
}

DEFAULT_MODES="base_no_test_adapt ttac_v5_2_latest ttac_v5_4_delta ttac_v5_4_delta_change_amp ttac_v5_4_delta_move_amp ttac_v5_4_delta_suppress_sticky ttac_v5_4_delta_tv_weighted ttac_v5_4_delta_tv_gate ttac_v5_4_delta_mass_gate ttac_v5_4_delta_tv_mass_gate ttac_v5_4_delta_support ttac_v5_4_low_value_delta ttac_v5_4_anti_stay"
MODE_LIST="${MODE_LIST:-${DEFAULT_MODES}}"
for mode in ${MODE_LIST}; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" TAG_PREFIX="${TAG_PREFIX}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import numpy as np

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
prefix = os.environ.get("TAG_PREFIX", "v5_4_sanity")
rows = []
for p in sorted(run.glob(f"reward_summary_cross_{prefix}_*.csv")):
    mode = p.name.removeprefix(f"reward_summary_cross_{prefix}_").removesuffix(".csv")
    data = defaultdict(list)
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            data[r["policy_labels"]].append(float(r["total_reward"]))
    pair_means = [float(np.mean(v)) for v in data.values()]
    rows.append({
        "mode": mode,
        "mean": float(np.mean(pair_means)),
        "std_pair": float(np.std(pair_means)),
        "num_pairs": len(pair_means),
        "csv": str(p),
    })

base = next((r["mean"] for r in rows if r["mode"] == "base_no_test_adapt"), None)
latest = next((r["mean"] for r in rows if r["mode"] == "ttac_v5_2_latest"), None)
for r in rows:
    r["delta_vs_base"] = "" if base is None else float(r["mean"] - base)
    r["delta_vs_v5_2_latest"] = "" if latest is None else float(r["mean"] - latest)

rows = sorted(rows, key=lambda r: r["mode"])
fieldnames = [
    "mode",
    "mean",
    "delta_vs_base",
    "delta_vs_v5_2_latest",
    "std_pair",
    "num_pairs",
    "csv",
]
with (report / "reward_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md = ["# TTAC v5.4 loss variants sanity", ""]
md.append("| mode | XP mean | vs base | vs v5.2 latest | std_pair | num_pairs |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    db = r["delta_vs_base"]
    dl = r["delta_vs_v5_2_latest"]
    dbs = "" if db == "" else f"{db:.3f}"
    dls = "" if dl == "" else f"{dl:.3f}"
    md.append(
        f"| {r['mode']} | {r['mean']:.3f} | {dbs} | {dls} | {r['std_pair']:.3f} | {r['num_pairs']} |"
    )
(report / "reward_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY

log "PIPELINE_DONE report=${REPORT_DIR}"
