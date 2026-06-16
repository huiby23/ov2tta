#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
cd "$ROOT"
source "$ROOT/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export ZSC_FORWARD_CHUNK_SIZE="${ZSC_FORWARD_CHUNK_SIZE:-32768}"

TS="${TS:-$(date +%Y%m%d_%H%M%S_aw_failure_attr)}"
OUT_ROOT="${OUT_ROOT:-reports/ttac_v2_aw_failure_attribution_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v2_aw_failure_attribution_${TS}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
QUEUE="$OUT_ROOT/queue.log"
: > "$QUEUE"

BAD_RUN_DIR="${BAD_RUN_DIR:-runs/ttac_v2_aw_full_state_aug_64_16_10M_seed42_10seeds_20260603_164724_aw_full/20260603-164834_1xmzwzbl_counter_circuit_avs-full}"
FROZEN_REPORT_DIR="${FROZEN_REPORT_DIR:-reports/ttac_v2_frozen_adapter_aw_eval_20260603_210048_frozen_adapter}"
FROZEN_QUEUE="${FROZEN_QUEUE:-logs/ttac_v2_frozen_adapter_aw_eval_20260603_210048_frozen_adapter/queue.log}"

NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-50}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-20}"
MAX_PAIRS="${MAX_PAIRS:-30}"
TOP_K="${TOP_K:-10}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-256}"

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA:-2.0}"
TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}"
TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER:-1.0}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE"; }

wait_for_frozen_pipeline() {
  log "WAIT frozen pipeline queue=$FROZEN_QUEUE"
  while true; do
    if [ -f "$FROZEN_QUEUE" ] && grep -q "PIPELINE_DONE" "$FROZEN_QUEUE"; then
      break
    fi
    if ! ps -eo args | grep -E "run_ttac_v2_frozen_adapter_aw_eval_64_16|ttac_v2_frozen_adapter_full_state_aug" | grep -v grep >/dev/null; then
      if [ -f "$FROZEN_REPORT_DIR/run_dir_full.txt" ]; then
        log "Frozen process not found; continuing because run_dir_full exists."
        break
      fi
    fi
    sleep 180
  done
  log "WAIT_DONE"
}

run_diag() {
  local name="$1" run_dir="$2" mode="$3"
  local out="$OUT_ROOT/$name"
  local log_file="$LOG_DIR/$name.log"
  mkdir -p "$out"
  log "DIAG_START name=$name mode=$mode run_dir=$run_dir"
  "$PYTHON" experiments/overcooked_v2_experiments/ttac_v2/utils/zsc_diagnostics.py \
    --single_run_dir "$run_dir" \
    --single_name "$name" \
    --single_backend ttac \
    --single_eval_mode "$mode" \
    --layout counter_circuit \
    --eval_seed 42 \
    --num_eval_seeds "$NUM_EVAL_SEEDS" \
    --num_diag_episodes "$NUM_DIAG_EPISODES" \
    --top_k "$TOP_K" \
    --compatibility_sample_limit "$COMPATIBILITY_SAMPLE_LIMIT" \
    --max_pairs "$MAX_PAIRS" \
    --ttac_adapter_scale "$TTAC_ADAPTER_SCALE" \
    --ttac_test_hist_kl_coef "$TTAC_TEST_HIST_KL_COEF" \
    --ttac_test_ego_kl_coef "$TTAC_TEST_EGO_KL_COEF" \
    --ttac_test_cur_kl_coef "$TTAC_TEST_CUR_KL_COEF" \
    --ttac_test_lr "$TTAC_TEST_LR" \
    --ttac_test_update_steps "$TTAC_TEST_UPDATE_STEPS" \
    --ttac_history_len "$TTAC_HISTORY_LEN" \
    --ttac_test_project_beta "$TTAC_TEST_PROJECT_BETA" \
    --ttac_test_support_min_prob "$TTAC_TEST_SUPPORT_MIN_PROB" \
    --ttac_test_support_max_entropy "$TTAC_TEST_SUPPORT_MAX_ENTROPY" \
    --ttac_test_advantage_power "$TTAC_TEST_ADVANTAGE_POWER" \
    --output_dir "$out" > "$log_file" 2>&1
  log "DIAG_DONE name=$name"
}

summarize() {
  OUT_ROOT="$OUT_ROOT" BAD_RUN_DIR="$BAD_RUN_DIR" FROZEN_RUN_DIR="$FROZEN_RUN_DIR" "$PYTHON" - <<'PY'
from pathlib import Path
import csv
import math
import os
import re

import pandas as pd

root = Path(os.environ["OUT_ROOT"])
runs = {
    "aw_train": Path(os.environ["BAD_RUN_DIR"]),
    "frozen_train": Path(os.environ["FROZEN_RUN_DIR"]),
}
methods = [
    ("aw_train_no_adapt", "aw_train", "base_no_test_adapt"),
    ("aw_train_online_aw", "aw_train", "ttac_advantage_weighted"),
    ("frozen_train_no_adapt", "frozen_train", "base_no_test_adapt"),
    ("frozen_train_online_aw", "frozen_train", "ttac_advantage_weighted"),
]


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return f"{v:.4f}"
    return str(v)


def read_reward(path, method, field):
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get("method") == method and row.get(field, ""):
                    return float(row[field])
    except FileNotFoundError:
        pass
    return math.nan


def mean_field(path, method, field, row_type=None):
    vals = []
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get("method") != method:
                    continue
                if row_type is not None and row.get("row_type") != row_type:
                    continue
                if row.get(field, "") == "":
                    continue
                vals.append(float(row[field]))
    except FileNotFoundError:
        return math.nan
    return sum(vals) / len(vals) if vals else math.nan


def full_eval_stats(run_dir, mode):
    sp_csv = run_dir / f"reward_summary_sp_full_{mode}_sp.csv"
    cross_csv = run_dir / f"reward_summary_cross_full_{mode}_cross.csv"
    stats = {
        "Full_SP": math.nan,
        "Full_XP_all100": math.nan,
        "Full_XP90": math.nan,
        "Full_diag10": math.nan,
    }
    if sp_csv.exists():
        sp = pd.read_csv(sp_csv)
        stats["Full_SP"] = float(sp.total_reward.mean())
    if cross_csv.exists():
        cross = pd.read_csv(cross_csv)
        stats["Full_XP_all100"] = float(cross.total_reward.mean())
        ids = cross.policy_labels.str.extract(r"cross-(\d+)_(\d+)").astype(int)
        xp = cross[ids[0] != ids[1]]
        diag = cross[ids[0] == ids[1]]
        stats["Full_XP90"] = float(xp.total_reward.mean())
        stats["Full_diag10"] = float(diag.total_reward.mean())
    return stats


rows = []
for name, run_key, mode in methods:
    diag = root / name
    row = {
        "method": name,
        "run_group": run_key,
        "mode": mode,
        **full_eval_stats(runs[run_key], mode),
        "Diag_SP": read_reward(diag / "reward_summary.csv", name, "sp_mean"),
        "Diag_XP": read_reward(diag / "reward_summary.csv", name, "xp_mean"),
        "OOS": mean_field(diag / "coverage_summary.csv", name, "xp_out_of_sp_support_rate", "cross_coverage"),
        "InBoth": mean_field(diag / "coverage_summary.csv", name, "in_both_support_rate", "cross_coverage"),
        "Agreement": mean_field(diag / "shared_state_mismatch.csv", name, "action_agreement"),
        "Policy_TV": mean_field(diag / "shared_state_mismatch.csv", name, "policy_tv"),
        "Symmetric_KL": mean_field(diag / "shared_state_mismatch.csv", name, "symmetric_kl"),
        "Value_shift": mean_field(diag / "shared_state_mismatch.csv", name, "value_shift"),
        "Joint_optimal_rate": mean_field(diag / "complementarity_summary.csv", name, "joint_optimal_rate"),
        "Joint_regret": mean_field(diag / "complementarity_summary.csv", name, "joint_regret"),
        "Unilateral_regret": mean_field(diag / "complementarity_summary.csv", name, "unilateral_regret"),
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(root / "summary.csv", index=False)

lines = [
    "# TTACv2 AW Failure Attribution",
    "",
    f"- bad AW-train run: `{runs['aw_train']}`",
    f"- frozen-train run: `{runs['frozen_train']}`",
    "",
    "Full rewards use existing 500-seed evaluation CSVs. OOS/Agreement/TV/Regret use a matched diagnostics config.",
    "",
    "| Method | Full SP | Full XP90 | OOS | InBoth | Agreement | Policy TV | Joint regret |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for _, r in df.iterrows():
    lines.append(
        "| "
        + " | ".join(
            [
                r["method"],
                fmt(r["Full_SP"]),
                fmt(r["Full_XP90"]),
                fmt(r["OOS"]),
                fmt(r["InBoth"]),
                fmt(r["Agreement"]),
                fmt(r["Policy_TV"]),
                fmt(r["Joint_regret"]),
            ]
        )
        + " |"
    )


def delta(a_name, b_name, keys):
    a = df[df.method == a_name].iloc[0]
    b = df[df.method == b_name].iloc[0]
    return {k: b[k] - a[k] for k in keys}

keys = [
    "Full_SP",
    "Full_XP90",
    "OOS",
    "InBoth",
    "Agreement",
    "Policy_TV",
    "Symmetric_KL",
    "Joint_optimal_rate",
    "Joint_regret",
    "Unilateral_regret",
]
comparisons = [
    ("AW online effect inside bad AW-train run", "aw_train_no_adapt", "aw_train_online_aw"),
    ("AW online effect inside frozen-train run", "frozen_train_no_adapt", "frozen_train_online_aw"),
    ("Training damage: bad AW-train no-adapt minus frozen-train no-adapt", "frozen_train_no_adapt", "aw_train_no_adapt"),
]
lines += ["", "## Deltas", ""]
for title, a, b in comparisons:
    lines += [f"### {title}", ""]
    d = delta(a, b, keys)
    for k in keys:
        lines.append(f"- {k}: {fmt(d[k])}")
    lines.append("")

(root / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
PY
}

wait_for_frozen_pipeline
FROZEN_RUN_DIR="$(cat "$FROZEN_REPORT_DIR/run_dir_full.txt")"
log "BAD_RUN_DIR=$BAD_RUN_DIR"
log "FROZEN_RUN_DIR=$FROZEN_RUN_DIR"

run_diag aw_train_no_adapt "$BAD_RUN_DIR" base_no_test_adapt
run_diag aw_train_online_aw "$BAD_RUN_DIR" ttac_advantage_weighted
run_diag frozen_train_no_adapt "$FROZEN_RUN_DIR" base_no_test_adapt
run_diag frozen_train_online_aw "$FROZEN_RUN_DIR" ttac_advantage_weighted
summarize | tee -a "$QUEUE"
log "DONE"
