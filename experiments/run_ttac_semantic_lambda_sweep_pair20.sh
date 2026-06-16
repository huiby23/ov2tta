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
REPORT_DIR="${REPORT_DIR:-reports/ttac_semantic_surrogate_sweep_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_semantic_surrogate_sweep_${TS}}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
LAMBDAS="${LAMBDAS:-0.0 0.25 0.5 1.0 2.0}"

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA:-2.0}"
TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}"
TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER:-1.0}"
TTAC_TEST_CONTRAST_BETA="${TTAC_TEST_CONTRAST_BETA:-1.0}"
TTAC_TEST_CONTRAST_FLOOR="${TTAC_TEST_CONTRAST_FLOOR:-0.0}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"

tag_lambda() { echo "$1" | sed 's/-/m/g; s/\./p/g'; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

run_eval() {
  local lambda="$1"
  local mode="$2"
  local tag="lam$(tag_lambda "${lambda}")"
  local out_tag="semantic_sweep_${tag}_${mode}"
  local expected="${RUN_DIR}/reward_summary_cross_${out_tag}.csv"
  local fallback="${RUN_DIR}/reward_summary_cross_semantic_pair20_100_${mode}.csv"
  if [ -f "${expected}" ]; then
    log "SKIP existing lambda=${lambda} mode=${mode} csv=${expected}"
    return 0
  fi
  if [ "${lambda}" = "0.25" ] && [ -f "${fallback}" ]; then
    log "REUSE lambda=0.25 mode=${mode} source=${fallback}"
    return 0
  fi
  log "EVAL_START lambda=${lambda} mode=${mode} seeds=${NUM_SEEDS} pairings=${MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v2/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${out_tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --max_pairings "${MAX_PAIRINGS}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" \
    --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" \
    --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --ttac_test_contrast_beta "${TTAC_TEST_CONTRAST_BETA}" \
    --ttac_test_contrast_floor "${TTAC_TEST_CONTRAST_FLOOR}" \
    --ttac_test_semantic_lambda "${lambda}" \
    > "${LOG_DIR}/eval_${out_tag}.log" 2>&1
  log "EVAL_DONE lambda=${lambda} mode=${mode}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" LAMBDAS="${LAMBDAS}" "${PYTHON}" - <<'PY'
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
lambdas = [float(x) for x in os.environ["LAMBDAS"].split()]

modes = [
    "base_no_test_adapt",
    "ttac_ego_advantage_weighted",
    "ttac_semantic_ego_aw",
    "ttac_semantic_ego_aw_wrong_history",
    "ttac_semantic_ego_aw_delayed_history",
    "ttac_semantic_ego_aw_random_history",
]
short = {
    "base_no_test_adapt": "base",
    "ttac_ego_advantage_weighted": "ego_aw",
    "ttac_semantic_ego_aw": "true",
    "ttac_semantic_ego_aw_wrong_history": "wrong",
    "ttac_semantic_ego_aw_delayed_history": "delayed",
    "ttac_semantic_ego_aw_random_history": "random",
}

def tag_lambda(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")

def csv_for(lambda_value: float, mode: str) -> Path | None:
    tag = tag_lambda(lambda_value)
    candidates = [run / f"reward_summary_cross_semantic_sweep_lam{tag}_{mode}.csv"]
    if abs(lambda_value - 0.25) < 1e-9:
        candidates.append(run / f"reward_summary_cross_semantic_pair20_100_{mode}.csv")
    if mode in ("base_no_test_adapt", "ttac_ego_advantage_weighted"):
        candidates.append(run / f"reward_summary_cross_semantic_pair20_100_{mode}.csv")
    for path in candidates:
        if path.exists():
            return path
    return None

def load(path: Path) -> dict[str, float]:
    data = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["policy_labels"]].append(float(row["total_reward"]))
    return {key: float(np.mean(values)) for key, values in data.items()}

rows = []
for lam in lambdas:
    per = {}
    sources = {}
    missing = []
    for mode in modes:
        path = csv_for(lam, mode)
        if path is None:
            missing.append(mode)
            continue
        per[mode] = load(path)
        sources[mode] = str(path)
    if missing:
        rows.append({"lambda": lam, "complete": False, "missing_modes": ";".join(missing)})
        continue
    pairs = sorted(set.intersection(*(set(values) for values in per.values())))
    table = pd.DataFrame([{short[mode]: per[mode][pair] for mode in modes} | {"pair": pair} for pair in pairs])
    true = float(table["true"].mean())
    base = float(table["base"].mean())
    ego_aw = float(table["ego_aw"].mean())
    wrong = float(table["wrong"].mean())
    delayed = float(table["delayed"].mean())
    random = float(table["random"].mean())
    max_wd_pair = np.maximum(table["wrong"].to_numpy(), table["delayed"].to_numpy())
    true_arr = table["true"].to_numpy()
    gap_max_wd = true - max(wrong, delayed)
    winrate_max_wd = float(np.mean(true_arr > max_wd_pair))
    passed = bool(
        true > base
        and true > wrong
        and true > delayed
        and true > random
        and (gap_max_wd >= 2.0 or winrate_max_wd >= 0.75)
    )
    rows.append({
        "lambda": lam,
        "complete": True,
        "missing_modes": "",
        "base_mean": base,
        "ego_aw_mean": ego_aw,
        "true_mean": true,
        "wrong_mean": wrong,
        "delayed_mean": delayed,
        "random_mean": random,
        "true_minus_base": true - base,
        "true_minus_ego_aw": true - ego_aw,
        "true_minus_wrong": true - wrong,
        "true_minus_delayed": true - delayed,
        "true_minus_random": true - random,
        "true_minus_max_wrong_delayed": gap_max_wd,
        "winrate_true_vs_max_wrong_delayed": winrate_max_wd,
        "winrate_true_vs_wrong": float(np.mean(true_arr > table["wrong"].to_numpy())),
        "winrate_true_vs_delayed": float(np.mean(true_arr > table["delayed"].to_numpy())),
        "num_pairs": len(pairs),
        "pass_success_criterion": passed,
        "source_true_csv": sources["ttac_semantic_ego_aw"],
    })

summary = pd.DataFrame(rows).sort_values("lambda")
out_csv = report / "lambda_sweep_summary.csv"
summary.to_csv(out_csv, index=False)
complete = summary[summary["complete"] == True].copy()
if len(complete):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(complete["lambda"], complete["true_minus_base"], marker="o", label="true-base")
    ax1.plot(complete["lambda"], complete["true_minus_max_wrong_delayed"], marker="o", label="true-max(wrong,delayed)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(2.0, color="gray", linewidth=0.8, linestyle="--", label="2 XP target")
    ax1.set_xlabel("TTAC_TEST_SEMANTIC_LAMBDA")
    ax1.set_ylabel("Reward gap")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(complete["lambda"], complete["winrate_true_vs_max_wrong_delayed"], color="tab:green", marker="s", label="win-rate vs max(wrong,delayed)")
    ax2.axhline(0.75, color="tab:green", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Pair win-rate")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report / "lambda_vs_reward_gap.png", dpi=180)
    plt.close(fig)

try:
    table = summary.to_markdown(index=False)
except Exception:
    table = summary.to_csv(index=False)
passed = complete[complete["pass_success_criterion"] == True] if len(complete) else complete
if len(passed):
    best = passed.sort_values(["true_minus_max_wrong_delayed", "winrate_true_vs_max_wrong_delayed"], ascending=False).iloc[0]
    decision = f"PASS: best_lambda={best['lambda']} gap_vs_max_wrong_delayed={best['true_minus_max_wrong_delayed']:.3f} winrate={best['winrate_true_vs_max_wrong_delayed']:.3f}. Run full validation."
else:
    decision = "FAIL/PENDING: no completed lambda satisfies the true-history semantic success criterion. Do not run 500-seed full validation unless this changes."
(report / "lambda_sweep_summary.md").write_text(
    "# TTAC Semantic Lambda Sweep\n\n"
    f"run_dir: `{run}`\n\n"
    f"decision: **{decision}**\n\n"
    + table
    + "\n\nArtifacts:\n"
    f"- `{out_csv}`\n"
    f"- `{report / 'lambda_vs_reward_gap.png'}`\n"
)
print(report / "lambda_sweep_summary.md")
print(summary.to_string(index=False))
print(decision)
PY
}

log "SWEEP_START run_dir=${RUN_DIR} report=${REPORT_DIR} lambdas=${LAMBDAS}"
for lambda in ${LAMBDAS}; do
  for mode in ttac_semantic_ego_aw ttac_semantic_ego_aw_wrong_history ttac_semantic_ego_aw_random_history ttac_semantic_ego_aw_delayed_history; do
    run_eval "${lambda}" "${mode}"
    summarize | tee -a "${QUEUE}"
  done
done
summarize | tee -a "${QUEUE}"
log "SWEEP_DONE"
