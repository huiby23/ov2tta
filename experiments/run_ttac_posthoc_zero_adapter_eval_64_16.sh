#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S_posthoc_zero_adapter)}"

cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_posthoc_zero_adapter_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_posthoc_zero_adapter_eval_${TS}}"
SEED="${SEED:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
ONLINE_EVAL_PAIRINGS_PER_CHUNK="${ONLINE_EVAL_PAIRINGS_PER_CHUNK:-8}"

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

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

run_eval() {
  local mode="$1"
  local split="$2"
  local cross_args=()
  local suffix="sp"
  if [ "${split}" = "cross" ]; then
    cross_args=(--cross)
    suffix="cross"
  fi
  local tag="posthoc_zero_adapter_${mode}_${suffix}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  log "EVAL_START mode=${mode} split=${split} seeds=${EVAL_SEEDS} batches=${EVAL_BATCHES}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo \
    --d "${RUN_DIR}" "${cross_args[@]}" --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" \
    --ttac_mode "${mode}" --output_tag "${tag}" --eval_batches "${EVAL_BATCHES}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --online_eval_pairings_per_chunk "${ONLINE_EVAL_PAIRINGS_PER_CHUNK}" > "${log_file}" 2>&1
  log "EVAL_DONE mode=${mode} split=${split}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

report = Path(os.environ["REPORT_DIR"])
run_dir = Path(os.environ["RUN_DIR"])
rows = []
for csv in sorted(run_dir.glob("reward_summary_posthoc_zero_adapter_*.csv")):
    name = csv.stem.replace("reward_summary_posthoc_zero_adapter_", "")
    df = pd.read_csv(csv)
    split = "XP" if name.endswith("_cross") else "SP"
    mode = name.rsplit("_cross" if split == "XP" else "_sp", 1)[0]
    row = {
        "mode": mode,
        "split": split,
        "mean_reward": float(df.total_reward.mean()),
        "std_reward": float(df.total_reward.std()),
        "min_reward": float(df.total_reward.min()),
        "max_reward": float(df.total_reward.max()),
        "n": int(len(df)),
        "csv": str(csv),
    }
    if split == "XP" and "policy_labels" in df.columns:
        ids = df.policy_labels.str.extract(r"cross-(\d+)_(\d+)").astype(int)
        xp = df[ids[0] != ids[1]]
        diag = df[ids[0] == ids[1]]
        row["xp90_mean"] = float(xp.total_reward.mean())
        row["diag10_mean"] = float(diag.total_reward.mean())
    rows.append(row)
out = pd.DataFrame(rows)
if len(out):
    out = out.sort_values(["split", "mode"])
    out.to_csv(report / "posthoc_zero_adapter_eval_summary.csv", index=False)
    try:
        table = out.to_markdown(index=False)
    except Exception:
        table = out.to_csv(index=False)
    (report / "posthoc_zero_adapter_eval_summary.md").write_text(
        "# TTAC Posthoc Zero-Adapter Eval\n\n"
        f"run_dir: `{run_dir}`\n\n" + table + "\n"
    )
    print(out.to_string(index=False))
else:
    print("No posthoc zero-adapter CSVs found.")
PY
}

log "PIPELINE_START run_dir=${RUN_DIR} report=${REPORT_DIR}"
for mode in ttac_adapter_off base_no_test_adapt ttac_advantage_weighted ttac_projected_confident; do
  run_eval "${mode}" sp
  run_eval "${mode}" cross
  summarize | tee -a "${QUEUE}"
done
log "PIPELINE_DONE"
