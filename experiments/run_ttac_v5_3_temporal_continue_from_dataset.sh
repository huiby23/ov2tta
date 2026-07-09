#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
REPORT_DIR="${REPORT_DIR:?REPORT_DIR is required}"
RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
LOG_DIR="${LOG_DIR:-logs/$(basename "${REPORT_DIR}")_continue}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

SEED="${SEED:-42}"
DATASET="${REPORT_DIR}/dataset/agreement_dataset.npz"
ESTIMATOR_DIR="${REPORT_DIR}/agreement_estimator_temporal"
ESTIMATOR="${ESTIMATOR_DIR}/agreement_estimator.npz"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
EVAL_MAX_PAIRINGS="${EVAL_MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_V5_2_TV_THRESHOLD="${TTAC_V5_2_TV_THRESHOLD:-0.03}"
MODE_LIST="${MODE_LIST:-base_no_test_adapt ttac_v5_2_latest ttac_v5_2_tv_gate}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

test -f "${DATASET}"

if [ "${SKIP_TRAIN}" = "1" ]; then
  test -f "${ESTIMATOR}"
  log "SKIP_TEMPORAL_ESTIMATOR_TRAIN estimator=${ESTIMATOR}"
else
  log "CONTINUE_TEMPORAL_ESTIMATOR_TRAIN_START dataset=${DATASET} steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/train_agreement_estimator.py \
    --dataset "${DATASET}" \
    --output_dir "${ESTIMATOR_DIR}" \
    --seed "${SEED}" \
    --steps "${TRAIN_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --frame_dim 128 \
    --gru_dim 128 \
    --hidden_dim 256 \
    > "${LOG_DIR}/train_agreement_estimator.log" 2>&1
  log "CONTINUE_TEMPORAL_ESTIMATOR_TRAIN_DONE estimator=${ESTIMATOR}"
fi

run_eval() {
  local mode="$1"
  local tag="v5_3_temporal_${mode}"
  log "EVAL_START mode=${mode} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_3_temporal_estimator/utils/visualize_ppo.py \
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
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE mode=${mode}"
}

for mode in ${MODE_LIST}; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import numpy as np

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
rows = []
for p in sorted(run.glob("reward_summary_cross_v5_3_temporal_*.csv")):
    mode = p.name.removeprefix("reward_summary_cross_v5_3_temporal_").removesuffix(".csv")
    data = defaultdict(list)
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            data[r["policy_labels"]].append(float(r["total_reward"]))
    vals = [float(np.mean(v)) for v in data.values()]
    rows.append({"mode": mode, "mean": float(np.mean(vals)), "std_pair": float(np.std(vals)), "num_pairs": len(vals), "csv": str(p)})
rows = sorted(rows, key=lambda r: r["mode"])
with (report / "pair20_reward_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "mean", "std_pair", "num_pairs", "csv"])
    writer.writeheader()
    writer.writerows(rows)
md = ["# TTAC v5.3 temporal estimator pair20 reward summary", ""]
md.append("| mode | mean | std_pair | num_pairs |")
md.append("|---|---:|---:|---:|")
for r in rows:
    md.append(f"| {r['mode']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report / "pair20_reward_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
log "CONTINUE_DONE report=${REPORT_DIR}"
