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

PPO_STANDARD_RUN="${PPO_STANDARD_RUN:-runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
TTAC_RUN_DIR="${TTAC_RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_standard_no_state_aug_64_16_seed42_10seeds_${TS}}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_no_state_aug_pair20_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_no_state_aug_pair20_${TS}}"
SEED="${SEED:-42}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
NUM_DATASET_SEEDS="${NUM_DATASET_SEEDS:-100}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-250000}"
TRANSITION_STRIDE="${TRANSITION_STRIDE:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
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

DATASET_DIR="${REPORT_DIR}/dataset"
ESTIMATOR_DIR="${REPORT_DIR}/agreement_estimator"
DATASET="${DATASET_DIR}/agreement_dataset.npz"
ESTIMATOR="${ESTIMATOR_DIR}/agreement_estimator.npz"

{
  echo "source_ppo_standard_run=${PPO_STANDARD_RUN}"
  echo "ttac_zero_adapter_run=${TTAC_RUN_DIR}"
  echo "state_aug=no"
  echo "eval_pairings=${EVAL_MAX_PAIRINGS}"
  echo "eval_num_seeds=${EVAL_NUM_SEEDS}"
} > "${REPORT_DIR}/experiment_config.txt"

log "STAGE0_CONVERT_START source=${PPO_STANDARD_RUN} output=${TTAC_RUN_DIR}"
if [ ! -d "${TTAC_RUN_DIR}/run_0/ckpt_final" ]; then
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/convert_ppo_to_ttac_zero_adapter.py \
    --ppo_run_dir "${PPO_STANDARD_RUN}" \
    --output_run_dir "${TTAC_RUN_DIR}" \
    --adapter_scale 0.5 \
    --test_lr "${TTAC_TEST_LR}" \
    --test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --history_len "${HISTORY_LEN}" \
    > "${LOG_DIR}/convert_ppo_standard_to_ttac.log" 2>&1
  {
    echo
    echo "## No-State-Aug Note"
    echo
    echo "This conversion source is PPO CNN standard/no-state-aug, not PPO state-aug."
    echo "The historical converter name mentions state-aug, but the source run above is the authoritative setting."
  } >> "${TTAC_RUN_DIR}/conversion_manifest.md"
else
  log "STAGE0_CONVERT_SKIP existing=${TTAC_RUN_DIR}"
fi
log "STAGE0_CONVERT_DONE"

log "STAGE1_DATASET_START pairs=${MAX_PAIRINGS} seeds=${NUM_DATASET_SEEDS} max_transitions=${MAX_TRANSITIONS}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/collect_agreement_dataset.py \
  --run_dir "${TTAC_RUN_DIR}" \
  --output_dir "${DATASET_DIR}" \
  --seed "${SEED}" \
  --num_eval_seeds "${NUM_DATASET_SEEDS}" \
  --max_pairings "${MAX_PAIRINGS}" \
  --transition_stride "${TRANSITION_STRIDE}" \
  --max_transitions "${MAX_TRANSITIONS}" \
  --history_len "${HISTORY_LEN}" \
  > "${LOG_DIR}/collect_agreement_dataset.log" 2>&1
log "STAGE1_DATASET_DONE dataset=${DATASET}"

log "STAGE2_ESTIMATOR_TRAIN_START steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
"${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/train_agreement_estimator.py \
  --dataset "${DATASET}" \
  --output_dir "${ESTIMATOR_DIR}" \
  --seed "${SEED}" \
  --steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  > "${LOG_DIR}/train_agreement_estimator.log" 2>&1
log "STAGE2_ESTIMATOR_TRAIN_DONE estimator=${ESTIMATOR}"

run_eval() {
  local mode="$1"
  local tag="v5_2_no_state_${mode}"
  log "EVAL_START mode=${mode} seeds=${EVAL_NUM_SEEDS} pairings=${EVAL_MAX_PAIRINGS}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/visualize_ppo.py \
    --d "${TTAC_RUN_DIR}" \
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

log "STAGE3_PAIR20_EVAL_START modes=${MODE_LIST}"
for mode in ${MODE_LIST}; do
  run_eval "${mode}"
done

REPORT_DIR="${REPORT_DIR}" TTAC_RUN_DIR="${TTAC_RUN_DIR}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import numpy as np

run = Path(os.environ["TTAC_RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
rows = []
for p in sorted(run.glob("reward_summary_cross_v5_2_no_state_*.csv")):
    mode = p.name.removeprefix("reward_summary_cross_v5_2_no_state_").removesuffix(".csv")
    data = defaultdict(list)
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            data[r["policy_labels"]].append(float(r["total_reward"]))
    vals = [float(np.mean(v)) for v in data.values()]
    rows.append({
        "mode": mode,
        "mean": float(np.mean(vals)),
        "std_pair": float(np.std(vals)),
        "num_pairs": len(vals),
        "csv": str(p),
    })
rows = sorted(rows, key=lambda r: r["mode"])
with (report / "pair20_reward_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "mean", "std_pair", "num_pairs", "csv"])
    writer.writeheader()
    writer.writerows(rows)
md = ["# TTAC v5.2 no-state-aug pair20 reward summary", ""]
md.append("| mode | XP mean | std_pair | num_pairs |")
md.append("|---|---:|---:|---:|")
for r in rows:
    md.append(f"| {r['mode']} | {r['mean']:.3f} | {r['std_pair']:.3f} | {r['num_pairs']} |")
(report / "pair20_reward_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
log "PIPELINE_DONE report=${REPORT_DIR} ttac_run=${TTAC_RUN_DIR}"
