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
ESTIMATOR="${ESTIMATOR:-reports/ttac_v5_agreement_estimator_20260609_180926/agreement_estimator/agreement_estimator.npz}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_2_sharded_full_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_2_sharded_full_eval_${TS}}"

MODE="${MODE:?Set MODE, e.g. ttac_v5_2_tv_gate}"
OUTPUT_TAG="${OUTPUT_TAG:-sharded_${MODE}_${TS}}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-500}"
TOTAL_PAIRINGS="${TOTAL_PAIRINGS:-100}"
SHARD_SIZE="${SHARD_SIZE:-10}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TV_THRESHOLD="${TV_THRESHOLD:-0.03}"
FUNCTIONAL_EVAL="${FUNCTIONAL_EVAL:-0}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

cat > "${REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
MODE=${MODE}
OUTPUT_TAG=${OUTPUT_TAG}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
TOTAL_PAIRINGS=${TOTAL_PAIRINGS}
SHARD_SIZE=${SHARD_SIZE}
PARALLEL_JOBS=${PARALLEL_JOBS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TV_THRESHOLD=${TV_THRESHOLD}
FUNCTIONAL_EVAL=${FUNCTIONAL_EVAL}
EOF

run_shard() {
  local start="$1"
  local count="$2"
  local shard_tag="${OUTPUT_TAG}_shard${start}_${count}"
  local shard_log="${LOG_DIR}/eval_${shard_tag}.log"
  local functional_args=()
  if [[ "${FUNCTIONAL_EVAL}" == "1" || "${FUNCTIONAL_EVAL}" == "true" ]]; then
    functional_args+=(--functional_eval)
  fi
  log "SHARD_START start=${start} count=${count} mode=${MODE} tag=${shard_tag}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_2_state_selection/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${MODE}" \
    --output_tag "${shard_tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --pairing_start "${start}" \
    --max_pairings "${count}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    --ttac_v5_2_tv_threshold "${TV_THRESHOLD}" \
    "${functional_args[@]}" \
    > "${shard_log}" 2>&1
  log "SHARD_DONE start=${start} count=${count} tag=${shard_tag}"
}

active_jobs=0
for ((start = 0; start < TOTAL_PAIRINGS; start += SHARD_SIZE)); do
  count="${SHARD_SIZE}"
  if (( start + count > TOTAL_PAIRINGS )); then
    count=$((TOTAL_PAIRINGS - start))
  fi
  run_shard "${start}" "${count}" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= PARALLEL_JOBS )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done
wait

log "MERGE_START output_tag=${OUTPUT_TAG}"
REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" OUTPUT_TAG="${OUTPUT_TAG}" TOTAL_PAIRINGS="${TOTAL_PAIRINGS}" SHARD_SIZE="${SHARD_SIZE}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import statistics

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
tag = os.environ["OUTPUT_TAG"]
total_pairings = int(os.environ["TOTAL_PAIRINGS"])
shard_size = int(os.environ["SHARD_SIZE"])

merged_path = run / f"reward_summary_cross_{tag}_merged.csv"
summary_path = report / "sharded_full_eval_summary.csv"
pair_path = report / "sharded_full_eval_pair_rewards.csv"

shard_paths = []
for start in range(0, total_pairings, shard_size):
    count = min(shard_size, total_pairings - start)
    p = run / f"reward_summary_cross_{tag}_shard{start}_{count}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing shard CSV: {p}")
    shard_paths.append(p)

rows = []
header = None
for p in shard_paths:
    with p.open(newline="") as f:
        reader = csv.reader(f)
        local_header = next(reader)
        if header is None:
            header = local_header
        elif local_header != header:
            raise ValueError(f"Header mismatch in {p}: {local_header} != {header}")
        rows.extend(list(reader))

with merged_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

label_idx = header.index("policy_labels")
reward_idx = header.index("total_reward")
pair_values = defaultdict(list)
for row in rows:
    pair_values[row[label_idx]].append(float(row[reward_idx]))

def label_kind(label):
    m = re.search(r"cross-(\d+)_(\d+)$", label)
    if not m:
        return "unknown"
    return "sp" if m.group(1) == m.group(2) else "xp"

per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
sp_vals = [v for k, v in per_pair.items() if label_kind(k) == "sp"]
xp_vals = [v for k, v in per_pair.items() if label_kind(k) == "xp"]
all_vals = list(per_pair.values())
summary = {
    "tag": tag,
    "xp_mean": statistics.fmean(xp_vals) if xp_vals else float("nan"),
    "sp_mean": statistics.fmean(sp_vals) if sp_vals else float("nan"),
    "all_mean": statistics.fmean(all_vals) if all_vals else float("nan"),
    "xp_pairs": len(xp_vals),
    "sp_pairs": len(sp_vals),
    "num_pairs": len(all_vals),
    "num_rows": len(rows),
    "merged_csv": str(merged_path),
}
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)
with pair_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["policy_labels", "kind", "pair_reward", "num_annotations"]
    )
    writer.writeheader()
    for label, value in sorted(per_pair.items()):
        writer.writerow({
            "policy_labels": label,
            "kind": label_kind(label),
            "pair_reward": value,
            "num_annotations": len(pair_values[label]),
        })
md = [
    "# Sharded TTAC full eval summary",
    "",
    "| metric | value |",
    "|---|---:|",
    f"| xp_mean | {summary['xp_mean']:.3f} |",
    f"| sp_mean | {summary['sp_mean']:.3f} |",
    f"| all_mean | {summary['all_mean']:.3f} |",
    f"| xp_pairs | {summary['xp_pairs']} |",
    f"| sp_pairs | {summary['sp_pairs']} |",
    f"| num_pairs | {summary['num_pairs']} |",
    f"| num_rows | {summary['num_rows']} |",
    "",
    f"Merged CSV: `{merged_path}`",
]
(report / "sharded_full_eval_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
log "MERGE_DONE report=${REPORT_DIR}"
