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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_8_full_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_8_full_eval_${TS}}"

MODE="${MODE:-ttac_v5_8_logit_bias}"
OUTPUT_TAG="${OUTPUT_TAG:-v5_8_full_${MODE}_${TS}}"
SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
SEED_CHUNKS="${SEED_CHUNKS:-5}"
SEED_CHUNK_STRIDE="${SEED_CHUNK_STRIDE:-1000}"
TOTAL_PAIRINGS="${TOTAL_PAIRINGS:-100}"
SHARD_SIZE="${SHARD_SIZE:-20}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESET_QUEUE="${RESET_QUEUE:-1}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_UPDATE_INTERVAL="${TTAC_UPDATE_INTERVAL:-1}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_CACHE_ESTIMATOR_TARGET="${TTAC_CACHE_ESTIMATOR_TARGET:-1}"
TTAC_V5_8_LOGIT_STEP_SIZE="${TTAC_V5_8_LOGIT_STEP_SIZE:-0.003}"
TTAC_V5_8_BIAS_CLIP="${TTAC_V5_8_BIAS_CLIP:-2.0}"
TTAC_V5_8_BIAS_DECAY="${TTAC_V5_8_BIAS_DECAY:-0.0}"
TTAC_V5_8_LATENT_DIM="${TTAC_V5_8_LATENT_DIM:-6}"
TTAC_V5_8_LATENT_SCALE="${TTAC_V5_8_LATENT_SCALE:-1.0}"
TTAC_LATENT_DECODER_PATH="${TTAC_LATENT_DECODER_PATH:-}"
TTAC_LATENT_DECODER_HISTORY_LEN="${TTAC_LATENT_DECODER_HISTORY_LEN:-50}"
TTAC_LATENT_DECODER_HISTORY_MODE="${TTAC_LATENT_DECODER_HISTORY_MODE:-full}"
TTAC_LATENT_DECODER_BLEND_ALPHA="${TTAC_LATENT_DECODER_BLEND_ALPHA:-0.5}"
TTAC_LATENT_DECODER_BLEND_CLIP="${TTAC_LATENT_DECODER_BLEND_CLIP:-2.0}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
if [[ "${RESET_QUEUE}" == "1" || "${RESET_QUEUE}" == "true" ]]; then
  : > "${QUEUE}"
else
  touch "${QUEUE}"
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

cat > "${REPORT_DIR}/config.env" <<CFG
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
MODE=${MODE}
OUTPUT_TAG=${OUTPUT_TAG}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
SEED_CHUNKS=${SEED_CHUNKS}
SEED_CHUNK_STRIDE=${SEED_CHUNK_STRIDE}
TOTAL_PAIRINGS=${TOTAL_PAIRINGS}
SHARD_SIZE=${SHARD_SIZE}
PARALLEL_JOBS=${PARALLEL_JOBS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_UPDATE_INTERVAL=${TTAC_UPDATE_INTERVAL}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TTAC_CACHE_ESTIMATOR_TARGET=${TTAC_CACHE_ESTIMATOR_TARGET}
TTAC_V5_8_LOGIT_STEP_SIZE=${TTAC_V5_8_LOGIT_STEP_SIZE}
TTAC_V5_8_BIAS_CLIP=${TTAC_V5_8_BIAS_CLIP}
TTAC_V5_8_BIAS_DECAY=${TTAC_V5_8_BIAS_DECAY}
TTAC_V5_8_LATENT_DIM=${TTAC_V5_8_LATENT_DIM}
TTAC_V5_8_LATENT_SCALE=${TTAC_V5_8_LATENT_SCALE}
TTAC_LATENT_DECODER_PATH=${TTAC_LATENT_DECODER_PATH}
TTAC_LATENT_DECODER_HISTORY_LEN=${TTAC_LATENT_DECODER_HISTORY_LEN}
TTAC_LATENT_DECODER_HISTORY_MODE=${TTAC_LATENT_DECODER_HISTORY_MODE}
TTAC_LATENT_DECODER_BLEND_ALPHA=${TTAC_LATENT_DECODER_BLEND_ALPHA}
TTAC_LATENT_DECODER_BLEND_CLIP=${TTAC_LATENT_DECODER_BLEND_CLIP}
CFG

run_shard() {
  local start="$1"
  local count="$2"
  local seed_chunk="$3"
  local chunk_seed="$4"
  local shard_tag="${OUTPUT_TAG}_seedchunk${seed_chunk}_seed${chunk_seed}_shard${start}_${count}"
  local shard_log="${LOG_DIR}/eval_${shard_tag}.log"
  local shard_csv="${RUN_DIR}/reward_summary_cross_${shard_tag}.csv"
  if [[ ("${SKIP_EXISTING}" == "1" || "${SKIP_EXISTING}" == "true") && -s "${shard_csv}" ]]; then
    log "SHARD_SKIP existing=${shard_csv}"
    return 0
  fi
  log "SHARD_START seed_chunk=${seed_chunk} seed=${chunk_seed} start=${start} count=${count} mode=${MODE} tag=${shard_tag}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${chunk_seed}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${MODE}" \
    --output_tag "${shard_tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --pairing_start "${start}" \
    --max_pairings "${count}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_update_interval "${TTAC_UPDATE_INTERVAL}" \
    --ttac_update_gate none \
    --ttac_cache_estimator_target "${TTAC_CACHE_ESTIMATOR_TARGET}" \
    --ttac_v5_8_logit_step_size "${TTAC_V5_8_LOGIT_STEP_SIZE}" \
    --ttac_v5_8_bias_clip "${TTAC_V5_8_BIAS_CLIP}" \
    --ttac_v5_8_bias_decay "${TTAC_V5_8_BIAS_DECAY}" \
    --ttac_v5_8_latent_dim "${TTAC_V5_8_LATENT_DIM}" \
    --ttac_v5_8_latent_scale "${TTAC_V5_8_LATENT_SCALE}" \
    --ttac_latent_decoder_path "${TTAC_LATENT_DECODER_PATH}" \
    --ttac_latent_decoder_history_len "${TTAC_LATENT_DECODER_HISTORY_LEN}" \
    --ttac_latent_decoder_history_mode "${TTAC_LATENT_DECODER_HISTORY_MODE}" \
    --ttac_latent_decoder_blend_alpha "${TTAC_LATENT_DECODER_BLEND_ALPHA}" \
    --ttac_latent_decoder_blend_clip "${TTAC_LATENT_DECODER_BLEND_CLIP}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    > "${shard_log}" 2>&1
  log "SHARD_DONE seed_chunk=${seed_chunk} seed=${chunk_seed} start=${start} count=${count} tag=${shard_tag}"
}

active_jobs=0
for ((seed_chunk = 0; seed_chunk < SEED_CHUNKS; seed_chunk += 1)); do
  chunk_seed=$((SEED + seed_chunk * SEED_CHUNK_STRIDE))
  for ((start = 0; start < TOTAL_PAIRINGS; start += SHARD_SIZE)); do
    count="${SHARD_SIZE}"
    if (( start + count > TOTAL_PAIRINGS )); then
      count=$((TOTAL_PAIRINGS - start))
    fi
    run_shard "${start}" "${count}" "${seed_chunk}" "${chunk_seed}" &
    active_jobs=$((active_jobs + 1))
    if (( active_jobs >= PARALLEL_JOBS )); then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
done
wait

log "MERGE_START output_tag=${OUTPUT_TAG}"
REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" OUTPUT_TAG="${OUTPUT_TAG}" TOTAL_PAIRINGS="${TOTAL_PAIRINGS}" SHARD_SIZE="${SHARD_SIZE}" SEED="${SEED}" SEED_CHUNKS="${SEED_CHUNKS}" SEED_CHUNK_STRIDE="${SEED_CHUNK_STRIDE}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv, os, re, statistics
run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
tag = os.environ["OUTPUT_TAG"]
total_pairings = int(os.environ["TOTAL_PAIRINGS"])
shard_size = int(os.environ["SHARD_SIZE"])
base_seed = int(os.environ["SEED"])
seed_chunks = int(os.environ["SEED_CHUNKS"])
seed_chunk_stride = int(os.environ["SEED_CHUNK_STRIDE"])
shard_paths = []
for seed_chunk in range(seed_chunks):
    chunk_seed = base_seed + seed_chunk * seed_chunk_stride
    for start in range(0, total_pairings, shard_size):
        count = min(shard_size, total_pairings - start)
        p = run / f"reward_summary_cross_{tag}_seedchunk{seed_chunk}_seed{chunk_seed}_shard{start}_{count}.csv"
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
            raise ValueError(f"Header mismatch in {p}")
        rows.extend(list(reader))
merged_path = run / f"reward_summary_cross_{tag}_merged.csv"
with merged_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
label_idx = header.index("policy_labels")
reward_idx = header.index("total_reward")
pair_values = defaultdict(list)
for row in rows:
    pair_values[row[label_idx]].append(float(row[reward_idx]))
def kind(label):
    m = re.search(r"cross-(\d+)_(\d+)$", label)
    if not m:
        return "unknown"
    return "sp" if m.group(1) == m.group(2) else "xp"
per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
sp_vals = [v for k, v in per_pair.items() if kind(k) == "sp"]
xp_vals = [v for k, v in per_pair.items() if kind(k) == "xp"]
all_vals = list(per_pair.values())
summary_rows = [
    ("files", len(shard_paths)),
    ("rows", len(rows)),
    ("pairs", len(per_pair)),
    ("SP", statistics.fmean(sp_vals) if sp_vals else float("nan")),
    ("XP", statistics.fmean(xp_vals) if xp_vals else float("nan")),
    ("ALL", statistics.fmean(all_vals) if all_vals else float("nan")),
]
summary_csv = report / "sharded_full_eval_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(summary_rows)
summary_md = report / "sharded_full_eval_summary.md"
with summary_md.open("w") as f:
    f.write("# TTAC v5.8 Sharded Full Eval Summary\n\n")
    f.write("| metric | value |\n|---|---:|\n")
    for k, v in summary_rows:
        if isinstance(v, float):
            f.write(f"| {k} | {v:.3f} |\n")
        else:
            f.write(f"| {k} | {v} |\n")
print(summary_md)
PY
log "MERGE_DONE ${REPORT_DIR}/sharded_full_eval_summary.md"
