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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_8_fast_online_pair20_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_8_fast_online_pair20_${TS}}"

SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_UPDATE_INTERVAL="${TTAC_UPDATE_INTERVAL:-1}"
TTAC_UPDATE_GATE="${TTAC_UPDATE_GATE:-none}"
TTAC_UPDATE_TV_THRESHOLD="${TTAC_UPDATE_TV_THRESHOLD:-0.03}"
TTAC_UPDATE_VALUE_MARGIN="${TTAC_UPDATE_VALUE_MARGIN:-0.0}"
TTAC_CACHE_ESTIMATOR_TARGET="${TTAC_CACHE_ESTIMATOR_TARGET:-1}"
TTAC_V5_8_LOGIT_STEP_SIZE="${TTAC_V5_8_LOGIT_STEP_SIZE:-0.003}"
TTAC_V5_8_BIAS_CLIP="${TTAC_V5_8_BIAS_CLIP:-2.0}"
TTAC_V5_8_BIAS_DECAY="${TTAC_V5_8_BIAS_DECAY:-0.0}"
TTAC_V5_8_LATENT_DIM="${TTAC_V5_8_LATENT_DIM:-3}"
TTAC_V5_8_LATENT_SCALE="${TTAC_V5_8_LATENT_SCALE:-1.0}"
TTAC_LATENT_DECODER_PATH="${TTAC_LATENT_DECODER_PATH:-}"
TTAC_LATENT_DECODER_HISTORY_LEN="${TTAC_LATENT_DECODER_HISTORY_LEN:-50}"
TTAC_LATENT_DECODER_HISTORY_MODE="${TTAC_LATENT_DECODER_HISTORY_MODE:-full}"
TTAC_LATENT_DECODER_BLEND_ALPHA="${TTAC_LATENT_DECODER_BLEND_ALPHA:-0.5}"
TTAC_LATENT_DECODER_BLEND_CLIP="${TTAC_LATENT_DECODER_BLEND_CLIP:-2.0}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
MODES="${MODES:-base_no_test_adapt ttac_v5_2_latest ttac_v5_6_latest_light}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

cat > "${REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
MAX_PAIRINGS=${MAX_PAIRINGS}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_UPDATE_INTERVAL=${TTAC_UPDATE_INTERVAL}
TTAC_UPDATE_GATE=${TTAC_UPDATE_GATE}
TTAC_UPDATE_TV_THRESHOLD=${TTAC_UPDATE_TV_THRESHOLD}
TTAC_UPDATE_VALUE_MARGIN=${TTAC_UPDATE_VALUE_MARGIN}
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
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
MODES=${MODES}
EOF

for mode in ${MODES}; do
  tag="v5_6_light_pair20_${mode}_${TS}"
  log "MODE_START mode=${mode} tag=${tag}"
  start_s=$(date +%s)
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v5_8_fast_online/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --pairing_start 0 \
    --max_pairings "${MAX_PAIRINGS}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_update_interval "${TTAC_UPDATE_INTERVAL}" \
    --ttac_update_gate "${TTAC_UPDATE_GATE}" \
    --ttac_update_tv_threshold "${TTAC_UPDATE_TV_THRESHOLD}" \
    --ttac_update_value_margin "${TTAC_UPDATE_VALUE_MARGIN}" \
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
    > "${LOG_DIR}/eval_${mode}.log" 2>&1
  end_s=$(date +%s)
  duration=$((end_s - start_s))
  echo "${mode},${tag},${duration}" >> "${REPORT_DIR}/durations.csv"
  log "MODE_DONE mode=${mode} seconds=${duration}"
done

RUN_DIR="${RUN_DIR}" REPORT_DIR="${REPORT_DIR}" TS="${TS}" MODES="${MODES}" "${PYTHON}" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import statistics

run_dir = Path(os.environ["RUN_DIR"])
report_dir = Path(os.environ["REPORT_DIR"])
ts = os.environ["TS"]
modes = os.environ["MODES"].split()

durations = {}
duration_path = report_dir / "durations.csv"
if duration_path.exists():
    for line in duration_path.read_text().splitlines():
        mode, tag, seconds = line.split(",")
        durations[mode] = int(seconds)

rows = []
for mode in modes:
    tag = f"v5_6_light_pair20_{mode}_{ts}"
    csv_path = run_dir / f"reward_summary_cross_{tag}.csv"
    if not csv_path.exists():
        rows.append({"mode": mode, "status": "missing", "csv": str(csv_path)})
        continue
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    pair_values = defaultdict(list)
    for row in data:
        pair_values[row["policy_labels"]].append(float(row["total_reward"]))

    def kind(label: str) -> str:
        m = re.search(r"cross-(\d+)_(\d+)$", label)
        if not m:
            return "unknown"
        return "sp" if m.group(1) == m.group(2) else "xp"

    per_pair = {k: statistics.fmean(v) for k, v in pair_values.items()}
    sp = [v for k, v in per_pair.items() if kind(k) == "sp"]
    xp = [v for k, v in per_pair.items() if kind(k) == "xp"]
    all_vals = list(per_pair.values())
    rows.append({
        "mode": mode,
        "status": "ok",
        "pairs": len(per_pair),
        "episodes": len(data),
        "sp": statistics.fmean(sp) if sp else float("nan"),
        "xp": statistics.fmean(xp) if xp else float("nan"),
        "all": statistics.fmean(all_vals) if all_vals else float("nan"),
        "seconds": durations.get(mode, -1),
        "csv": str(csv_path),
    })

summary_csv = report_dir / "summary.csv"
with summary_csv.open("w", newline="") as f:
    fieldnames = ["mode", "status", "pairs", "episodes", "sp", "xp", "all", "seconds", "csv"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

summary_md = report_dir / "summary.md"
with summary_md.open("w") as f:
    f.write("# TTAC v5.6 Light Online Pair20\n\n")
    f.write("| mode | XP | SP | ALL | pairs | episodes | seconds |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        if row["status"] != "ok":
            f.write(f"| {row['mode']} | missing | missing | missing | 0 | 0 | -1 |\n")
            continue
        f.write(
            f"| {row['mode']} | {row['xp']:.3f} | {row['sp']:.3f} | "
            f"{row['all']:.3f} | {row['pairs']} | {row['episodes']} | {row['seconds']} |\n"
        )
    f.write("\n")
    f.write("Note: `ttac_v5_6_latest_light` preserves the v5.2 latest estimator target, "
            "but computes CE/KL only on the selected latest query/current ego obs instead "
            "of full history buffers.\n")

print(summary_md)
PY

log "SUMMARY_READY ${REPORT_DIR}/summary.md"
