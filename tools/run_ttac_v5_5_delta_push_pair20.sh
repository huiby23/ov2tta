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
EXP_DIR="${EXP_DIR:-experiments/overcooked_v2_experiments/ttac_v5_5_delta_push}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v5_5_delta_push_pair20_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v5_5_delta_push_pair20_${TS}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-v5_5_delta_push_pair20_${TS}}"

SEED="${SEED:-42}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
PAIRING_START="${PAIRING_START:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
HISTORY_LEN="${HISTORY_LEN:-50}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_V5_AGREEMENT_COEF="${TTAC_V5_AGREEMENT_COEF:-20.0}"
TTAC_V5_SUPPORT_COEF="${TTAC_V5_SUPPORT_COEF:-1.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_V5_2_TV_THRESHOLD="${TTAC_V5_2_TV_THRESHOLD:-0.03}"
TTAC_V5_4_DELTA_MARGIN="${TTAC_V5_4_DELTA_MARGIN:-0.0}"
TTAC_V5_4_DELTA_SCALE="${TTAC_V5_4_DELTA_SCALE:-2.0}"
TTAC_V5_4_DELTA_MASS_THRESHOLD="${TTAC_V5_4_DELTA_MASS_THRESHOLD:-0.03}"

if [[ -n "${MODES:-}" ]]; then
  read -r -a MODE_LIST <<< "${MODES}"
else
  MODE_LIST=(
    base_no_test_adapt
    ttac_v5_2_latest
    ttac_v5_4_delta
    ttac_v5_4_delta_push
    ttac_v5_4_delta_push_wrong_history
    ttac_v5_4_delta_push_random_history
    ttac_v5_4_delta_push_delayed_history
    ttac_v5_5_top_delta
    ttac_v5_5_top_delta_wrong_history
    ttac_v5_5_top_delta_random_history
    ttac_v5_5_top_delta_delayed_history
  )
fi

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

cat > "${REPORT_DIR}/config.env" <<EOF
RUN_DIR=${RUN_DIR}
ESTIMATOR=${ESTIMATOR}
EXP_DIR=${EXP_DIR}
OUTPUT_PREFIX=${OUTPUT_PREFIX}
SEED=${SEED}
EVAL_NUM_SEEDS=${EVAL_NUM_SEEDS}
MAX_PAIRINGS=${MAX_PAIRINGS}
PAIRING_START=${PAIRING_START}
EVAL_BATCHES=${EVAL_BATCHES}
HISTORY_LEN=${HISTORY_LEN}
TTAC_TEST_LR=${TTAC_TEST_LR}
TTAC_TEST_UPDATE_STEPS=${TTAC_TEST_UPDATE_STEPS}
TTAC_V5_AGREEMENT_COEF=${TTAC_V5_AGREEMENT_COEF}
TTAC_V5_SUPPORT_COEF=${TTAC_V5_SUPPORT_COEF}
TTAC_TEST_EGO_KL_COEF=${TTAC_TEST_EGO_KL_COEF}
TTAC_TEST_CUR_KL_COEF=${TTAC_TEST_CUR_KL_COEF}
TTAC_TEST_HIST_KL_COEF=${TTAC_TEST_HIST_KL_COEF}
TTAC_V5_2_TV_THRESHOLD=${TTAC_V5_2_TV_THRESHOLD}
TTAC_V5_4_DELTA_MARGIN=${TTAC_V5_4_DELTA_MARGIN}
TTAC_V5_4_DELTA_SCALE=${TTAC_V5_4_DELTA_SCALE}
TTAC_V5_4_DELTA_MASS_THRESHOLD=${TTAC_V5_4_DELTA_MASS_THRESHOLD}
MODES=${MODE_LIST[*]}
EOF

for mode in "${MODE_LIST[@]}"; do
  tag="${OUTPUT_PREFIX}_${mode}"
  log="${LOG_DIR}/${tag}.log"
  echo "[$(date '+%F %T')] MODE_START ${mode}"
  "${PYTHON}" "${EXP_DIR}/utils/visualize_ppo.py" \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${EVAL_NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --pairing_start "${PAIRING_START}" \
    --max_pairings "${MAX_PAIRINGS}" \
    --ttac_v5_estimator_path "${ESTIMATOR}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${HISTORY_LEN}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_v5_agreement_coef "${TTAC_V5_AGREEMENT_COEF}" \
    --ttac_v5_support_coef "${TTAC_V5_SUPPORT_COEF}" \
    --ttac_v5_2_tv_threshold "${TTAC_V5_2_TV_THRESHOLD}" \
    --ttac_v5_4_delta_margin "${TTAC_V5_4_DELTA_MARGIN}" \
    --ttac_v5_4_delta_scale "${TTAC_V5_4_DELTA_SCALE}" \
    --ttac_v5_4_delta_mass_threshold "${TTAC_V5_4_DELTA_MASS_THRESHOLD}" \
    > "${log}" 2>&1
  echo "[$(date '+%F %T')] MODE_DONE ${mode}"
done

export RUN_DIR REPORT_DIR OUTPUT_PREFIX
export MODES="${MODE_LIST[*]}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import csv
import os
import re

run_dir = Path(os.environ["RUN_DIR"])
report_dir = Path(os.environ["REPORT_DIR"])
prefix = os.environ["OUTPUT_PREFIX"]
modes = os.environ.get("MODES", "").split()
if not modes:
    modes = [
        "base_no_test_adapt",
        "ttac_v5_2_latest",
        "ttac_v5_4_delta",
        "ttac_v5_4_delta_push",
        "ttac_v5_4_delta_push_wrong_history",
        "ttac_v5_4_delta_push_random_history",
        "ttac_v5_4_delta_push_delayed_history",
        "ttac_v5_5_top_delta",
        "ttac_v5_5_top_delta_wrong_history",
        "ttac_v5_5_top_delta_random_history",
        "ttac_v5_5_top_delta_delayed_history",
    ]

pat = re.compile(r"cross-(\d+)_(\d+)")
rows = []
for mode in modes:
    path = run_dir / f"reward_summary_cross_{prefix}_{mode}.csv"
    rewards = []
    sp = []
    xp = []
    if not path.exists():
        rows.append({"mode": mode, "rows": 0, "pairs": 0, "SP": "", "XP": "", "ALL": "", "path": str(path)})
        continue
    labels = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            rew = float(r["total_reward"])
            rewards.append(rew)
            lab = r["policy_labels"]
            labels.add(lab)
            m = pat.match(lab)
            if m and m.group(1) == m.group(2):
                sp.append(rew)
            else:
                xp.append(rew)
    rows.append({
        "mode": mode,
        "rows": len(rewards),
        "pairs": len(labels),
        "SP": f"{sum(sp)/len(sp):.3f}" if sp else "",
        "XP": f"{sum(xp)/len(xp):.3f}" if xp else "",
        "ALL": f"{sum(rewards)/len(rewards):.3f}" if rewards else "",
        "path": str(path),
    })

csv_path = report_dir / "delta_push_pair20_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mode", "rows", "pairs", "SP", "XP", "ALL", "path"])
    writer.writeheader()
    writer.writerows(rows)

md_path = report_dir / "delta_push_pair20_summary.md"
with md_path.open("w") as f:
    f.write("# TTAC v5.5 Delta Push Pair20 Summary\n\n")
    f.write("| mode | rows | pairs | SP | XP | ALL |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['mode']} | {r['rows']} | {r['pairs']} | {r['SP']} | {r['XP']} | {r['ALL']} |\n")
    f.write("\n")
    f.write(f"CSV: `{csv_path}`\n")

print(md_path)
PY
