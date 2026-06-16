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

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"

NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
FULL_NUM_SEEDS="${FULL_NUM_SEEDS:-10}"
FULL_TOTAL_TIMESTEPS="${FULL_TOTAL_TIMESTEPS:-10000000}"
FULL_REW_SHAPING_HORIZON="${FULL_REW_SHAPING_HORIZON:-5000000}"
FULL_NUM_ITERATIONS="${FULL_NUM_ITERATIONS:-10}"

SMOKE_NUM_SEEDS="${SMOKE_NUM_SEEDS:-1}"
SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-262144}"
SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON:-131072}"
SMOKE_NUM_ITERATIONS="${SMOKE_NUM_ITERATIONS:-1}"

EVAL_SEEDS="${EVAL_SEEDS:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
ONLINE_EVAL_PAIRINGS_PER_CHUNK="${ONLINE_EVAL_PAIRINGS_PER_CHUNK:-8}"

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF:-0.05}"
TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF:-0.005}"
TTAC_TRAIN_PROJECT_BETA="${TTAC_TRAIN_PROJECT_BETA:-2.0}"
TTAC_TRAIN_SUPPORT_MIN_PROB="${TTAC_TRAIN_SUPPORT_MIN_PROB:-0.05}"
TTAC_TRAIN_SUPPORT_MAX_ENTROPY="${TTAC_TRAIN_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TRAIN_ADVANTAGE_POWER="${TTAC_TRAIN_ADVANTAGE_POWER:-1.0}"

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

LOG_DIR="${LOG_DIR:-logs/ttac_v2_aw_full_pipeline_${TS}}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v2_aw_full_pipeline_${TS}}"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

run_train() {
  local phase="$1" num_seeds="$2" total_steps="$3" shaping="$4" iterations="$5" wandb_mode="$6"
  local prefix="ttac_v2_aw_${phase}_state_aug_64_16_10M_seed42_10seeds_${TS}"
  local log_file="${LOG_DIR}/train_${phase}.log"
  log "TRAIN_START phase=${phase} prefix=${prefix} seeds=${num_seeds} steps=${total_steps} iterations=${iterations}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.main \
    +experiment=cnn +env=original env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" NUM_SEEDS="${num_seeds}" NUM_CHECKPOINTS=3 +NUM_ITERATIONS="${iterations}" VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE="${wandb_mode}" \
    model.TOTAL_TIMESTEPS="${total_steps}" model.REW_SHAPING_HORIZON="${shaping}" \
    model.NUM_ENVS="${NUM_ENVS}" model.NUM_STEPS="${NUM_STEPS}" model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE}" model.TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF}" model.TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF}" \
    model.TTAC_TRAIN_SURROGATE=advantage_weighted model.TTAC_TRAIN_PROJECT_BETA="${TTAC_TRAIN_PROJECT_BETA}" \
    model.TTAC_TRAIN_SUPPORT_MIN_PROB="${TTAC_TRAIN_SUPPORT_MIN_PROB}" model.TTAC_TRAIN_SUPPORT_MAX_ENTROPY="${TTAC_TRAIN_SUPPORT_MAX_ENTROPY}" \
    model.TTAC_TRAIN_ADVANTAGE_POWER="${TTAC_TRAIN_ADVANTAGE_POWER}" \
    model.TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF}" model.TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF}" model.TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF}" \
    model.TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS}" model.TTAC_TEST_LR="${TTAC_TEST_LR}" model.TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN}" \
    model.TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA}" model.TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB}" \
    model.TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY}" model.TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER}" \
    > "${log_file}" 2>&1

  local run_dir
  run_dir="$(find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "${run_dir}" ]; then
    log "ERROR no run dir found for prefix=${prefix}"
    return 1
  fi
  echo "${run_dir}" > "${REPORT_DIR}/run_dir_${phase}.txt"
  log "TRAIN_DONE phase=${phase} run_dir=${run_dir}"
}

run_eval() {
  local run_dir="$1" eval_mode="$2" split="$3"
  local cross_args=()
  local summary="sp"
  if [ "${split}" = "cross" ]; then
    cross_args=(--cross)
    summary="cross"
  fi
  local out_tag="full_${eval_mode}_${summary}"
  local log_file="${LOG_DIR}/eval_${eval_mode}_${summary}.log"
  log "EVAL_START mode=${eval_mode} split=${split} seeds=${EVAL_SEEDS} batches=${EVAL_BATCHES}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo \
    --d "${run_dir}" "${cross_args[@]}" --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" \
    --ttac_mode "${eval_mode}" --output_tag "${out_tag}" --eval_batches "${EVAL_BATCHES}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --online_eval_pairings_per_chunk "${ONLINE_EVAL_PAIRINGS_PER_CHUNK}" > "${log_file}" 2>&1
  log "EVAL_DONE mode=${eval_mode} split=${split}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" RUN_DIR="${1}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import pandas as pd

report = Path(os.environ["REPORT_DIR"])
run_dir = Path(os.environ["RUN_DIR"])
rows = []
for csv in sorted(run_dir.glob("reward_summary_*.csv")):
    name = csv.stem
    if "full_" not in name:
        continue
    df = pd.read_csv(csv)
    split = "XP" if "cross" in name else "SP"
    mode = name.split("full_", 1)[1].rsplit("_" + ("cross" if split == "XP" else "sp"), 1)[0]
    rows.append({
        "mode": mode,
        "split": split,
        "mean_reward": float(df.total_reward.mean()),
        "std_reward": float(df.total_reward.std()),
        "min_reward": float(df.total_reward.min()),
        "max_reward": float(df.total_reward.max()),
        "n": int(len(df)),
        "csv": str(csv),
    })
out = pd.DataFrame(rows)
if len(out):
    out = out.sort_values(["split", "mode"])
    out.to_csv(report / "full_eval_summary.csv", index=False)
    try:
        table = out.to_markdown(index=False)
    except Exception:
        table = out.to_csv(index=False)
    (report / "full_eval_summary.md").write_text(
        "# TTACv2 advantage-weighted full pipeline eval\n\n"
        f"run_dir: `{run_dir}`\n\n"
        + table
        + "\n"
    )
    print(out.to_string(index=False))
else:
    print("No full eval CSV found.")
PY
}

log "PIPELINE_START report=${REPORT_DIR} log=${LOG_DIR}"
run_train smoke "${SMOKE_NUM_SEEDS}" "${SMOKE_TOTAL_TIMESTEPS}" "${SMOKE_REW_SHAPING_HORIZON}" "${SMOKE_NUM_ITERATIONS}" disabled
run_train full "${FULL_NUM_SEEDS}" "${FULL_TOTAL_TIMESTEPS}" "${FULL_REW_SHAPING_HORIZON}" "${FULL_NUM_ITERATIONS}" online

RUN_DIR="$(cat "${REPORT_DIR}/run_dir_full.txt")"
for mode in base_no_test_adapt ttac_advantage_weighted; do
  run_eval "${RUN_DIR}" "${mode}" sp
  run_eval "${RUN_DIR}" "${mode}" cross
done
summarize "${RUN_DIR}" | tee -a "${QUEUE}"
log "PIPELINE_DONE"
