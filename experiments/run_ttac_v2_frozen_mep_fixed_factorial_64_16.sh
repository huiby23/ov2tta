#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S_frozen_mep_fixed)}"

cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"

POPULATION_RUN_DIR="${POPULATION_RUN_DIR:-runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full}"
MAX_POLICIES="${MAX_POLICIES:-10}"

NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
FULL_NUM_SEEDS="${FULL_NUM_SEEDS:-10}"
FULL_TOTAL_TIMESTEPS="${FULL_TOTAL_TIMESTEPS:-10000000}"
FULL_REW_SHAPING_HORIZON="${FULL_REW_SHAPING_HORIZON:-5000000}"
FULL_NUM_ITERATIONS="${FULL_NUM_ITERATIONS:-10}"

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

LOG_DIR="${LOG_DIR:-logs/ttac_v2_frozen_mep_fixed_${TS}}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v2_frozen_mep_fixed_${TS}}"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"
}

run_train() {
  local tag="$1"
  local surrogate="$2"
  local prefix="ttac_v2_${tag}_state_aug_64_16_10M_seed42_10seeds_${TS}"
  local log_file="${LOG_DIR}/train_${tag}.log"

  log "TRAIN_START tag=${tag} surrogate=${surrogate} prefix=${prefix}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.main \
    +experiment=cnn +env=original env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" NUM_SEEDS="${FULL_NUM_SEEDS}" NUM_CHECKPOINTS=3 +NUM_ITERATIONS="${FULL_NUM_ITERATIONS}" VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${WANDB_ENTITY}" wandb.PROJECT="${WANDB_PROJECT}" wandb.WANDB_MODE=online \
    +TTAC_V2_FROZEN_POPULATION.RUN_DIR="${POPULATION_RUN_DIR}" \
    +TTAC_V2_FROZEN_POPULATION.MAX_POLICIES="${MAX_POLICIES}" \
    +TTAC_V2_FROZEN_POPULATION.STOCHASTIC=true \
    model.TOTAL_TIMESTEPS="${FULL_TOTAL_TIMESTEPS}" model.REW_SHAPING_HORIZON="${FULL_REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" model.NUM_STEPS="${NUM_STEPS}" model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE}" model.TTAC_ZERO_INIT_ADAPTER_UP=false model.TTAC_TRAIN_ADAPTER_READOUT_SCALE=1.0 \
    model.TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF}" model.TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF}" \
    model.TTAC_TRAIN_SURROGATE="${surrogate}" model.TTAC_TRAIN_PROJECT_BETA="${TTAC_TRAIN_PROJECT_BETA}" \
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
    log "ERROR no run dir found for tag=${tag} prefix=${prefix}"
    return 1
  fi
  echo "${run_dir}" > "${REPORT_DIR}/run_dir_${tag}.txt"
  log "TRAIN_DONE tag=${tag} run_dir=${run_dir}"
}

run_eval() {
  local tag="$1"
  local run_dir="$2"
  local eval_mode="$3"
  local split="$4"
  local cross_args=()
  local summary="sp"
  if [ "${split}" = "cross" ]; then
    cross_args=(--cross)
    summary="cross"
  fi
  local out_tag="${tag}_${eval_mode}_${summary}"
  local log_file="${LOG_DIR}/eval_${out_tag}.log"

  log "EVAL_START tag=${tag} mode=${eval_mode} split=${split} seeds=${EVAL_SEEDS} batches=${EVAL_BATCHES}"
  "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo \
    --d "${run_dir}" "${cross_args[@]}" --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" \
    --ttac_mode "${eval_mode}" --output_tag "${out_tag}" --eval_batches "${EVAL_BATCHES}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --online_eval_pairings_per_chunk "${ONLINE_EVAL_PAIRINGS_PER_CHUNK}" > "${log_file}" 2>&1
  log "EVAL_DONE tag=${tag} mode=${eval_mode} split=${split}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

report = Path(os.environ["REPORT_DIR"])
rows = []
for tag_file in sorted(report.glob("run_dir_*.txt")):
    tag = tag_file.stem.replace("run_dir_", "")
    run_dir = Path(tag_file.read_text().strip())
    for csv in sorted(run_dir.glob("reward_summary_*.csv")):
        name = csv.stem
        if tag not in name:
            continue
        df = pd.read_csv(csv)
        split = "XP" if "cross" in name else "SP"
        suffix = "cross" if split == "XP" else "sp"
        mode = name.split(f"{tag}_", 1)[1].rsplit(f"_{suffix}", 1)[0]
        row = {
            "tag": tag,
            "mode": mode,
            "split": split,
            "mean_reward": float(df.total_reward.mean()),
            "std_reward": float(df.total_reward.std()),
            "min_reward": float(df.total_reward.min()),
            "max_reward": float(df.total_reward.max()),
            "n": int(len(df)),
            "run_dir": str(run_dir),
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
    out = out.sort_values(["tag", "split", "mode"])
    out.to_csv(report / "frozen_mep_fixed_eval_summary.csv", index=False)
    try:
        table = out.to_markdown(index=False)
    except Exception:
        table = out.to_csv(index=False)
    (report / "frozen_mep_fixed_eval_summary.md").write_text(
        "# TTACv2 fixed FrozenMEP eval summary\n\n"
        f"report_dir: `{report}`\n\n" + table + "\n"
    )
    print(out.to_string(index=False))
else:
    print("No eval CSV found.")
PY
}

run_experiment() {
  local tag="$1"
  local surrogate="$2"
  shift 2
  local eval_modes=("$@")
  run_train "${tag}" "${surrogate}"
  local run_dir
  run_dir="$(cat "${REPORT_DIR}/run_dir_${tag}.txt")"
  for mode in "${eval_modes[@]}"; do
    run_eval "${tag}" "${run_dir}" "${mode}" sp
    run_eval "${tag}" "${run_dir}" "${mode}" cross
  done
  summarize | tee -a "${QUEUE}"
}

log "PIPELINE_START report=${REPORT_DIR} log=${LOG_DIR}"
log "Using fixed state_sample_run.py with real TTAC_V2_FROZEN_POPULATION support"
run_experiment frozen_mep_aw_fixed advantage_weighted base_no_test_adapt ttac_advantage_weighted ttac_projected_confident
run_experiment frozen_mep_pc_fixed projected_confident base_no_test_adapt ttac_projected_confident ttac_advantage_weighted
log "PIPELINE_DONE"
