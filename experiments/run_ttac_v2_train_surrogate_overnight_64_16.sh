#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
cd "$ROOT"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/ttac_v2_train_surrogate_overnight_${TS}}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v2_train_surrogate_overnight_${TS}}"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
QUEUE="$LOG_DIR/queue.log"
: > "$QUEUE"

POPULATION_RUN_DIR="${POPULATION_RUN_DIR:-runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full}"
MAX_POLICIES="${MAX_POLICIES:-10}"
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

QUICK_EVAL_SEEDS="${QUICK_EVAL_SEEDS:-50}"
QUICK_MAX_PAIRINGS="${QUICK_MAX_PAIRINGS:-10}"
QUICK_EVAL_BATCHES="${QUICK_EVAL_BATCHES:-5}"
QUICK_PAIRINGS_PER_CHUNK="${QUICK_PAIRINGS_PER_CHUNK:-5}"

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_AGREEMENT_COEF="${TTAC_AGREEMENT_COEF:-0.05}"
TTAC_TRAIN_KL_COEF="${TTAC_TRAIN_KL_COEF:-0.005}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE"; }

wait_for_current_sweep() {
  log "Waiting for current TTACv2 surrogate eval sweep to finish, if any."
  while true; do
    local active
    active="$(ps -eo pid,ppid,cmd | grep -E 'run_ttac_v2_frozen_surrogate_sweep_parallel\.sh|visualize_ppo.*surrogate_' | grep -v -E 'grep|awk|run_ttac_v2_train_surrogate_overnight' || true)"
    if [ -z "$active" ]; then
      break
    fi
    echo "$active" | tee -a "$QUEUE"
    sleep 120
  done
  log "No active surrogate sweep detected."
}

train_one() {
  local tag="$1" surrogate="$2" beta="$3" min_prob="$4" max_entropy="$5" mode="$6" num_seeds="$7" total_steps="$8" shaping="$9" iterations="${10}" wandb_mode="${11}"
  local prefix="ttac_v2_${tag}_state_aug_64_16_10M_seed42_10seeds_${TS}"
  local log_file="$LOG_DIR/train_${tag}_${mode}.log"
  log "TRAIN_START tag=${tag} mode=${mode} surrogate=${surrogate} beta=${beta} min_prob=${min_prob} max_entropy=${max_entropy}"
  "$PYTHON" -m overcooked_v2_experiments.ttac_v2.main \
    +experiment=cnn +env=original env.ENV_KWARGS.layout="$LAYOUT" \
    SEED="$SEED" NUM_SEEDS="$num_seeds" NUM_CHECKPOINTS=3 +NUM_ITERATIONS="$iterations" VISUALIZE=False \
    +OPTIONAL_PREFIX="$prefix" \
    wandb.ENTITY="$WANDB_ENTITY" wandb.PROJECT="$WANDB_PROJECT" wandb.WANDB_MODE="$wandb_mode" \
    +TTAC_V2_FROZEN_POPULATION.RUN_DIR="$POPULATION_RUN_DIR" \
    +TTAC_V2_FROZEN_POPULATION.MAX_POLICIES="$MAX_POLICIES" \
    +TTAC_V2_FROZEN_POPULATION.STOCHASTIC=true \
    model.TOTAL_TIMESTEPS="$total_steps" model.REW_SHAPING_HORIZON="$shaping" \
    model.NUM_ENVS="$NUM_ENVS" model.NUM_STEPS="$NUM_STEPS" model.UPDATE_EPOCHS="$UPDATE_EPOCHS" model.NUM_MINIBATCHES="$NUM_MINIBATCHES" \
    model.TTAC_ADAPTER_SCALE="$TTAC_ADAPTER_SCALE" model.TTAC_AGREEMENT_COEF="$TTAC_AGREEMENT_COEF" model.TTAC_TRAIN_KL_COEF="$TTAC_TRAIN_KL_COEF" \
    model.TTAC_TRAIN_SURROGATE="$surrogate" model.TTAC_TRAIN_PROJECT_BETA="$beta" model.TTAC_TRAIN_SUPPORT_MIN_PROB="$min_prob" model.TTAC_TRAIN_SUPPORT_MAX_ENTROPY="$max_entropy" \
    model.TTAC_TEST_HIST_KL_COEF="$TTAC_TEST_HIST_KL_COEF" model.TTAC_TEST_EGO_KL_COEF="$TTAC_TEST_EGO_KL_COEF" model.TTAC_TEST_CUR_KL_COEF="$TTAC_TEST_CUR_KL_COEF" \
    model.TTAC_TEST_UPDATE_STEPS="$TTAC_TEST_UPDATE_STEPS" model.TTAC_TEST_LR="$TTAC_TEST_LR" model.TTAC_HISTORY_LEN="$TTAC_HISTORY_LEN" \
    > "$log_file" 2>&1

  local run_dir
  run_dir="$(find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$run_dir" ]; then
    log "ERROR no run dir found for prefix ${prefix}"
    return 1
  fi
  echo "$run_dir" > "$REPORT_DIR/run_dir_${tag}_${mode}.txt"
  log "TRAIN_DONE tag=${tag} mode=${mode} run_dir=${run_dir}"
}

quick_eval_one() {
  local tag="$1" run_dir="$2" eval_mode="$3" out_tag="$4" beta="$5" min_prob="$6" max_entropy="$7"
  local log_file="$LOG_DIR/eval_${tag}_${out_tag}.log"
  log "QUICK_EVAL_START tag=${tag} mode=${eval_mode} out=${out_tag}"
  "$PYTHON" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo \
    --d "$run_dir" --cross --num_seeds "$QUICK_EVAL_SEEDS" --no_viz --seed "$SEED" \
    --ttac_mode "$eval_mode" --output_tag "$out_tag" --eval_batches "$QUICK_EVAL_BATCHES" --max_pairings "$QUICK_MAX_PAIRINGS" \
    --ttac_adapter_scale "$TTAC_ADAPTER_SCALE" --ttac_test_hist_kl_coef "$TTAC_TEST_HIST_KL_COEF" --ttac_test_ego_kl_coef "$TTAC_TEST_EGO_KL_COEF" --ttac_test_cur_kl_coef "$TTAC_TEST_CUR_KL_COEF" \
    --ttac_test_lr "$TTAC_TEST_LR" --ttac_test_update_steps "$TTAC_TEST_UPDATE_STEPS" --ttac_history_len "$TTAC_HISTORY_LEN" \
    --ttac_test_project_beta "$beta" --ttac_test_support_min_prob "$min_prob" --ttac_test_support_max_entropy "$max_entropy" \
    --online_eval_pairings_per_chunk "$QUICK_PAIRINGS_PER_CHUNK" > "$log_file" 2>&1
  log "QUICK_EVAL_DONE tag=${tag} out=${out_tag}"
}

summarize_eval() {
  "$PYTHON" - <<'PY'
from pathlib import Path
import os
import pandas as pd
report = Path(os.environ['REPORT_DIR'])
rows = []
for f in report.glob('run_dir_*_full.txt'):
    tag = f.name[len('run_dir_'):-len('_full.txt')]
    run_dir = Path(f.read_text().strip())
    for csv in sorted(run_dir.glob('reward_summary_cross_quick_*.csv')):
        name = csv.stem.replace('reward_summary_cross_quick_', '')
        df = pd.read_csv(csv)
        rows.append({
            'tag': tag,
            'eval': name,
            'mean_xp': float(df.total_reward.mean()),
            'std': float(df.total_reward.std()),
            'min': float(df.total_reward.min()),
            'max': float(df.total_reward.max()),
            'n': int(len(df)),
            'run_dir': str(run_dir),
            'csv': str(csv),
        })
out = pd.DataFrame(rows)
if len(out):
    out = out.sort_values(['tag', 'mean_xp'], ascending=[True, False])
    out.to_csv(report / 'quick_eval_summary.csv', index=False)
    try:
        table = out.to_markdown(index=False)
    except Exception:
        table = out.to_csv(index=False)
    (report / 'quick_eval_summary.md').write_text('# TTACv2 surrogate-train overnight quick XP\n\n' + table + '\n')
    print(out.to_string(index=False))
else:
    print('No quick eval CSV found yet.')
PY
}

run_variant() {
  local tag="$1" surrogate="$2" beta="$3" min_prob="$4" max_entropy="$5" primary_eval_mode="$6"
  train_one "$tag" "$surrogate" "$beta" "$min_prob" "$max_entropy" smoke "$SMOKE_NUM_SEEDS" "$SMOKE_TOTAL_TIMESTEPS" "$SMOKE_REW_SHAPING_HORIZON" "$SMOKE_NUM_ITERATIONS" disabled
  train_one "$tag" "$surrogate" "$beta" "$min_prob" "$max_entropy" full "$FULL_NUM_SEEDS" "$FULL_TOTAL_TIMESTEPS" "$FULL_REW_SHAPING_HORIZON" "$FULL_NUM_ITERATIONS" online
  local run_dir
  run_dir="$(cat "$REPORT_DIR/run_dir_${tag}_full.txt")"
  quick_eval_one "$tag" "$run_dir" base_no_test_adapt "quick_base_no_test_adapt" "$beta" "$min_prob" "$max_entropy"
  quick_eval_one "$tag" "$run_dir" "$primary_eval_mode" "quick_${primary_eval_mode}" "$beta" "$min_prob" "$max_entropy"
  summarize_eval | tee -a "$QUEUE"
}

export REPORT_DIR
log "OVERNIGHT_START report=${REPORT_DIR} log=${LOG_DIR}"
log "Plan: wait sweep -> smoke projected/projconf -> full train projected/projconf -> quick XP eval. Optional confident runs only if RUN_CONFIDENT=1."
wait_for_current_sweep

run_variant projected_train projected 2.0 0.05 1.5 ttac_projected_history
run_variant projconf_train projected_confident 2.0 0.05 1.5 ttac_projected_confident

if [ "${RUN_CONFIDENT:-0}" = "1" ]; then
  run_variant confident_train confident 2.0 0.05 1.5 ttac_confident_history
fi

summarize_eval | tee -a "$QUEUE"
log "OVERNIGHT_DONE"
