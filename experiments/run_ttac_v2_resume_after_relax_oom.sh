#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/logs/ttac_v2_resume_${TS}"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export MODE="${MODE:-full}"
export WANDB_MODE="${WANDB_MODE:-online}"
export EVAL_BATCHES="${EVAL_BATCHES:-10}"
export EVAL_SEEDS="${EVAL_SEEDS:-500}"
export NUM_ENVS="${NUM_ENVS:-64}"
export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
export NUM_SEEDS="${NUM_SEEDS:-10}"
export SEED="${SEED:-42}"
export TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
export TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
export TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
export TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
export TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
export TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
export TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
export TTAC_ONLINE_EVAL_PAIRINGS_PER_CHUNK="${TTAC_ONLINE_EVAL_PAIRINGS_PER_CHUNK:-9}"
RELAX_RUN_DIR="${RELAX_RUN_DIR:-runs/ttac_v2_relax_selfplay_state_aug_64_16_10M_seed42_10seeds_20260531_012443/20260531-012457_tmtpxwwa_counter_circuit_avs-full}"
log(){ echo "[$(date +%F %T)] $*" | tee -a "${LOG_DIR}/queue.log"; }
run_stage(){ local name="$1"; shift; log "START ${name}"; "$@" > "${LOG_DIR}/${name}.log" 2>&1; log "DONE ${name}"; }
eval_one(){
  local mode="$1"
  local sp="${RELAX_RUN_DIR}/reward_summary_sp_${mode}.csv"
  local cross="${RELAX_RUN_DIR}/reward_summary_cross_${mode}.csv"
  if [[ -f "${sp}" && -f "${cross}" ]]; then
    log "SKIP relax eval ${mode}: both SP and cross exist"
    return
  fi
  if [[ ! -f "${sp}" ]]; then
    log "START relax eval ${mode} SP"
    "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo --d "${RELAX_RUN_DIR}" --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" --ttac_mode "${mode}" --output_tag "${mode}" --eval_batches "${EVAL_BATCHES}" \
      --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" --ttac_test_lr "${TTAC_TEST_LR}" --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" --ttac_history_len "${TTAC_HISTORY_LEN}" --online_eval_pairings_per_chunk "${TTAC_ONLINE_EVAL_PAIRINGS_PER_CHUNK:-9}" > "${LOG_DIR}/relax_eval_${mode}_sp.log" 2>&1
    log "DONE relax eval ${mode} SP"
  fi
  if [[ ! -f "${cross}" ]]; then
    log "START relax eval ${mode} cross"
    "${PYTHON}" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo --d "${RELAX_RUN_DIR}" --cross --num_seeds "${EVAL_SEEDS}" --no_viz --seed "${SEED}" --ttac_mode "${mode}" --output_tag "${mode}" --eval_batches "${EVAL_BATCHES}" \
      --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" --ttac_test_lr "${TTAC_TEST_LR}" --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" --ttac_history_len "${TTAC_HISTORY_LEN}" --online_eval_pairings_per_chunk "${TTAC_ONLINE_EVAL_PAIRINGS_PER_CHUNK:-9}" > "${LOG_DIR}/relax_eval_${mode}_cross.log" 2>&1
    log "DONE relax eval ${mode} cross"
  fi
}
log "RESUME after TTACv2 relax eval OOM; run_dir=${RELAX_RUN_DIR}"
for mode in base_no_test_adapt ttac_true_history ttac_wrong_history ttac_adapter_off; do eval_one "${mode}"; done
run_stage ttac_v2_frozen_population experiments/run_ttac_v2_frozen_population_64_16.sh
run_stage ttac_v2_sync_population experiments/run_ttac_v2_sync_population_64_16.sh
run_stage gamma_ttac experiments/run_gamma_ttac_mappo_state_aug_full.sh
log "ALL_DONE"
