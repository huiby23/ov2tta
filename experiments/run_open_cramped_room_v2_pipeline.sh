#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
source "${ROOT}/experiments/repro_env.sh"

LAYOUT="${LAYOUT:-open_cramped_room_v2}"
MODE="${MODE:-smoke}"  # validate | smoke | full | all
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-reports/open_cramped_room_v2_${TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/open_cramped_room_v2_${TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_ROOT}"

SEED="${SEED:-42}"
SMOKE_NUM_SEEDS="${SMOKE_NUM_SEEDS:-1}"
SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-500000}"
SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON:-250000}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-16}"
SMOKE_NUM_MINIBATCHES="${SMOKE_NUM_MINIBATCHES:-4}"
SMOKE_NUM_ITERATIONS="${SMOKE_NUM_ITERATIONS:-1}"
SMOKE_EVAL_SEEDS="${SMOKE_EVAL_SEEDS:-8}"

NUM_SEEDS="${NUM_SEEDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
STATE_AUG_ITERATIONS="${STATE_AUG_ITERATIONS:-10}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
DIAG_EVAL_SEEDS="${DIAG_EVAL_SEEDS:-500}"
DIAG_EPISODES="${DIAG_EPISODES:-20}"
DIAG_MAX_PAIRS="${DIAG_MAX_PAIRS:-30}"
DIAG_COMPAT_LIMIT="${DIAG_COMPAT_LIMIT:-256}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_MODE="${WANDB_MODE:-online}"
SMOKE_WANDB_MODE="${SMOKE_WANDB_MODE:-disabled}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

print_repro_env

echo "[open-cramped] layout=${LAYOUT} mode=${MODE} tag=${TAG} report_dir=${REPORT_DIR}"

run_validate() {
  echo "[open-cramped] validate_start $(date -Is)"
  "${PYTHON}" -m py_compile \
    JaxMARL/jaxmarl/environments/overcooked_v2/layouts.py \
    experiments/overcooked_v2_experiments/tools/validate_layout.py \
    experiments/overcooked_v2_experiments/tools/summarize_layout_diagnostics.py
  "${PYTHON}" experiments/overcooked_v2_experiments/tools/validate_layout.py \
    --layout "${LAYOUT}" \
    --output-dir "${REPORT_DIR}/layout_validation" \
    --required-objects 0 1 R P B X
  echo "[open-cramped] validate_done $(date -Is)"
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d -name "*_${LAYOUT}_avs-full" 2>/dev/null | sort | tail -n 1
}

run_eval() {
  local run_dir="$1"
  local seeds="$2"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[open-cramped] missing run_dir for eval: ${run_dir}" >&2
    return 1
  fi
  echo "[open-cramped] eval_start $(date -Is) run_dir=${run_dir} seeds=${seeds}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --seed "${SEED}" \
    --num_seeds "${seeds}" \
    --all \
    --no_viz
  echo "[open-cramped] eval_done $(date -Is) run_dir=${run_dir}"
}

run_ppo_train() {
  local label="$1"
  local state_aug="$2"
  local prefix="$3"
  local num_seeds="$4"
  local total_timesteps="$5"
  local rew_horizon="$6"
  local num_envs="$7"
  local minibatches="$8"
  local iterations="$9"
  local wandb_mode="${10}"
  local log_file="${LOG_ROOT}/${label}.log"
  echo "[open-cramped] train_start $(date -Is) label=${label} state_aug=${state_aug} prefix=${prefix} log=${log_file}"
  local args=(
    -m overcooked_v2_experiments.ppo.main
    +experiment=cnn
    +env=open_cramped_room_v2
    SEED="${SEED}"
    NUM_SEEDS="${num_seeds}"
    NUM_CHECKPOINTS=3
    VISUALIZE=False
    +OPTIONAL_PREFIX="${prefix}"
    wandb.ENTITY="${WANDB_ENTITY}"
    wandb.PROJECT="${WANDB_PROJECT}"
    wandb.WANDB_MODE="${wandb_mode}"
    model.TOTAL_TIMESTEPS="${total_timesteps}"
    model.REW_SHAPING_HORIZON="${rew_horizon}"
    model.NUM_ENVS="${num_envs}"
    model.NUM_STEPS="${NUM_STEPS}"
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}"
    model.NUM_MINIBATCHES="${minibatches}"
  )
  if [[ "${state_aug}" == "1" ]]; then
    args+=(+NUM_ITERATIONS="${iterations}")
  fi
  "${PYTHON}" "${args[@]}" 2>&1 | tee "${log_file}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "${run_dir}" > "${REPORT_DIR}/${label}_run_dir.txt"
  echo "[open-cramped] train_done $(date -Is) label=${label} run_dir=${run_dir}"
}

run_diag() {
  local label="$1"
  local method_name="$2"
  local backend="$3"
  local run_dir="$4"
  local out_dir="${REPORT_DIR}/diagnostics/${label}"
  mkdir -p "${out_dir}"
  echo "[open-cramped] diagnostics_start $(date -Is) label=${label} run_dir=${run_dir}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py \
    --single_run_dir "${run_dir}" \
    --single_name "${method_name}" \
    --single_backend "${backend}" \
    --layout "${LAYOUT}" \
    --eval_seed "${SEED}" \
    --num_eval_seeds "${DIAG_EVAL_SEEDS}" \
    --num_diag_episodes "${DIAG_EPISODES}" \
    --max_pairs "${DIAG_MAX_PAIRS}" \
    --compatibility_sample_limit "${DIAG_COMPAT_LIMIT}" \
    --output_dir "${out_dir}"
  echo "[open-cramped] diagnostics_done $(date -Is) label=${label} out=${out_dir}"
}

run_smoke() {
  run_ppo_train "smoke_ppo_standard" 0 "open_cramped_room_v2_smoke_ppo_standard_${TAG}" "${SMOKE_NUM_SEEDS}" "${SMOKE_TOTAL_TIMESTEPS}" "${SMOKE_REW_SHAPING_HORIZON}" "${SMOKE_NUM_ENVS}" "${SMOKE_NUM_MINIBATCHES}" "${SMOKE_NUM_ITERATIONS}" "${SMOKE_WANDB_MODE}"
  run_eval "$(cat "${REPORT_DIR}/smoke_ppo_standard_run_dir.txt")" "${SMOKE_EVAL_SEEDS}"
  run_ppo_train "smoke_ppo_state_aug" 1 "open_cramped_room_v2_smoke_ppo_state_aug_${TAG}" "${SMOKE_NUM_SEEDS}" "${SMOKE_TOTAL_TIMESTEPS}" "${SMOKE_REW_SHAPING_HORIZON}" "${SMOKE_NUM_ENVS}" "${SMOKE_NUM_MINIBATCHES}" "${SMOKE_NUM_ITERATIONS}" "${SMOKE_WANDB_MODE}"
  run_eval "$(cat "${REPORT_DIR}/smoke_ppo_state_aug_run_dir.txt")" "${SMOKE_EVAL_SEEDS}"
}

run_full_prestudy() {
  local ppo_std_prefix="open_cramped_room_v2_ppo_cnn_standard_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}"
  local ppo_sa_prefix="open_cramped_room_v2_ppo_cnn_state_aug_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}"
  run_ppo_train "ppo_cnn_standard" 0 "${ppo_std_prefix}" "${NUM_SEEDS}" "${TOTAL_TIMESTEPS}" "${REW_SHAPING_HORIZON}" "${NUM_ENVS}" "${NUM_MINIBATCHES}" "${STATE_AUG_ITERATIONS}" "${WANDB_MODE}"
  run_eval "$(cat "${REPORT_DIR}/ppo_cnn_standard_run_dir.txt")" "${EVAL_SEEDS}"
  run_diag "ppo_cnn_standard" "PPO CNN standard" "ppo" "$(cat "${REPORT_DIR}/ppo_cnn_standard_run_dir.txt")"

  run_ppo_train "ppo_cnn_state_aug" 1 "${ppo_sa_prefix}" "${NUM_SEEDS}" "${TOTAL_TIMESTEPS}" "${REW_SHAPING_HORIZON}" "${NUM_ENVS}" "${NUM_MINIBATCHES}" "${STATE_AUG_ITERATIONS}" "${WANDB_MODE}"
  run_eval "$(cat "${REPORT_DIR}/ppo_cnn_state_aug_run_dir.txt")" "${EVAL_SEEDS}"
  run_diag "ppo_cnn_state_aug" "PPO CNN state-aug" "ppo" "$(cat "${REPORT_DIR}/ppo_cnn_state_aug_run_dir.txt")"

  PREFIX="open_cramped_room_v2_mep_ent0.1_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}" \
  LAYOUT="${LAYOUT}" SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  NUM_ENVS="${NUM_ENVS}" NUM_STEPS="${NUM_STEPS}" UPDATE_EPOCHS="${UPDATE_EPOCHS}" NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
  MEP_ENT_COEF=0.1 WANDB_MODE="${WANDB_MODE}" WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" \
  PYTHON="${PYTHON}" experiments/run_mep_64_16.sh 2>&1 | tee "${LOG_ROOT}/mep_ent0.1_standard.log"
  local mep_dir
  mep_dir="$(latest_run_dir "open_cramped_room_v2_mep_ent0.1_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}")"
  echo "${mep_dir}" > "${REPORT_DIR}/mep_ent0.1_standard_run_dir.txt"
  run_eval "${mep_dir}" "${EVAL_SEEDS}"
  run_diag "mep_ent0.1_standard" "MEP ent=0.1 standard" "ppo" "${mep_dir}"

  PREFIX="open_cramped_room_v2_trajedi_div0.1_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}" \
  LAYOUT="${LAYOUT}" SEED="${SEED}" NUM_SEEDS="${NUM_SEEDS}" TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
  NUM_ENVS="${NUM_ENVS}" NUM_STEPS="${NUM_STEPS}" UPDATE_EPOCHS="${UPDATE_EPOCHS}" NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
  TRAJEDI_DIV_WEIGHT=0.1 WANDB_MODE="${WANDB_MODE}" WANDB_ENTITY="${WANDB_ENTITY}" WANDB_PROJECT="${WANDB_PROJECT}" \
  PYTHON="${PYTHON}" experiments/run_trajedi_64_16.sh 2>&1 | tee "${LOG_ROOT}/trajedi_div0.1_standard.log"
  local trajedi_dir
  trajedi_dir="$(latest_run_dir "open_cramped_room_v2_trajedi_div0.1_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_seed${SEED}_${NUM_SEEDS}seeds_${TAG}")"
  echo "${trajedi_dir}" > "${REPORT_DIR}/trajedi_div0.1_standard_run_dir.txt"
  run_eval "${trajedi_dir}" "${EVAL_SEEDS}"
  run_diag "trajedi_div0.1_standard" "TrajeDi div=0.1 standard" "ppo" "${trajedi_dir}"

  "${PYTHON}" experiments/overcooked_v2_experiments/tools/summarize_layout_diagnostics.py \
    --diagnostics-root "${REPORT_DIR}/diagnostics" \
    --output-dir "${REPORT_DIR}/summary"
}

case "${MODE}" in
  validate)
    run_validate
    ;;
  smoke)
    run_validate
    run_smoke
    ;;
  full)
    run_validate
    run_full_prestudy
    ;;
  all)
    run_validate
    run_smoke
    run_full_prestudy
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected validate|smoke|full|all" >&2
    exit 2
    ;;
esac

echo "[open-cramped] done $(date -Is) report_dir=${REPORT_DIR} log_root=${LOG_ROOT}"
