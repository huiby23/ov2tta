#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/figure4_128_32_no_state_aug_full_obs}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-${JOB_ROOT}/queue.pid}"
CURRENT_PID_FILE="${CURRENT_PID_FILE:-logs/figure4_256_64_full_suite/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

is_pid_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

current_pid() {
  [[ -f "${CURRENT_PID_FILE}" ]] && cat "${CURRENT_PID_FILE}" || true
}

queue_running() {
  [[ -f "${QUEUE_PID_FILE}" ]] && is_pid_running "$(cat "${QUEUE_PID_FILE}")"
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

set_scale() {
  if [[ "${MODE}" == "smoke" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1024}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-512}"
    NUM_ENVS="${NUM_ENVS:-8}"
    NUM_STEPS="${NUM_STEPS:-8}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-1}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-1}"
    NUM_SEEDS="${NUM_SEEDS:-2}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    EVAL_SEEDS="${EVAL_SEEDS:-5}"
  elif [[ "${MODE}" == "full" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-15000000}"
    NUM_ENVS="${NUM_ENVS:-128}"
    NUM_STEPS="${NUM_STEPS:-256}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    EVAL_SEEDS="${EVAL_SEEDS:-500}"
  else
    echo "[128-32-no-sa-full] unknown MODE=${MODE}; use smoke or full" >&2
    exit 2
  fi
}

write_seed_manifest() {
  local method="$1"
  "${PYTHON}" experiments/tools/seed_manifest.py \
    --seed "${SEED}" \
    --num-seeds "${NUM_SEEDS}" \
    --num-iterations 1 \
    --out "${JOB_ROOT}/seed_manifest_${method}_${JOB_TAG}.json" || true
}

summarize_csv() {
  local name="$1"
  local csv_path="$2"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[128-32-no-sa-full] missing_csv name=${name} csv=${csv_path}" >&2
    return 1
  fi
  "${PYTHON}" - <<PY
import csv, re, statistics
from pathlib import Path
path = Path("${csv_path}")
name = "${name}"
by = {}
with path.open() as f:
    for r in csv.DictReader(f):
        label = r.get("policy_labels", "")
        m = re.match(r"cross-(\d+)_(\d+)", label)
        if not m:
            continue
        pair = (int(m.group(1)), int(m.group(2)))
        by.setdefault(pair, []).append(float(r["total_reward"]))
pm = {k: sum(v) / len(v) for k, v in by.items()}
sp = [v for (i, j), v in pm.items() if i == j]
xp = [v for (i, j), v in pm.items() if i != j]
sp_mean = sum(sp) / len(sp) if sp else float("nan")
xp_mean = sum(xp) / len(xp) if xp else float("nan")
sp_std = statistics.pstdev(sp) if len(sp) > 1 else 0.0
xp_std = statistics.pstdev(xp) if len(xp) > 1 else 0.0
print(f"[128-32-no-sa-full] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} csv={path}")
PY
}

eval_ppo_like() {
  local name="$1"
  local run_dir="$2"
  local script="$3"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[128-32-no-sa-full] missing_run_dir name=${name} run_dir=${run_dir}" >&2
    exit 2
  fi
  echo "[128-32-no-sa-full] eval_start=$(date -Is) method=${name} run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${script}" \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz
  summarize_csv "${name}" "${run_dir}/reward_summary_cross.csv"
  echo "[128-32-no-sa-full] eval_done=$(date -Is) method=${name}"
}

train_ppo_full_obs() {
  local arch="$1"
  local method="ppo_${arch}_no_state_aug_full_obs"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[128-32-no-sa-full] train_start=$(date -Is) method=${method} prefix=${prefix}"
  write_seed_manifest "${method}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/main.py \
    +env=default \
    model="${arch}" \
    +env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[128-32-no-sa-full] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py"
}

train_mappo_full_obs() {
  local arch="$1"
  local method="mappo_${arch}_no_state_aug_full_obs"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[128-32-no-sa-full] train_start=$(date -Is) method=${method} prefix=${prefix}"
  write_seed_manifest "${method}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/main.py \
    +env=default \
    model="${arch}" \
    +env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[128-32-no-sa-full] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/mappo/utils/visualize.py"
}

train_ppo_e3t_full_obs() {
  local stage="$1"
  local actor_condition="predicted_partner"
  local enable_ce="True"
  case "${stage}" in
    predicted_ce) actor_condition="predicted_partner"; enable_ce="True" ;;
    no_ce) actor_condition="predicted_partner"; enable_ce="False" ;;
    *) echo "[128-32-no-sa-full] unknown ppo_e3t stage=${stage}" >&2; exit 2 ;;
  esac
  local method="ppo_e3t_${stage}_no_state_aug_full_obs"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[128-32-no-sa-full] train_start=$(date -Is) method=${method} prefix=${prefix}"
  write_seed_manifest "${method}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo_e3t_official/main.py \
    +env=default \
    model=cnn \
    +env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.E3T_PARTNER_SOURCE="main_policy" \
    model.E3T_ACTOR_CONDITION="${actor_condition}" \
    model.E3T_ENABLE_CE="${enable_ce}" \
    model.E3T_CONDITION_ACTOR=True \
    model.USE_OFFICIAL_E3T_PARTNER=False \
    model.RAND=0.0 \
    model.COPY=0.0
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[128-32-no-sa-full] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/visualize_ppo.py"
}

run_method() {
  case "$1" in
    ppo_cnn_no_state_aug_full_obs) train_ppo_full_obs cnn ;;
    ppo_rnn_no_state_aug_full_obs) train_ppo_full_obs rnn ;;
    mappo_cnn_no_state_aug_full_obs) train_mappo_full_obs cnn ;;
    mappo_rnn_no_state_aug_full_obs) train_mappo_full_obs rnn ;;
    ppo_e3t_predicted_ce_no_state_aug_full_obs) train_ppo_e3t_full_obs predicted_ce ;;
    ppo_e3t_no_ce_no_state_aug_full_obs) train_ppo_e3t_full_obs no_ce ;;
    *) echo "[128-32-no-sa-full] unknown method=$1" >&2; exit 2 ;;
  esac
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_128_32_no_state_aug_full_obs}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-ppo_cnn_no_state_aug_full_obs ppo_rnn_no_state_aug_full_obs mappo_cnn_no_state_aug_full_obs mappo_rnn_no_state_aug_full_obs ppo_e3t_predicted_ce_no_state_aug_full_obs ppo_e3t_no_ce_no_state_aug_full_obs}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  set_scale
  echo "[128-32-no-sa-full] launch_time=$(date -Is) host=$(hostname) root=${ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[128-32-no-sa-full] mode=${MODE} layout=${LAYOUT} seed=${SEED} methods=${METHODS} obs=full state_aug=false"
  echo "[128-32-no-sa-full] scale total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} num_seeds=${NUM_SEEDS} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} ce_epochs=${CONTEXT_UPDATE_EPOCHS} eval_seeds=${EVAL_SEEDS}"
  for method in ${METHODS}; do
    run_method "${method}"
  done
  echo "[128-32-no-sa-full] pipeline_done=$(date -Is)"
}

queue() {
  trap 'rm -f "${QUEUE_PID_FILE}"' EXIT
  local cpid
  cpid="$(current_pid)"
  if is_pid_running "${cpid}"; then
    echo "[128-32-no-sa-full] queued_after_current current_pid=${cpid} wait_seconds=${WAIT_SECONDS} start_time=$(date -Is)"
    while is_pid_running "${cpid}"; do
      sleep "${WAIT_SECONDS}"
    done
    echo "[128-32-no-sa-full] current_finished observed=$(date -Is)"
  else
    echo "[128-32-no-sa-full] no_current_pipeline_starting_now time=$(date -Is)"
  fi
  pipeline
}

status() {
  if queue_running; then
    local qpid
    qpid="$(cat "${QUEUE_PID_FILE}")"
    echo "[128-32-no-sa-full] queue_running pid=${qpid}"
    ps -fp "${qpid}" || true
  else
    echo "[128-32-no-sa-full] queue_not_running"
    [[ -f "${QUEUE_PID_FILE}" ]] && echo "[128-32-no-sa-full] stale_queue_pid=$(cat "${QUEUE_PID_FILE}")"
  fi
  local cpid
  cpid="$(current_pid)"
  if is_pid_running "${cpid}"; then
    echo "[128-32-no-sa-full] current_pipeline_running pid=${cpid}"
    pgrep -P "${cpid}" -af || true
  else
    echo "[128-32-no-sa-full] no_active_current_pipeline"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[128-32-no-sa-full] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 80 "${log}"
  fi
}

start() {
  if queue_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_128_32_no_state_aug_full_obs}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-ppo_cnn_no_state_aug_full_obs ppo_rnn_no_state_aug_full_obs mappo_cnn_no_state_aug_full_obs mappo_rnn_no_state_aug_full_obs ppo_e3t_predicted_ce_no_state_aug_full_obs ppo_e3t_no_ce_no_state_aug_full_obs}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="${JOB_ROOT}/figure4_128_32_no_state_aug_full_obs_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" MODE="${MODE}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" \
    SEED="${SEED}" METHODS="${METHODS}" WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" \
    NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" \
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" \
    NUM_SEEDS="${NUM_SEEDS:-}" NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-42}" \
    CURRENT_PID_FILE="${CURRENT_PID_FILE}" QUEUE_PID_FILE="${QUEUE_PID_FILE}" LATEST_LOG="${LATEST_LOG}" \
    JOB_ROOT="${JOB_ROOT}" WAIT_SECONDS="${WAIT_SECONDS}" \
    bash "${BASH_SOURCE[0]}" queue > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${QUEUE_PID_FILE}"
  echo "[128-32-no-sa-full] queued pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  queue) queue ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if queue_running; then kill "$(cat "${QUEUE_PID_FILE}")"; else echo "[128-32-no-sa-full] queue_not_running"; fi ;;
  *) echo "Usage: $0 [start|queue|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
