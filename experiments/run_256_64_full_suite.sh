#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/figure4_256_64_full_suite}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "${JOB_ROOT}"
source "${ROOT}/experiments/repro_env.sh"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[256-64] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[256-64] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[256-64] not running"
    [[ -f "${PID_FILE}" ]] && echo "[256-64] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[256-64] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
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
    NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
    EVAL_SEEDS="${EVAL_SEEDS:-5}"
  elif [[ "${MODE}" == "full" ]]; then
    # RNN historical/default paper-budget scale, applied to every method for this controlled rerun.
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-15000000}"
    NUM_ENVS="${NUM_ENVS:-256}"
    NUM_STEPS="${NUM_STEPS:-256}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
    EVAL_SEEDS="${EVAL_SEEDS:-500}"
  else
    echo "[256-64] unknown MODE=${MODE}; use smoke or full" >&2
    exit 2
  fi
}

summarize_csv() {
  local name="$1"
  local csv="$2"
  if [[ ! -f "${csv}" ]]; then
    echo "[256-64] missing_csv name=${name} csv=${csv}" >&2
    return 1
  fi
  "${PYTHON}" - <<PY
import csv, re, statistics
from pathlib import Path
path = Path("${csv}")
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
print(f"[256-64] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} csv={path}")
PY
}

train_ppo() {
  local arch="$1"
  local method="ppo_${arch}_state_aug"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  local model="${arch}"
  echo "[256-64] train_start=$(date -Is) method=${method} prefix=${prefix}"
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${SEED}" --num-seeds "${NUM_SEEDS}" --num-iterations "${NUM_ITERATIONS}" --out "${JOB_ROOT}/seed_manifest_${method}_${JOB_TAG}.json" || true
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/main.py \
    +env=default \
    model="${model}" \
    +NUM_ITERATIONS="${NUM_ITERATIONS}" \
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
  echo "[256-64] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py"
}

train_mappo() {
  local arch="$1"
  local method="mappo_${arch}_state_aug"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  local model="${arch}"
  echo "[256-64] train_start=$(date -Is) method=${method} prefix=${prefix}"
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${SEED}" --num-seeds "${NUM_SEEDS}" --num-iterations "${NUM_ITERATIONS}" --out "${JOB_ROOT}/seed_manifest_${method}_${JOB_TAG}.json" || true
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/main.py \
    +env=default \
    model="${model}" \
    +NUM_ITERATIONS="${NUM_ITERATIONS}" \
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
  echo "[256-64] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/mappo/utils/visualize.py"
}

train_ppo_e3t() {
  local stage="$1"
  local actor_condition="predicted_partner"
  local enable_ce="True"
  local condition_actor="True"
  case "${stage}" in
    predicted_ce) actor_condition="predicted_partner"; enable_ce="True"; condition_actor="True" ;;
    constant_ce) actor_condition="constant"; enable_ce="True"; condition_actor="True" ;;
    no_ce) actor_condition="predicted_partner"; enable_ce="False"; condition_actor="True" ;;
    no_actor_condition) actor_condition="predicted_partner"; enable_ce="True"; condition_actor="False" ;;
    *) echo "[256-64] unknown ppo_e3t stage=${stage}" >&2; exit 2 ;;
  esac
  local method="ppo_e3t_${stage}_state_aug"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[256-64] train_start=$(date -Is) method=${method} prefix=${prefix}"
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${SEED}" --num-seeds "${NUM_SEEDS}" --num-iterations "${NUM_ITERATIONS}" --out "${JOB_ROOT}/seed_manifest_${method}_${JOB_TAG}.json" || true
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo_e3t_official/main.py \
    +env=default \
    model=cnn \
    +NUM_ITERATIONS="${NUM_ITERATIONS}" \
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
    model.E3T_CONDITION_ACTOR="${condition_actor}" \
    model.USE_OFFICIAL_E3T_PARTNER=False \
    model.RAND=0.0 \
    model.COPY=0.0
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[256-64] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_ppo_like "${method}" "${run_dir}" "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/visualize_ppo.py"
}

eval_ppo_like() {
  local name="$1"
  local run_dir="$2"
  local script="$3"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[256-64] missing_run_dir name=${name} run_dir=${run_dir}" >&2
    exit 2
  fi
  echo "[256-64] eval_start=$(date -Is) method=${name} run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${script}" \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz
  summarize_csv "${name}" "${run_dir}/reward_summary_cross.csv"
  echo "[256-64] eval_done=$(date -Is) method=${name}"
}

run_method() {
  case "$1" in
    ppo_cnn_state_aug) train_ppo cnn ;;
    ppo_rnn_state_aug) train_ppo rnn ;;
    mappo_cnn_state_aug) train_mappo cnn ;;
    mappo_rnn_state_aug) train_mappo rnn ;;
    ppo_e3t_predicted_ce|ppo_e3t_predicted_ce_state_aug) train_ppo_e3t predicted_ce ;;
    ppo_e3t_constant_ce|ppo_e3t_constant_ce_state_aug) train_ppo_e3t constant_ce ;;
    ppo_e3t_no_ce|ppo_e3t_no_ce_state_aug) train_ppo_e3t no_ce ;;
    ppo_e3t_no_actor_condition|ppo_e3t_no_actor_condition_state_aug) train_ppo_e3t no_actor_condition ;;
    *) echo "[256-64] unknown method=$1" >&2; exit 2 ;;
  esac
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_256_64}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-ppo_cnn_state_aug ppo_rnn_state_aug mappo_cnn_state_aug mappo_rnn_state_aug ppo_e3t_predicted_ce}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  set_scale
  echo "[256-64] launch_time=$(date -Is) host=$(hostname) root=${ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[256-64] mode=${MODE} layout=${LAYOUT} seed=${SEED} methods=${METHODS}"
  echo "[256-64] scale total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} num_seeds=${NUM_SEEDS} iterations=${NUM_ITERATIONS} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} ce_epochs=${CONTEXT_UPDATE_EPOCHS} eval_seeds=${EVAL_SEEDS}"
  for method in ${METHODS}; do
    run_method "${method}"
  done
  echo "[256-64] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_256_64}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-ppo_cnn_state_aug ppo_rnn_state_aug mappo_cnn_state_aug mappo_rnn_state_aug ppo_e3t_predicted_ce}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="${JOB_ROOT}/figure4_256_64_full_suite_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" MODE="${MODE}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" \
    SEED="${SEED}" METHODS="${METHODS}" WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" \
    NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" \
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" \
    NUM_SEEDS="${NUM_SEEDS:-}" NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-}" NUM_ITERATIONS="${NUM_ITERATIONS:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-42}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[256-64] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline)
    : "${MODE:=full}" "${PREFIX_BASE:=figure4_256_64}" "${LAYOUT:=counter_circuit}" "${SEED:=42}"
    : "${METHODS:=ppo_cnn_state_aug ppo_rnn_state_aug mappo_cnn_state_aug mappo_rnn_state_aug ppo_e3t_predicted_ce}"
    : "${WANDB_MODE:=online}" "${WANDB_PROJECT:=ov2-paper-repro}" "${WANDB_ENTITY:=huiby_tsinghua23}" "${EVAL_SEED:=42}"
    pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[256-64] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
