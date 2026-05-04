#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/mappo_rnn_64_16_rerun}"
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

summarize_csv() {
  local name="$1"
  local csv_path="$2"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[mappo-rnn-64-16] missing_csv name=${name} csv=${csv_path}" >&2
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
print(f"[mappo-rnn-64-16] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} csv={path}")
PY
}

write_seed_manifest() {
  local method="$1"
  local iterations="$2"
  "${PYTHON}" experiments/tools/seed_manifest.py \
    --seed "${SEED}" \
    --num-seeds "${NUM_SEEDS}" \
    --num-iterations "${iterations}" \
    --out "${JOB_ROOT}/seed_manifest_${method}_${JOB_TAG}.json" || true
}

eval_mappo() {
  local name="$1"
  local run_dir="$2"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[mappo-rnn-64-16] missing_run_dir name=${name} run_dir=${run_dir}" >&2
    exit 2
  fi
  echo "[mappo-rnn-64-16] eval_start=$(date -Is) method=${name} run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/utils/visualize.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz
  summarize_csv "${name}" "${run_dir}/reward_summary_cross.csv"
  echo "[mappo-rnn-64-16] eval_done=$(date -Is) method=${name}"
}

train_common_args=()

train_standard() {
  local method="mappo_rnn_standard"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[mappo-rnn-64-16] train_start=$(date -Is) method=${method} prefix=${prefix}"
  write_seed_manifest "${method}" 1
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/main.py \
    +env=default \
    model=rnn \
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
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.LR="${LR}" \
    model.ENT_COEF="${ENT_COEF}" \
    model.MAX_GRAD_NORM="${MAX_GRAD_NORM}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[mappo-rnn-64-16] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_mappo "${method}" "${run_dir}"
}

train_state_aug() {
  local method="mappo_rnn_state_aug"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  echo "[mappo-rnn-64-16] train_start=$(date -Is) method=${method} prefix=${prefix} iterations=${NUM_ITERATIONS}"
  write_seed_manifest "${method}" "${NUM_ITERATIONS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/main.py \
    +env=default \
    model=rnn \
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
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    model.LR="${LR}" \
    model.ENT_COEF="${ENT_COEF}" \
    model.MAX_GRAD_NORM="${MAX_GRAD_NORM}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[mappo-rnn-64-16] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_mappo "${method}" "${run_dir}"
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_mappo_rnn_64_16_rerun}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export NUM_SEEDS="${NUM_SEEDS:-10}"
  export NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  export NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
  export REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  export NUM_ENVS="${NUM_ENVS:-64}"
  export NUM_STEPS="${NUM_STEPS:-256}"
  export UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
  export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  export LR="${LR:-0.00025}"
  export ENT_COEF="${ENT_COEF:-0.01}"
  export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.25}"
  export EVAL_SEEDS="${EVAL_SEEDS:-500}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export METHODS="${METHODS:-standard state_aug}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  echo "[mappo-rnn-64-16] launch_time=$(date -Is) host=$(hostname) root=${ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[mappo-rnn-64-16] seed=${SEED} num_seeds=${NUM_SEEDS} methods=${METHODS} layout=${LAYOUT}"
  echo "[mappo-rnn-64-16] scale total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} iterations=${NUM_ITERATIONS} eval_seeds=${EVAL_SEEDS}"
  echo "[mappo-rnn-64-16] hparams LR=${LR} ENT_COEF=${ENT_COEF} MAX_GRAD_NORM=${MAX_GRAD_NORM}"
  for method in ${METHODS}; do
    case "${method}" in
      standard) train_standard ;;
      state_aug) train_state_aug ;;
      *) echo "[mappo-rnn-64-16] unknown method=${method}" >&2; exit 2 ;;
    esac
  done
  echo "[mappo-rnn-64-16] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[mappo-rnn-64-16] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[mappo-rnn-64-16] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[mappo-rnn-64-16] not running"
    [[ -f "${PID_FILE}" ]] && echo "[mappo-rnn-64-16] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[mappo-rnn-64-16] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="${JOB_ROOT}/mappo_rnn_64_16_rerun_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    PREFIX_BASE="${PREFIX_BASE:-}" LAYOUT="${LAYOUT:-}" SEED="${SEED:-}" METHODS="${METHODS:-}" \
    WANDB_MODE="${WANDB_MODE:-}" WANDB_PROJECT="${WANDB_PROJECT:-}" WANDB_ENTITY="${WANDB_ENTITY:-}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" \
    NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" NUM_SEEDS="${NUM_SEEDS:-}" NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-}" \
    NUM_ITERATIONS="${NUM_ITERATIONS:-}" LR="${LR:-}" ENT_COEF="${ENT_COEF:-}" MAX_GRAD_NORM="${MAX_GRAD_NORM:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-42}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[mappo-rnn-64-16] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[mappo-rnn-64-16] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
