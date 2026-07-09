#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
JOB_ROOT="${JOB_ROOT:-${ROOT}/logs/ac_method_migrations_gate}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
mkdir -p "${JOB_ROOT}"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

summarize_csv() {
  local method="$1"
  local run_dir="$2"
  local tag="$3"
  "${PYTHON}" - <<PY
import csv, re
from pathlib import Path

method = "${method}"
run_dir = Path("${run_dir}")
tag = "${tag}"
sp_path = run_dir / f"reward_summary_sp_{tag}.csv"
xp_path = run_dir / f"reward_summary_cross_{tag}.csv"

def mean(values):
    return sum(values) / len(values) if values else float("nan")

sp_vals = []
if sp_path.exists():
    with sp_path.open() as f:
        sp_vals = [float(r["total_reward"]) for r in csv.DictReader(f)]

xp_vals = []
cross_sp_vals = []
if xp_path.exists():
    with xp_path.open() as f:
        for r in csv.DictReader(f):
            label = r.get("policy_labels", "")
            m = re.match(r"cross-(\d+)_(\d+)", label)
            if not m:
                continue
            val = float(r["total_reward"])
            if m.group(1) == m.group(2):
                cross_sp_vals.append(val)
            else:
                xp_vals.append(val)

print(
    f"[ac-method-gate] summary method={method} "
    f"SP={mean(sp_vals):.3f} XP={mean(xp_vals):.3f} "
    f"cross_self={mean(cross_sp_vals):.3f} "
    f"sp_rows={len(sp_vals)} xp_rows={len(xp_vals)} run_dir={run_dir}",
    flush=True,
)
PY
}

eval_one() {
  local method="$1"
  local run_dir="$2"
  local tag="eval${EVAL_SEEDS}"
  echo "[ac-method-gate] eval_start=$(date -Is) method=${method} run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${ROOT}/experiments/overcooked_v2_experiments/mappo/utils/visualize.py" \
    --d "${run_dir}" \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz \
    --output_tag "${tag}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${ROOT}/experiments/overcooked_v2_experiments/mappo/utils/visualize.py" \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz \
    --output_tag "${tag}"
  summarize_csv "${method}" "${run_dir}" "${tag}"
  echo "[ac-method-gate] eval_done=$(date -Is) method=${method}"
}

train_one() {
  local method="$1"
  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  local run_dir="${RUN_ROOT}/${prefix}"
  local extra_args=()
  if [[ "${FULL_TRAIN_MASK}" == "1" || "${FULL_TRAIN_MASK}" == "true" || "${FULL_TRAIN_MASK}" == "True" ]]; then
    extra_args+=(--full_train_mask)
  fi
  echo "[ac-method-gate] train_start=$(date -Is) method=${method} prefix=${prefix}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" "${ROOT}/experiments/overcooked_v2_experiments/ac_method_migrations/train.py" \
    --method "${method}" \
    --optional_prefix "${prefix}" \
    --output_root "${RUN_ROOT}" \
    --layout "${LAYOUT}" \
    --seed "${SEED}" \
    --num_seeds "${NUM_SEEDS}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --rew_shaping_horizon "${REW_SHAPING_HORIZON}" \
    --num_envs "${NUM_ENVS}" \
    --num_steps "${NUM_STEPS}" \
    --update_epochs "${UPDATE_EPOCHS}" \
    --num_minibatches "${NUM_MINIBATCHES}" \
    --lr "${LR}" \
    --lr_warmup "${LR_WARMUP}" \
    --critic_hidden_size "${CRITIC_HIDDEN_SIZE}" \
    "${extra_args[@]}"
  echo "[ac-method-gate] train_done=$(date -Is) method=${method} run_dir=${run_dir}"
  eval_one "${method}" "${run_dir}"
}

pipeline() {
  cd "${ROOT}"
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-ac_method_migration_gate}"
  export RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/ac_method_migrations_gate}"
  export METHODS="${METHODS:-coma_ppo maac happo vtrace}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export NUM_SEEDS="${NUM_SEEDS:-10}"
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
  export REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-500000}"
  export NUM_ENVS="${NUM_ENVS:-64}"
  export NUM_STEPS="${NUM_STEPS:-256}"
  export UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
  export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  export LR="${LR:-0.0004}"
  export LR_WARMUP="${LR_WARMUP:-0.05}"
  export CRITIC_HIDDEN_SIZE="${CRITIC_HIDDEN_SIZE:-128}"
  export FULL_TRAIN_MASK="${FULL_TRAIN_MASK:-0}"
  export EVAL_SEEDS="${EVAL_SEEDS:-50}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  mkdir -p "${RUN_ROOT}"
  echo "[ac-method-gate] launch_time=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[ac-method-gate] methods=${METHODS} num_seeds=${NUM_SEEDS} total=${TOTAL_TIMESTEPS} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} full_train_mask=${FULL_TRAIN_MASK}"
  for method in ${METHODS}; do
    train_one "${method}"
  done
  echo "[ac-method-gate] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[ac-method-gate] running pid=${pid}"
    ps -fp "${pid}" || true
    pgrep -P "${pid}" -af || true
  else
    echo "[ac-method-gate] not running"
    [[ -f "${PID_FILE}" ]] && echo "[ac-method-gate] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[ac-method-gate] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  local log="${JOB_ROOT}/ac_method_migrations_gate_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    PREFIX_BASE="${PREFIX_BASE:-}" RUN_ROOT="${RUN_ROOT:-}" METHODS="${METHODS:-}" LAYOUT="${LAYOUT:-}" \
    SEED="${SEED:-}" NUM_SEEDS="${NUM_SEEDS:-}" TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" \
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" LR="${LR:-}" \
    LR_WARMUP="${LR_WARMUP:-}" CRITIC_HIDDEN_SIZE="${CRITIC_HIDDEN_SIZE:-}" \
    FULL_TRAIN_MASK="${FULL_TRAIN_MASK:-}" EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-}" \
    bash "$0" run >"${log}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "[ac-method-gate] started pid=$(cat "${PID_FILE}") log=${log}"
}

stop() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[ac-method-gate] stopping pid=${pid}"
    kill "${pid}" || true
  else
    echo "[ac-method-gate] not running"
  fi
}

case "${1:-status}" in
  run) pipeline ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|stop|run}" >&2; exit 2 ;;
esac
