#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/official_fcp_64_16_official_aligned}"
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
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

summarize_cross() {
  local name="$1"
  local csv_path="$2"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[official-fcp-aligned] missing_csv name=${name} csv=${csv_path}" >&2
    return 1
  fi
  "${PYTHON}" - "${name}" "${csv_path}" <<'PY'
import csv, math, re, statistics, sys
name, path = sys.argv[1], sys.argv[2]
by_pair = {}
with open(path) as f:
    for r in csv.DictReader(f):
        m = re.match(r"cross-(\d+)_(\d+)$", r.get("policy_labels", ""))
        if not m:
            continue
        pair = (int(m.group(1)), int(m.group(2)))
        by_pair.setdefault(pair, []).append(float(r["total_reward"]))
pair_mean = {k: sum(v) / len(v) for k, v in by_pair.items()}
sp = [v for (i, j), v in pair_mean.items() if i == j]
xp = [v for (i, j), v in pair_mean.items() if i != j]
sp_mean = sum(sp) / len(sp) if sp else math.nan
xp_mean = sum(xp) / len(xp) if xp else math.nan
sp_std = statistics.pstdev(sp) if len(sp) > 1 else 0.0
xp_std = statistics.pstdev(xp) if len(xp) > 1 else 0.0
print(f"[official-fcp-aligned] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} n_sp_pairs={len(sp)} n_xp_pairs={len(xp)} csv={path}")
PY
}

train_population() {
  local existing
  existing="$(latest_run_dir "${POP_PREFIX}")"
  if [[ -n "${existing}" && -d "${existing}/run_$((POP_NUM_SEEDS - 1))/ckpt_final" ]]; then
    echo "[official-fcp-aligned] population_reuse run_dir=${existing}"
    printf "%s\n" "${existing}" > "${JOB_ROOT}/population.run_dir"
    return
  fi

  echo "[official-fcp-aligned] population_train_start=$(date -Is) prefix=${POP_PREFIX} num_seeds=${POP_NUM_SEEDS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-sp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${POP_NUM_SEEDS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${POP_PREFIX}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"

  local run_dir
  run_dir="$(latest_run_dir "${POP_PREFIX}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_$((POP_NUM_SEEDS - 1))/ckpt_final" ]]; then
    echo "[official-fcp-aligned] population_missing_checkpoint run_dir=${run_dir}" >&2
    exit 1
  fi
  printf "%s\n" "${run_dir}" > "${JOB_ROOT}/population.run_dir"
  echo "[official-fcp-aligned] population_train_done=$(date -Is) run_dir=${run_dir}"
}

copy_population_official() {
  local run_dir rel
  run_dir="$(cat "${JOB_ROOT}/population.run_dir")"
  rel="${run_dir#runs/}"
  export FCP_DIR="fcp_populations/${rel}"
  rm -rf "${FCP_DIR}"
  echo "[official-fcp-aligned] copy_fcp_start=$(date -Is) rel=${rel} dest=${FCP_DIR}"
  bash experiments/copy_fcp.sh "${rel}"
  local group_count
  group_count="$(find "${FCP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'fcp_*' | wc -l | tr -d ' ')"
  echo "[official-fcp-aligned] copy_fcp_done=$(date -Is) dest=${FCP_DIR} groups=${group_count}"
  if [[ "${group_count}" != "${FCP_GROUPS}" ]]; then
    echo "[official-fcp-aligned] unexpected_fcp_group_count expected=${FCP_GROUPS} actual=${group_count}" >&2
    exit 1
  fi
  printf "%s\n" "${FCP_DIR}" > "${JOB_ROOT}/fcp_population.dir"
}

train_fcp() {
  local existing fcp_dir
  fcp_dir="$(cat "${JOB_ROOT}/fcp_population.dir")"
  existing="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -n "${existing}" && -d "${existing}/run_$((FCP_GROUPS - 1))/ckpt_final" ]]; then
    echo "[official-fcp-aligned] fcp_reuse run_dir=${existing}"
    printf "%s\n" "${existing}" > "${JOB_ROOT}/fcp.run_dir"
    return
  fi

  echo "[official-fcp-aligned] fcp_train_start=$(date -Is) prefix=${FCP_PREFIX} pop=${fcp_dir}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-fcp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    +FCP="${fcp_dir}" \
    SEED="${SEED}" \
    NUM_SEEDS=1 \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${FCP_PREFIX}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"

  local run_dir
  run_dir="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_$((FCP_GROUPS - 1))/ckpt_final" ]]; then
    echo "[official-fcp-aligned] fcp_missing_checkpoint run_dir=${run_dir}" >&2
    exit 1
  fi
  printf "%s\n" "${run_dir}" > "${JOB_ROOT}/fcp.run_dir"
  echo "[official-fcp-aligned] fcp_train_done=$(date -Is) run_dir=${run_dir}"
}

evaluate_fcp() {
  local run_dir
  run_dir="$(cat "${JOB_ROOT}/fcp.run_dir")"
  echo "[official-fcp-aligned] eval_start=$(date -Is) run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz
  summarize_cross "fcp_rnn_official_aligned_64_16" "${run_dir}/reward_summary_cross.csv"
  echo "[official-fcp-aligned] eval_done=$(date -Is) run_dir=${run_dir}"
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export POP_NUM_SEEDS="${POP_NUM_SEEDS:-80}"
  export FCP_GROUPS="${FCP_GROUPS:-10}"
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
  export REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  export NUM_ENVS="${NUM_ENVS:-64}"
  export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  export EVAL_SEEDS="${EVAL_SEEDS:-500}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export POP_PREFIX="${POP_PREFIX:-official_fcp64_16_aligned_sp80_${JOB_TAG}}"
  export FCP_PREFIX="${FCP_PREFIX:-official_fcp64_16_aligned_rnn_fcp_${JOB_TAG}}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

  print_repro_env
  echo "[official-fcp-aligned] pipeline_start=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[official-fcp-aligned] layout=${LAYOUT} seed=${SEED} pop_num_seeds=${POP_NUM_SEEDS} fcp_groups=${FCP_GROUPS}"
  echo "[official-fcp-aligned] aligned_overrides total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} envs=${NUM_ENVS} minibatches=${NUM_MINIBATCHES}; other rnn-sp/rnn-fcp params left official"
  echo "[official-fcp-aligned] prefixes pop=${POP_PREFIX} fcp=${FCP_PREFIX}"

  train_population
  copy_population_official
  train_fcp
  evaluate_fcp
  echo "[official-fcp-aligned] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[official-fcp-aligned] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[official-fcp-aligned] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[official-fcp-aligned] not running"
    [[ -f "${PID_FILE}" ]] && echo "[official-fcp-aligned] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[official-fcp-aligned] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 180 "${log}"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  local log="${JOB_ROOT}/official_fcp_aligned_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${tag}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHON="${PYTHON}" \
    LAYOUT="${LAYOUT:-}" SEED="${SEED:-}" POP_NUM_SEEDS="${POP_NUM_SEEDS:-}" FCP_GROUPS="${FCP_GROUPS:-}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" NUM_ENVS="${NUM_ENVS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-}" WANDB_MODE="${WANDB_MODE:-}" WANDB_PROJECT="${WANDB_PROJECT:-}" WANDB_ENTITY="${WANDB_ENTITY:-}" \
    POP_PREFIX="${POP_PREFIX:-}" FCP_PREFIX="${FCP_PREFIX:-}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[official-fcp-aligned] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[official-fcp-aligned] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
