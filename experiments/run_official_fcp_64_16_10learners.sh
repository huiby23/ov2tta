#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/official_fcp_64_16_10learners}"
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
    echo "[official-fcp-64-16] missing_csv name=${name} csv=${csv_path}" >&2
    return 1
  fi
  "${PYTHON}" - "${name}" "${csv_path}" <<'PY'
import csv, math, re, statistics, sys
name, path = sys.argv[1], sys.argv[2]
by_pair = {}
with open(path) as f:
    for r in csv.DictReader(f):
        label = r.get("policy_labels", "")
        m = re.match(r"cross-(\d+)_(\d+)$", label)
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
print(f"[official-fcp-64-16] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} n_sp_pairs={len(sp)} n_xp_pairs={len(xp)} csv={path}")
PY
}

register_result() {
  local run_dir="$1"
  local result_file="$2"
  mkdir -p runs_by_config/counter_circuit_full_obs_64env_16mb_10M/fcp/rnn_official_10learners
  ln -sfn "${ROOT}/${run_dir}" runs_by_config/counter_circuit_full_obs_64env_16mb_10M/fcp/rnn_official_10learners/latest
  if [[ -f experiments/run_registry.tsv ]]; then
    if ! grep -Fq "fcp_rnn_official_10learners_64_16" experiments/run_registry.tsv; then
      printf "counter_circuit_full_obs_64env_16mb_10M\tfcp\trnn_official_10learners\tfcp_rnn_official_10learners_64_16\t%s\t%s\tofficial FCP flow: 10 groups x 8 SP policies, 64 env / 16 minibatch / 10M\n" "${run_dir}" "${result_file}" >> experiments/run_registry.tsv
    fi
  fi
}

train_sp_group() {
  local group="$1"
  local group_seed=$((SEED + group * SEED_STRIDE))
  local prefix="${POP_PREFIX}_g$(printf '%02d' "${group}")"
  local existing
  existing="$(latest_run_dir "${prefix}")"
  if [[ -n "${existing}" && -d "${existing}/run_7/ckpt_final" ]]; then
    echo "[official-fcp-64-16] sp_group_reuse group=${group} run_dir=${existing}"
    printf "%s\n" "${existing}" > "${JOB_ROOT}/sp_group_${group}.run_dir"
    return
  fi
  echo "[official-fcp-64-16] sp_group_train_start=$(date -Is) group=${group} seed=${group_seed} prefix=${prefix}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-sp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${group_seed}" \
    NUM_SEEDS="${POPULATION_SIZE}" \
    NUM_CHECKPOINTS="${POP_NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_7/ckpt_final" ]]; then
    echo "[official-fcp-64-16] sp_group_missing_checkpoint group=${group} run_dir=${run_dir}" >&2
    exit 1
  fi
  printf "%s\n" "${run_dir}" > "${JOB_ROOT}/sp_group_${group}.run_dir"
  echo "[official-fcp-64-16] sp_group_train_done=$(date -Is) group=${group} run_dir=${run_dir}"
}

prepare_population() {
  rm -rf "${POP_DIR}"
  mkdir -p "${POP_DIR}"
  for group in $(seq 0 $((FCP_GROUPS - 1))); do
    local run_dir
    run_dir="$(cat "${JOB_ROOT}/sp_group_${group}.run_dir")"
    local dst_group="${POP_DIR}/fcp_$(printf '%02d' "${group}")"
    mkdir -p "${dst_group}"
    for policy in $(seq 0 $((POPULATION_SIZE - 1))); do
      local src="${run_dir}/run_${policy}"
      local dst="${dst_group}/run_${policy}"
      if [[ ! -d "${src}/ckpt_final" ]]; then
        echo "[official-fcp-64-16] missing_sp_ckpt group=${group} policy=${policy} src=${src}" >&2
        exit 1
      fi
      cp -a "${src}" "${dst}"
    done
  done
  echo "[official-fcp-64-16] population_ready dir=${POP_DIR} groups=${FCP_GROUPS} size=${POPULATION_SIZE} time=$(date -Is)"
}

train_fcp() {
  local existing
  existing="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -n "${existing}" && -d "${existing}/run_9/ckpt_final" ]]; then
    echo "[official-fcp-64-16] fcp_reuse run_dir=${existing}"
    printf "%s\n" "${existing}" > "${JOB_ROOT}/fcp.run_dir"
    return
  fi
  echo "[official-fcp-64-16] fcp_train_start=$(date -Is) prefix=${FCP_PREFIX} pop=${POP_DIR}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-fcp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    +FCP="${POP_DIR}" \
    SEED="${SEED}" \
    NUM_SEEDS=1 \
    NUM_CHECKPOINTS="${FCP_NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${FCP_PREFIX}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS="${NUM_ENVS}" \
    model.NUM_STEPS="${NUM_STEPS}" \
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  local run_dir
  run_dir="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_9/ckpt_final" ]]; then
    echo "[official-fcp-64-16] fcp_missing_checkpoint run_dir=${run_dir}" >&2
    exit 1
  fi
  printf "%s\n" "${run_dir}" > "${JOB_ROOT}/fcp.run_dir"
  echo "[official-fcp-64-16] fcp_train_done=$(date -Is) run_dir=${run_dir}"
}

evaluate_fcp() {
  local run_dir
  run_dir="$(cat "${JOB_ROOT}/fcp.run_dir")"
  echo "[official-fcp-64-16] eval_start=$(date -Is) run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --seed "${EVAL_SEED}" \
    --no_viz
  summarize_cross "fcp_rnn_official_10learners_64_16" "${run_dir}/reward_summary_cross.csv"
  register_result "${run_dir}" "reward_summary_cross.csv"
  echo "[official-fcp-64-16] eval_done=$(date -Is) run_dir=${run_dir}"
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export SEED_STRIDE="${SEED_STRIDE:-1000}"
  export FCP_GROUPS="${FCP_GROUPS:-10}"
  export POPULATION_SIZE="${POPULATION_SIZE:-8}"
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
  export REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  export NUM_ENVS="${NUM_ENVS:-64}"
  export NUM_STEPS="${NUM_STEPS:-256}"
  export UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
  export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  export POP_NUM_CHECKPOINTS="${POP_NUM_CHECKPOINTS:-3}"
  export FCP_NUM_CHECKPOINTS="${FCP_NUM_CHECKPOINTS:-1}"
  export EVAL_SEEDS="${EVAL_SEEDS:-500}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export POP_PREFIX="${POP_PREFIX:-official_fcp64_16_sp_pop_${JOB_TAG}}"
  export FCP_PREFIX="${FCP_PREFIX:-official_fcp64_16_rnn_10learners_${JOB_TAG}}"
  export POP_DIR="${POP_DIR:-fcp_populations/official_fcp64_16_counter_circuit_10x8_${JOB_TAG}}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

  print_repro_env
  echo "[official-fcp-64-16] pipeline_start=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[official-fcp-64-16] layout=${LAYOUT} seed=${SEED} groups=${FCP_GROUPS} pop_size=${POPULATION_SIZE} seed_stride=${SEED_STRIDE}"
  echo "[official-fcp-64-16] scale total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} envs=${NUM_ENVS} steps=${NUM_STEPS} epochs=${UPDATE_EPOCHS} minibatches=${NUM_MINIBATCHES} eval_seeds=${EVAL_SEEDS}"
  echo "[official-fcp-64-16] prefixes pop=${POP_PREFIX} fcp=${FCP_PREFIX} pop_dir=${POP_DIR}"

  for group in $(seq 0 $((FCP_GROUPS - 1))); do
    train_sp_group "${group}"
  done
  prepare_population
  train_fcp
  evaluate_fcp
  echo "[official-fcp-64-16] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[official-fcp-64-16] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[official-fcp-64-16] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[official-fcp-64-16] not running"
    [[ -f "${PID_FILE}" ]] && echo "[official-fcp-64-16] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[official-fcp-64-16] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 160 "${log}"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  local log="${JOB_ROOT}/official_fcp_64_16_10learners_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${tag}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHON="${PYTHON}" \
    LAYOUT="${LAYOUT:-}" SEED="${SEED:-}" SEED_STRIDE="${SEED_STRIDE:-}" FCP_GROUPS="${FCP_GROUPS:-}" POPULATION_SIZE="${POPULATION_SIZE:-}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" POP_NUM_CHECKPOINTS="${POP_NUM_CHECKPOINTS:-}" FCP_NUM_CHECKPOINTS="${FCP_NUM_CHECKPOINTS:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-}" WANDB_MODE="${WANDB_MODE:-}" WANDB_PROJECT="${WANDB_PROJECT:-}" WANDB_ENTITY="${WANDB_ENTITY:-}" \
    POP_PREFIX="${POP_PREFIX:-}" FCP_PREFIX="${FCP_PREFIX:-}" POP_DIR="${POP_DIR:-}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[official-fcp-64-16] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[official-fcp-64-16] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
