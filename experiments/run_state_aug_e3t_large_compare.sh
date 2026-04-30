#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh"

JOB_ROOT="${JOB_ROOT:-logs/state_aug_e3t_large_compare}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "${JOB_ROOT}"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[large-compare] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[large-compare] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[large-compare] not running"
    [[ -f "${PID_FILE}" ]] && echo "[large-compare] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[large-compare] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
}

set_scale() {
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1e7}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5e6}"
  NUM_ENVS="${NUM_ENVS:-64}"
  NUM_STEPS="${NUM_STEPS:-256}"
  UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
  CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
  NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  NUM_SEEDS="${NUM_SEEDS:-10}"
  NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
  NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
  EVAL_SEEDS="${EVAL_SEEDS:-500}"
}

summarize_csv() {
  local name="$1"
  local csv="$2"
  "${PYTHON}" - <<PY
import csv, re, statistics
from pathlib import Path
path = Path("${csv}")
name = "${name}"
by = {}
with path.open() as f:
    for r in csv.DictReader(f):
        m = re.match(r"cross-(\d+)_(\d+)", r["policy_labels"])
        if not m:
            continue
        p = (int(m.group(1)), int(m.group(2)))
        by.setdefault(p, []).append(float(r["total_reward"]))
pm = {k: sum(v) / len(v) for k, v in by.items()}
sp = [v for (i, j), v in pm.items() if i == j]
xp = [v for (i, j), v in pm.items() if i != j]
print(f"[large-compare] result name={name} SP={sum(sp)/len(sp):.4f} XP={sum(xp)/len(xp):.4f} SP_pair_std={statistics.pstdev(sp):.4f} XP_pair_std={statistics.pstdev(xp):.4f} csv={path}")
PY
}

run_ppo_state_aug() {
  local seed="$1"
  local prefix="${PREFIX_BASE}_ppo_state_aug_seed${seed}_${JOB_TAG}"
  echo "[large-compare] train_start=$(date -Is) method=ppo_state_aug seed=${seed} prefix=${prefix}"
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${seed}" --num-seeds "${NUM_SEEDS}" --num-iterations "${NUM_ITERATIONS}" --out "${JOB_ROOT}/seed_manifest_ppo_state_aug_seed${seed}_${JOB_TAG}.json" || true
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/main.py \
    +experiment=cnn \
    +env=original \
    +NUM_ITERATIONS="${NUM_ITERATIONS}" \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
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
  echo "[large-compare] train_done=$(date -Is) method=ppo_state_aug seed=${seed} run_dir=${run_dir}"
  echo "[large-compare] eval_start=$(date -Is) method=ppo_state_aug seed=${seed}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz
  summarize_csv "ppo_state_aug_seed${seed}" "${run_dir}/reward_summary_cross.csv"
  echo "[large-compare] eval_done=$(date -Is) method=ppo_state_aug seed=${seed}"
}

run_e3t_state_aug_predicted() {
  local seed="$1"
  local prefix="${PREFIX_BASE}_ppo_e3t_state_aug_predicted_ce_seed${seed}_${JOB_TAG}"
  echo "[large-compare] train_start=$(date -Is) method=ppo_e3t_state_aug_predicted_ce seed=${seed} prefix=${prefix}"
  "${PYTHON}" experiments/tools/seed_manifest.py --seed "${seed}" --num-seeds "${NUM_SEEDS}" --num-iterations "${NUM_ITERATIONS}" --out "${JOB_ROOT}/seed_manifest_ppo_e3t_state_aug_predicted_ce_seed${seed}_${JOB_TAG}.json" || true
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo_e3t_official/main.py \
    +experiment=cnn \
    +env=original \
    +NUM_ITERATIONS="${NUM_ITERATIONS}" \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
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
    model.E3T_ACTOR_CONDITION="predicted_partner" \
    model.E3T_ENABLE_CE=True \
    model.E3T_CONDITION_ACTOR=True \
    model.USE_OFFICIAL_E3T_PARTNER=False \
    model.RAND=0.0 \
    model.COPY=0.0
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[large-compare] train_done=$(date -Is) method=ppo_e3t_state_aug_predicted_ce seed=${seed} run_dir=${run_dir}"
  echo "[large-compare] eval_start=$(date -Is) method=ppo_e3t_state_aug_predicted_ce seed=${seed}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" "${PYTHON}" experiments/overcooked_v2_experiments/ppo_e3t_official/utils/visualize_ppo.py \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz
  summarize_csv "ppo_e3t_state_aug_predicted_ce_seed${seed}" "${run_dir}/reward_summary_cross.csv"
  echo "[large-compare] eval_done=$(date -Is) method=ppo_e3t_state_aug_predicted_ce seed=${seed}"
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  print_repro_env
  set_scale
  echo "[large-compare] launch_time=$(date -Is) host=$(hostname) root=${ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[large-compare] seeds=${SEEDS} num_seeds=${NUM_SEEDS} num_iterations=${NUM_ITERATIONS} eval_seeds=${EVAL_SEEDS}"
  for seed in ${SEEDS}; do
    run_ppo_state_aug "${seed}"
    run_e3t_state_aug_predicted "${seed}"
  done
  echo "[large-compare] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_large_compare}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEEDS="${SEEDS:-43 44}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  local log="${JOB_ROOT}/state_aug_e3t_large_compare_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" PREFIX_BASE="${PREFIX_BASE}" LAYOUT="${LAYOUT}" SEEDS="${SEEDS}" EVAL_SEED="${EVAL_SEED}" \
    WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" WANDB_ENTITY="${WANDB_ENTITY}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[large-compare] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) : "${SEEDS:=43 44}" "${EVAL_SEED:=42}" "${PREFIX_BASE:=figure4_large_compare}" "${LAYOUT:=counter_circuit}"; pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[large-compare] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
