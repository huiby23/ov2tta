#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/official_fcp_single_learner}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
SEED="${SEED:-42}"
SEEDS_PER_GROUP="${SEEDS_PER_GROUP:-8}"
EVAL_SEEDS="${EVAL_SEEDS:-100}"
POP_PREFIX="${POP_PREFIX:-official_fcp_single_pop_rnn_sp_${JOB_TAG}}"
POP_STAGE_NAME="${POP_STAGE_NAME:-official_fcp_single_population_${JOB_TAG}}"
POP_STAGE_DIR="runs/${POP_STAGE_NAME}"
POP_DIR="${POP_DIR:-fcp_populations/grounded_coord_simple_single_${JOB_TAG}}"
FCP_PREFIX="${FCP_PREFIX:-official_fcp_single_rnn_fcp_${JOB_TAG}}"
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
    echo "[official-fcp-single] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[official-fcp-single] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[official-fcp-single] not running"
    [[ -f "${PID_FILE}" ]] && echo "[official-fcp-single] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[official-fcp-single] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
  [[ -d "${POP_STAGE_DIR}" ]] && echo "[official-fcp-single] staged_runs=$(find "${POP_STAGE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | wc -l) stage=${POP_STAGE_DIR}"
  [[ -d "${POP_DIR}" ]] && echo "[official-fcp-single] population=${POP_DIR} groups=$(find "${POP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'fcp_*' | wc -l)"
}

summarize_cross() {
  local csv="$1"
  "${PYTHON}" - "${csv}" <<'PY'
import csv, re, sys
from statistics import mean
path = sys.argv[1]
sp=[]; xp=[]
with open(path) as f:
    for r in csv.DictReader(f):
        label = r.get("policy_labels") or r.get("checkpoint") or ""
        m = re.match(r"cross-(\d+)_(\d+)$", label)
        if not m:
            continue
        v = float(r["total_reward"])
        (sp if m.group(1) == m.group(2) else xp).append(v)
print(f"[summary] csv={path}")
print(f"[summary] SP={mean(sp) if sp else float('nan'):.3f} XP={mean(xp) if xp else float('nan'):.3f} n_sp={len(sp)} n_xp={len(xp)}")
PY
}

train_population() {
  echo "[official-fcp-single] population_train_start prefix=${POP_PREFIX} seeds=${SEEDS_PER_GROUP} time=$(date -Is)"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
      +experiment=rnn-sp \
      +env=grounded_coord_simple \
      NUM_SEEDS="${SEEDS_PER_GROUP}" \
      SEED="${SEED}" \
      VISUALIZE=False \
      wandb.ENTITY="${WANDB_ENTITY}" \
      wandb.PROJECT="${WANDB_PROJECT}" \
      wandb.WANDB_MODE="${WANDB_MODE}" \
      +OPTIONAL_PREFIX="${POP_PREFIX}"

  local run_dir
  run_dir="$(latest_run_dir "${POP_PREFIX}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[official-fcp-single] failed to find population run dir for prefix=${POP_PREFIX}" >&2
    exit 1
  fi
  echo "[official-fcp-single] population_train_done run_dir=${run_dir} time=$(date -Is)"

  rm -rf "${POP_STAGE_DIR}"
  mkdir -p "${POP_STAGE_DIR}"
  for i in $(seq 0 $((SEEDS_PER_GROUP - 1))); do
    local src="${run_dir}/run_${i}"
    local dst="${POP_STAGE_DIR}/run_${i}"
    if [[ ! -d "${src}/ckpt_final" ]]; then
      echo "[official-fcp-single] missing checkpoint ${src}/ckpt_final" >&2
      exit 1
    fi
    cp -a "${src}" "${dst}"
  done
  echo "[official-fcp-single] staged_population count=$(find "${POP_STAGE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | wc -l) stage=${POP_STAGE_DIR}"
}

build_population() {
  echo "[official-fcp-single] build_population_start stage=${POP_STAGE_NAME} target=${POP_DIR} time=$(date -Is)"
  rm -rf "fcp_populations/${POP_STAGE_NAME}" "${POP_DIR}"
  bash experiments/copy_fcp.sh "${POP_STAGE_NAME}"
  mv "fcp_populations/${POP_STAGE_NAME}" "${POP_DIR}"
  echo "[official-fcp-single] build_population_done groups=$(find "${POP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'fcp_*' | wc -l) target=${POP_DIR} time=$(date -Is)"
}

train_fcp() {
  echo "[official-fcp-single] fcp_train_start prefix=${FCP_PREFIX} population=${POP_DIR} time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
      +experiment=rnn-fcp \
      +env=grounded_coord_simple \
      NUM_SEEDS=1 \
      SEED="${SEED}" \
      +FCP="${POP_DIR}" \
      VISUALIZE=False \
      wandb.ENTITY="${WANDB_ENTITY}" \
      wandb.PROJECT="${WANDB_PROJECT}" \
      wandb.WANDB_MODE="${WANDB_MODE}" \
      +OPTIONAL_PREFIX="${FCP_PREFIX}"

  local run_dir
  run_dir="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[official-fcp-single] failed to find FCP run dir for prefix=${FCP_PREFIX}" >&2
    exit 1
  fi
  echo "[official-fcp-single] fcp_train_done run_dir=${run_dir} time=$(date -Is)"

  echo "[official-fcp-single] fcp_self_eval_start run_dir=${run_dir} seeds=${EVAL_SEEDS} time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
    "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
      --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${SEED}" --no_viz
  summarize_cross "${run_dir}/reward_summary_cross.csv"

  echo "[official-fcp-single] fcp_population_diag_start run_dir=${run_dir} pop=${POP_STAGE_DIR} seeds=${EVAL_SEEDS} time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" "${PYTHON}" - "${run_dir}" "${POP_STAGE_DIR}" "${EVAL_SEEDS}" <<'PY'
from pathlib import Path
from statistics import mean
import sys
import jax
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.evaluate import eval_pairing

fcp_dir = Path(sys.argv[1])
pop_stage = Path(sys.argv[2])
num_seeds = int(sys.argv[3])
fcp_ckpts, fcp_config = load_all_checkpoints(fcp_dir, final_only=True)
pop_ckpts, pop_config = load_all_checkpoints(pop_stage, final_only=True)

def pol(ckpts, cfg, idx):
    return PPOPolicy(ckpts[f"run_{idx}"]["ckpt_final"].params, cfg)

def score(name, pair):
    out = eval_pairing(pair, "grounded_coord_simple", jax.random.PRNGKey(42), env_kwargs={"agent_view_size": 2, "negative_rewards": True, "random_agent_positions": True, "sample_recipe_on_delivery": True}, num_seeds=num_seeds, no_viz=True)
    vals = [float(v.total_reward) for v in out.values()]
    print(f"[diag] {name} mean={mean(vals):.3f} min={min(vals):.3f} max={max(vals):.3f} nonzero={sum(v > 0 for v in vals)}/{len(vals)}")

score("FCP0+FCP0", PolicyPairing(pol(fcp_ckpts, fcp_config, 0), pol(fcp_ckpts, fcp_config, 0)))
score("FCP0+POP0", PolicyPairing(pol(fcp_ckpts, fcp_config, 0), pol(pop_ckpts, pop_config, 0)))
score("POP0+FCP0", PolicyPairing(pol(pop_ckpts, pop_config, 0), pol(fcp_ckpts, fcp_config, 0)))
score("POP0+POP0", PolicyPairing(pol(pop_ckpts, pop_config, 0), pol(pop_ckpts, pop_config, 0)))
PY
  echo "[official-fcp-single] fcp_eval_done run_dir=${run_dir} time=$(date -Is)"
}

pipeline() {
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  print_repro_env
  echo "[official-fcp-single] pipeline_start tag=${JOB_TAG} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} seeds_per_group=${SEEDS_PER_GROUP} time=$(date -Is)"
  train_population
  build_population
  train_fcp
  echo "[official-fcp-single] pipeline_done time=$(date -Is)"
  rm -f "${PID_FILE}"
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local log="${JOB_ROOT}/official_fcp_single_${JOB_TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${JOB_TAG}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHON="${PYTHON}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[official-fcp-single] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop)
    if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[official-fcp-single] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
