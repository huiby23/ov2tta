#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

JOB_ROOT="${JOB_ROOT:-logs/official_fcp_64_16_30m_single}"
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
    echo "[official-fcp-30m-single] missing_csv name=${name} csv=${csv_path}" >&2
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
print(f"[official-fcp-30m-single] result name={name} SP={sum(sp)/len(sp) if sp else math.nan:.4f} XP={sum(xp)/len(xp) if xp else math.nan:.4f} n_sp_pairs={len(sp)} n_xp_pairs={len(xp)} csv={path}")
PY
}

prepare_single_population() {
  local src_group="${SOURCE_FCP_POP}/fcp_${SOURCE_FCP_GROUP}"
  if [[ ! -d "${src_group}" ]]; then
    echo "[official-fcp-30m-single] missing source group ${src_group}" >&2
    exit 1
  fi
  rm -rf "${SINGLE_POP_DIR}"
  mkdir -p "${SINGLE_POP_DIR}"
  cp -a "${src_group}" "${SINGLE_POP_DIR}/fcp_1"
  echo "[official-fcp-30m-single] single_population_ready src=${src_group} dst=${SINGLE_POP_DIR}/fcp_1"
}

train_fcp() {
  local existing
  existing="$(latest_run_dir "${FCP_PREFIX}")"
  if [[ -n "${existing}" && -d "${existing}/run_0/ckpt_final" ]]; then
    echo "[official-fcp-30m-single] fcp_reuse run_dir=${existing}"
    printf "%s\n" "${existing}" > "${JOB_ROOT}/fcp.run_dir"
    return
  fi
  echo "[official-fcp-30m-single] fcp_train_start=$(date -Is) prefix=${FCP_PREFIX} pop=${SINGLE_POP_DIR}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-fcp \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    +FCP="${SINGLE_POP_DIR}" \
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
  if [[ -z "${run_dir}" || ! -d "${run_dir}/run_0/ckpt_final" ]]; then
    echo "[official-fcp-30m-single] fcp_missing_checkpoint run_dir=${run_dir}" >&2
    exit 1
  fi
  printf "%s\n" "${run_dir}" > "${JOB_ROOT}/fcp.run_dir"
  echo "[official-fcp-30m-single] fcp_train_done=$(date -Is) run_dir=${run_dir}"
}

evaluate_self() {
  local run_dir
  run_dir="$(cat "${JOB_ROOT}/fcp.run_dir")"
  echo "[official-fcp-30m-single] eval_self_start=$(date -Is) run_dir=${run_dir} eval_seeds=${EVAL_SEEDS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz
  summarize_cross "fcp_rnn_30m_single_64_16_self" "${run_dir}/reward_summary_cross.csv"
  echo "[official-fcp-30m-single] eval_self_done=$(date -Is) run_dir=${run_dir}"
}

evaluate_vs_population() {
  local run_dir out_csv
  run_dir="$(cat "${JOB_ROOT}/fcp.run_dir")"
  out_csv="${run_dir}/fcp_vs_source_population.csv"
  echo "[official-fcp-30m-single] eval_pop_start=$(date -Is) run_dir=${run_dir} source=${SOURCE_POP_RUN_DIR}"
  PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  "${PYTHON}" - "${run_dir}" "${SOURCE_POP_RUN_DIR}" "${out_csv}" "${EVAL_SEEDS_POP}" <<'PY'
from pathlib import Path
from statistics import mean, pstdev
import csv, sys, jax
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.evaluate import eval_pairing

fcp_dir=Path(sys.argv[1]); pop_run=Path(sys.argv[2]); out_csv=Path(sys.argv[3]); n=int(sys.argv[4])
fcp_ck,fcp_cfg=load_all_checkpoints(fcp_dir, final_only=True)
pop_ck,pop_cfg=load_all_checkpoints(pop_run, final_only=True)

def pol(ck,cfg,i): return PPOPolicy(ck[f"run_{i}"]["ckpt_final"].params, cfg)

def score(label, a, b, seed):
    out=eval_pairing(PolicyPairing(a,b), "counter_circuit", jax.random.PRNGKey(seed), env_kwargs={}, num_seeds=n, no_viz=True)
    vals=[float(v.total_reward) for v in out.values()]
    row={"pair":label,"mean":mean(vals),"std":pstdev(vals),"min":min(vals),"max":max(vals),"nonzero":sum(v!=0 for v in vals),"n":len(vals)}
    print(f"[official-fcp-30m-single] pop_result {label} mean={row['mean']:.3f} std={row['std']:.3f} min={row['min']:.1f} max={row['max']:.1f} nonzero={row['nonzero']}/{row['n']}", flush=True)
    return row

fcp0=pol(fcp_ck,fcp_cfg,0)
rows=[]
rows.append(score("FCP0+FCP0", fcp0, fcp0, 42))
for i in range(8):
    pi=pol(pop_ck,pop_cfg,i)
    rows.append(score(f"FCP0+POP{i}", fcp0, pi, 1000+i))
    rows.append(score(f"POP{i}+FCP0", pi, fcp0, 2000+i))
with out_csv.open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=["pair","mean","std","min","max","nonzero","n"])
    w.writeheader(); w.writerows(rows)
xp=[r["mean"] for r in rows if r["pair"] != "FCP0+FCP0"]
print(f"[official-fcp-30m-single] pop_summary self={rows[0]['mean']:.3f} xp_vs_pop={mean(xp):.3f} pairs={len(xp)} csv={out_csv}", flush=True)
PY
  echo "[official-fcp-30m-single] eval_pop_done=$(date -Is) run_dir=${run_dir}"
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}"
  export REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-15000000}"
  export NUM_ENVS="${NUM_ENVS:-64}"
  export NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  export EVAL_SEEDS="${EVAL_SEEDS:-500}"
  export EVAL_SEEDS_POP="${EVAL_SEEDS_POP:-100}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export SOURCE_FCP_POP="${SOURCE_FCP_POP:-fcp_populations/official_fcp64_16_aligned_sp80_20260508-101638/20260508-101651_tc1wpl5z_counter_circuit_avs-full}"
  export SOURCE_FCP_GROUP="${SOURCE_FCP_GROUP:-1}"
  export SOURCE_POP_RUN_DIR="${SOURCE_POP_RUN_DIR:-runs/official_fcp64_16_aligned_sp80_20260508-101638/20260508-101651_tc1wpl5z_counter_circuit_avs-full}"
  export SINGLE_POP_DIR="${SINGLE_POP_DIR:-fcp_populations/official_fcp64_16_30m_single_${JOB_TAG}}"
  export FCP_PREFIX="${FCP_PREFIX:-official_fcp64_16_30m_single_${JOB_TAG}}"
  export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

  print_repro_env
  echo "[official-fcp-30m-single] pipeline_start=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[official-fcp-30m-single] layout=${LAYOUT} seed=${SEED} total=${TOTAL_TIMESTEPS} rew_horizon=${REW_SHAPING_HORIZON} envs=${NUM_ENVS} minibatches=${NUM_MINIBATCHES}"
  echo "[official-fcp-30m-single] source_pop=${SOURCE_FCP_POP} source_group=fcp_${SOURCE_FCP_GROUP} source_run=${SOURCE_POP_RUN_DIR}"
  prepare_single_population
  train_fcp
  evaluate_self
  evaluate_vs_population
  echo "[official-fcp-30m-single] pipeline_done=$(date -Is)"
  rm -f "${PID_FILE}"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[official-fcp-30m-single] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[official-fcp-30m-single] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[official-fcp-30m-single] not running"
    [[ -f "${PID_FILE}" ]] && echo "[official-fcp-30m-single] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[official-fcp-30m-single] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 180 "${log}"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local tag="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  local log="${JOB_ROOT}/official_fcp_30m_single_${tag}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env JOB_TAG="${tag}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHON="${PYTHON}" \
    LAYOUT="${LAYOUT:-}" SEED="${SEED:-}" TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" \
    NUM_ENVS="${NUM_ENVS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEEDS_POP="${EVAL_SEEDS_POP:-}" EVAL_SEED="${EVAL_SEED:-}" \
    WANDB_MODE="${WANDB_MODE:-}" WANDB_PROJECT="${WANDB_PROJECT:-}" WANDB_ENTITY="${WANDB_ENTITY:-}" \
    SOURCE_FCP_POP="${SOURCE_FCP_POP:-}" SOURCE_FCP_GROUP="${SOURCE_FCP_GROUP:-}" SOURCE_POP_RUN_DIR="${SOURCE_POP_RUN_DIR:-}" SINGLE_POP_DIR="${SINGLE_POP_DIR:-}" FCP_PREFIX="${FCP_PREFIX:-}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[official-fcp-30m-single] started pid=${pid} log=${log}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[official-fcp-30m-single] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
