#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOB_ROOT="${JOB_ROOT:-logs/overnight_e3t_evidence_suite}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
mkdir -p "$JOB_ROOT"
source "$ROOT/experiments/repro_env.sh"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1
}

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

set_scale() {
  if [[ "$MODE" == "smoke" ]]; then
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
  elif [[ "$MODE" == "full" ]]; then
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-15000000}"
    NUM_ENVS="${NUM_ENVS:-128}"
    NUM_STEPS="${NUM_STEPS:-256}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-8}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
    NUM_SEEDS="${NUM_SEEDS:-10}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
    EVAL_SEEDS="${EVAL_SEEDS:-500}"
  else
    echo "[overnight-e3t] unknown MODE=$MODE; use smoke or full" >&2
    exit 2
  fi
}

write_seed_manifest() {
  local method="$1"
  local iterations="$2"
  "$PYTHON" experiments/tools/seed_manifest.py \
    --seed "$SEED" \
    --num-seeds "$NUM_SEEDS" \
    --num-iterations "$iterations" \
    --out "$JOB_ROOT/seed_manifest_${method}_${JOB_TAG}.json" || true
}

summarize_csv() {
  local name="$1"
  local csv_path="$2"
  if [[ ! -f "$csv_path" ]]; then
    echo "[overnight-e3t] missing_csv name=$name csv=$csv_path" >&2
    return 1
  fi
  "$PYTHON" - <<PY
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
print(f"[overnight-e3t] result name={name} SP={sp_mean:.4f} XP={xp_mean:.4f} SP_pair_std={sp_std:.4f} XP_pair_std={xp_std:.4f} csv={path}")
PY
}

eval_e3t() {
  local name="$1"
  local run_dir="$2"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "[overnight-e3t] missing_run_dir name=$name run_dir=$run_dir" >&2
    exit 2
  fi
  echo "[overnight-e3t] eval_start=$(date -Is) method=$name run_dir=$run_dir eval_seeds=$EVAL_SEEDS"
  PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "$PYTHON" experiments/overcooked_v2_experiments/ppo_e3t_official/utils/visualize_ppo.py \
    --d "$run_dir" \
    --cross \
    --num_seeds "$EVAL_SEEDS" \
    --seed "$EVAL_SEED" \
    --no_viz
  summarize_csv "$name" "$run_dir/reward_summary_cross.csv"
  echo "[overnight-e3t] eval_done=$(date -Is) method=$name"
}

train_e3t_variant() {
  local method="$1"
  local state_aug="$2"
  local enable_ce="$3"
  local condition_actor="$4"
  local actor_condition="$5"

  local prefix="${PREFIX_BASE}_${method}_${JOB_TAG}"
  local iterations="1"
  local iteration_arg=()
  if [[ "$state_aug" == "true" ]]; then
    iterations="$NUM_ITERATIONS"
    iteration_arg=(+NUM_ITERATIONS="$NUM_ITERATIONS")
  fi

  echo "[overnight-e3t] train_start=$(date -Is) method=$method prefix=$prefix state_aug=$state_aug ce=$enable_ce condition_actor=$condition_actor actor_condition=$actor_condition"
  write_seed_manifest "$method" "$iterations"

  PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "$PYTHON" experiments/overcooked_v2_experiments/ppo_e3t_official/main.py \
    +env=default \
    model=cnn \
    "${iteration_arg[@]}" \
    +env.ENV_KWARGS.layout="$LAYOUT" \
    SEED="$SEED" \
    NUM_SEEDS="$NUM_SEEDS" \
    NUM_CHECKPOINTS="$NUM_CHECKPOINTS" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="$prefix" \
    wandb.WANDB_MODE="$WANDB_MODE" \
    wandb.PROJECT="$WANDB_PROJECT" \
    wandb.ENTITY="$WANDB_ENTITY" \
    model.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    model.REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    model.NUM_ENVS="$NUM_ENVS" \
    model.NUM_STEPS="$NUM_STEPS" \
    model.UPDATE_EPOCHS="$UPDATE_EPOCHS" \
    model.CONTEXT_UPDATE_EPOCHS="$CONTEXT_UPDATE_EPOCHS" \
    model.NUM_MINIBATCHES="$NUM_MINIBATCHES" \
    model.E3T_PARTNER_SOURCE="main_policy" \
    model.E3T_ACTOR_CONDITION="$actor_condition" \
    model.E3T_ENABLE_CE="$enable_ce" \
    model.E3T_CONDITION_ACTOR="$condition_actor" \
    model.USE_OFFICIAL_E3T_PARTNER=False \
    model.RAND=0.0 \
    model.COPY=0.0

  local run_dir
  run_dir="$(latest_run_dir "$prefix")"
  echo "[overnight-e3t] train_done=$(date -Is) method=$method run_dir=$run_dir"
  printf "%s,%s,%s\n" "$method" "$state_aug" "$run_dir" >> "$JOB_ROOT/run_dirs_${JOB_TAG}.csv"
  eval_e3t "$method" "$run_dir"
}

run_method() {
  case "$1" in
    e3t_aux_only_state_aug)
      train_e3t_variant "e3t_aux_only_state_aug" true True False predicted_partner ;;
    e3t_no_ce_no_condition_state_aug)
      train_e3t_variant "e3t_no_ce_no_condition_state_aug" true False False predicted_partner ;;
    e3t_no_ce_conditioned_state_aug)
      train_e3t_variant "e3t_no_ce_conditioned_state_aug" true False True predicted_partner ;;
    e3t_constant_ce_state_aug)
      train_e3t_variant "e3t_constant_ce_state_aug" true True True constant ;;
    e3t_aux_only_no_state_aug_full_obs)
      train_e3t_variant "e3t_aux_only_no_state_aug_full_obs" false True False predicted_partner ;;
    e3t_no_ce_no_condition_no_state_aug_full_obs)
      train_e3t_variant "e3t_no_ce_no_condition_no_state_aug_full_obs" false False False predicted_partner ;;
    e3t_constant_ce_no_state_aug_full_obs)
      train_e3t_variant "e3t_constant_ce_no_state_aug_full_obs" false True True constant ;;
    *) echo "[overnight-e3t] unknown method=$1" >&2; exit 2 ;;
  esac
}

post_summary() {
  local out="runs/overnight_e3t_evidence_${JOB_TAG}"
  mkdir -p "$out"
  cp "$JOB_ROOT/run_dirs_${JOB_TAG}.csv" "$out/run_dirs.csv" 2>/dev/null || true
  grep "\[overnight-e3t\] result" "$LOG_FILE" > "$out/results.log" 2>/dev/null || true
  "$PYTHON" - <<PY
import re, csv
from pathlib import Path
log = Path("$LOG_FILE")
out = Path("$out")
rows = []
if log.exists():
    pat = re.compile(r"result name=(\S+) SP=([0-9.\-nan]+) XP=([0-9.\-nan]+) SP_pair_std=([0-9.\-nan]+) XP_pair_std=([0-9.\-nan]+) csv=(.*)")
    for line in log.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            rows.append({"method": m.group(1), "SP": m.group(2), "XP": m.group(3), "SP_pair_std": m.group(4), "XP_pair_std": m.group(5), "csv": m.group(6)})
with (out / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "SP", "XP", "SP_pair_std", "XP_pair_std", "csv"])
    w.writeheader(); w.writerows(rows)
print(f"[overnight-e3t] wrote_summary {out / 'summary.csv'} rows={len(rows)}")
PY
}

pipeline() {
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_overnight_e3t_evidence}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-e3t_aux_only_state_aug e3t_no_ce_no_condition_state_aug e3t_no_ce_conditioned_state_aug e3t_constant_ce_state_aug e3t_aux_only_no_state_aug_full_obs e3t_no_ce_no_condition_no_state_aug_full_obs e3t_constant_ce_no_state_aug_full_obs}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export EVAL_SEED="${EVAL_SEED:-42}"
  export PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  print_repro_env
  set_scale
  echo "method,state_aug,run_dir" > "$JOB_ROOT/run_dirs_${JOB_TAG}.csv"
  echo "[overnight-e3t] launch_time=$(date -Is) host=$(hostname) root=$ROOT cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[overnight-e3t] mode=$MODE layout=$LAYOUT seed=$SEED methods=$METHODS"
  echo "[overnight-e3t] scale total=$TOTAL_TIMESTEPS rew_horizon=$REW_SHAPING_HORIZON num_seeds=$NUM_SEEDS iterations=$NUM_ITERATIONS envs=$NUM_ENVS steps=$NUM_STEPS epochs=$UPDATE_EPOCHS minibatches=$NUM_MINIBATCHES ce_epochs=$CONTEXT_UPDATE_EPOCHS eval_seeds=$EVAL_SEEDS"
  for method in $METHODS; do
    run_method "$method"
  done
  post_summary
  echo "[overnight-e3t] pipeline_done=$(date -Is)"
  rm -f "$PID_FILE"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "[overnight-e3t] running pid=$pid"
    ps -fp "$pid" || true
    echo "[overnight-e3t] children:"
    pgrep -P "$pid" -af || true
  else
    echo "[overnight-e3t] not_running"
    [[ -f "$PID_FILE" ]] && echo "[overnight-e3t] stale_pid=$(cat "$PID_FILE")"
  fi
  if [[ -f "$LATEST_LOG" ]]; then
    local log
    log="$(cat "$LATEST_LOG")"
    echo "[overnight-e3t] latest_log=$log"
    [[ -f "$log" ]] && tail -n 100 "$log"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  export JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
  export MODE="${MODE:-full}"
  export PREFIX_BASE="${PREFIX_BASE:-figure4_overnight_e3t_evidence}"
  export LAYOUT="${LAYOUT:-counter_circuit}"
  export SEED="${SEED:-42}"
  export METHODS="${METHODS:-e3t_aux_only_state_aug e3t_no_ce_no_condition_state_aug e3t_no_ce_conditioned_state_aug e3t_constant_ce_state_aug e3t_aux_only_no_state_aug_full_obs e3t_no_ce_no_condition_no_state_aug_full_obs e3t_constant_ce_no_state_aug_full_obs}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
  export WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  local log="$JOB_ROOT/overnight_e3t_evidence_${JOB_TAG}.log"
  printf "%s\n" "$log" > "$LATEST_LOG"
  nohup env JOB_TAG="$JOB_TAG" MODE="$MODE" PREFIX_BASE="$PREFIX_BASE" LAYOUT="$LAYOUT" \
    SEED="$SEED" METHODS="$METHODS" WANDB_MODE="$WANDB_MODE" WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHON="$PYTHON" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}" REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-}" \
    NUM_ENVS="${NUM_ENVS:-}" NUM_STEPS="${NUM_STEPS:-}" UPDATE_EPOCHS="${UPDATE_EPOCHS:-}" \
    CONTEXT_UPDATE_EPOCHS="${CONTEXT_UPDATE_EPOCHS:-}" NUM_MINIBATCHES="${NUM_MINIBATCHES:-}" \
    NUM_SEEDS="${NUM_SEEDS:-}" NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-}" NUM_ITERATIONS="${NUM_ITERATIONS:-}" \
    EVAL_SEEDS="${EVAL_SEEDS:-}" EVAL_SEED="${EVAL_SEED:-42}" JOB_ROOT="$JOB_ROOT" \
    LOG_FILE="$log" bash "$BASH_SOURCE" pipeline > "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "[overnight-e3t] started pid=$pid log=$log"
}

case "${1:-start}" in
  start) start ;;
  pipeline)
    : "${MODE:=full}" "${PREFIX_BASE:=figure4_overnight_e3t_evidence}" "${LAYOUT:=counter_circuit}" "${SEED:=42}"
    : "${METHODS:=e3t_aux_only_state_aug e3t_no_ce_no_condition_state_aug e3t_no_ce_conditioned_state_aug e3t_constant_ce_state_aug e3t_aux_only_no_state_aug_full_obs e3t_no_ce_no_condition_no_state_aug_full_obs e3t_constant_ce_no_state_aug_full_obs}"
    : "${WANDB_MODE:=online}" "${WANDB_PROJECT:=ov2-paper-repro}" "${WANDB_ENTITY:=huiby_tsinghua23}" "${EVAL_SEED:=42}"
    : "${LOG_FILE:=${JOB_ROOT}/overnight_e3t_evidence_${JOB_TAG}.log}"
    pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "$LATEST_LOG")" ;;
  stop) if is_running; then kill "$(cat "$PID_FILE")"; else echo "[overnight-e3t] not_running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
