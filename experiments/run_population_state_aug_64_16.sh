#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
source "${ROOT}/experiments/repro_env.sh"

MODE="${1:-${MODE:-full}}"
JOB_TAG="${JOB_TAG:-$(date +%Y%m%d-%H%M%S)}"
JOB_ROOT="${JOB_ROOT:-logs/population_state_aug_64_16}"
mkdir -p "${JOB_ROOT}"

LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-3}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
METHODS="${METHODS:-trajedi_state_aug mep_state_aug}"
PREFIX_BASE="${PREFIX_BASE:-population_state_aug_64_16}"

if [[ "${MODE}" == "smoke" ]]; then
  NUM_SEEDS="${NUM_SEEDS:-2}"
  POPULATION_SIZE="${POPULATION_SIZE:-2}"
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-65536}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-32768}"
  NUM_ENVS="${NUM_ENVS:-8}"
  NUM_STEPS="${NUM_STEPS:-32}"
  UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
  NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
  STATE_AUG_ITERATIONS="${STATE_AUG_ITERATIONS:-2}"
  STATE_AUG_NUM_ROLLOUTS="${STATE_AUG_NUM_ROLLOUTS:-2}"
  STATE_AUG_STEP_SIZE="${STATE_AUG_STEP_SIZE:-20}"
  WANDB_MODE="${WANDB_MODE:-offline}"
else
  NUM_SEEDS="${NUM_SEEDS:-10}"
  POPULATION_SIZE="${POPULATION_SIZE:-5}"
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  NUM_ENVS="${NUM_ENVS:-64}"
  NUM_STEPS="${NUM_STEPS:-256}"
  UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
  NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
  STATE_AUG_ITERATIONS="${STATE_AUG_ITERATIONS:-10}"
  STATE_AUG_NUM_ROLLOUTS="${STATE_AUG_NUM_ROLLOUTS:-10}"
  STATE_AUG_STEP_SIZE="${STATE_AUG_STEP_SIZE:-10}"
fi

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

summarize_csv() {
  local name="$1"
  local csv="$2"
  if [[ ! -f "${csv}" ]]; then
    echo "[population-sa] missing_csv name=${name} csv=${csv}" >&2
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
print(f"[population-sa] result name={name} SP={sum(sp)/len(sp):.4f} XP={sum(xp)/len(xp):.4f} SP_pair_std={statistics.pstdev(sp) if len(sp)>1 else 0.0:.4f} XP_pair_std={statistics.pstdev(xp) if len(xp)>1 else 0.0:.4f} csv={path}")
PY
}

evaluate_run() {
  local run_dir="$1"
  echo "[population-sa] eval_start=$(date -Is) run_dir=${run_dir}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz
  summarize_csv "$(basename "$(dirname "${run_dir}")")" "${run_dir}/reward_summary_cross.csv"
}

train_mep_state_aug() {
  local prefix="${PREFIX_BASE}_mep_state_aug_mm${MEP_GROUP_MM:-1}_mp${MEP_GROUP_MP:-1}_K${POPULATION_SIZE}_ent${MEP_ENT_COEF:-0.1}_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_sa${STATE_AUG_ITERATIONS}_${JOB_TAG}"
  echo "[population-sa] train_start=$(date -Is) method=mep_state_aug prefix=${prefix}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.mep.main \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +NUM_ITERATIONS="${STATE_AUG_ITERATIONS}" \
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
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    MEP.POPULATION_SIZE="${POPULATION_SIZE}" \
    MEP.ENT_COEF="${MEP_ENT_COEF:-0.1}" \
    MEP.GROUP_MM="${MEP_GROUP_MM:-1}" \
    MEP.GROUP_MP="${MEP_GROUP_MP:-1}" \
    MEP.STATE_AUG_INCLUDE_PARTNERS="${STATE_AUG_INCLUDE_PARTNERS:-True}" \
    MEP.STATE_AUG_NUM_ROLLOUTS="${STATE_AUG_NUM_ROLLOUTS}" \
    MEP.STATE_AUG_STEP_SIZE="${STATE_AUG_STEP_SIZE}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[population-sa] train_done=$(date -Is) method=mep_state_aug run_dir=${run_dir}"
  evaluate_run "${run_dir}"
}

train_trajedi_state_aug() {
  local prefix="${PREFIX_BASE}_trajedi_state_aug_mm${TRAJEDI_GROUP_MM:-1}_mp${TRAJEDI_GROUP_MP:-1}_K${POPULATION_SIZE}_div${TRAJEDI_DIV_WEIGHT:-0.1}_${NUM_ENVS}_${NUM_MINIBATCHES}_${TOTAL_TIMESTEPS}_sa${STATE_AUG_ITERATIONS}_${JOB_TAG}"
  echo "[population-sa] train_start=$(date -Is) method=trajedi_state_aug prefix=${prefix}"
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  "${PYTHON}" -m overcooked_v2_experiments.trajedi.main \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +NUM_ITERATIONS="${STATE_AUG_ITERATIONS}" \
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
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    TRAJEDI.POPULATION_SIZE="${POPULATION_SIZE}" \
    TRAJEDI.DIV_WEIGHT="${TRAJEDI_DIV_WEIGHT:-0.1}" \
    TRAJEDI.GROUP_MM="${TRAJEDI_GROUP_MM:-1}" \
    TRAJEDI.GROUP_MP="${TRAJEDI_GROUP_MP:-1}" \
    TRAJEDI.STATE_AUG_INCLUDE_PARTNERS="${STATE_AUG_INCLUDE_PARTNERS:-True}" \
    TRAJEDI.STATE_AUG_NUM_ROLLOUTS="${STATE_AUG_NUM_ROLLOUTS}" \
    TRAJEDI.STATE_AUG_STEP_SIZE="${STATE_AUG_STEP_SIZE}"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  echo "[population-sa] train_done=$(date -Is) method=trajedi_state_aug run_dir=${run_dir}"
  evaluate_run "${run_dir}"
}

print_repro_env
echo "[population-sa] mode=${MODE} methods=${METHODS} layout=${LAYOUT} seed=${SEED} num_seeds=${NUM_SEEDS} pop=${POPULATION_SIZE} total=${TOTAL_TIMESTEPS} envs=${NUM_ENVS} steps=${NUM_STEPS} minibatches=${NUM_MINIBATCHES} sa_iters=${STATE_AUG_ITERATIONS}"

for method in ${METHODS}; do
  case "${method}" in
    mep_state_aug) train_mep_state_aug ;;
    trajedi_state_aug) train_trajedi_state_aug ;;
    *) echo "[population-sa] unknown method=${method}" >&2; exit 2 ;;
  esac
done

echo "[population-sa] done $(date -Is)"
