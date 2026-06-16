#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

MODE="${MODE:-smoke}"  # smoke or full
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LAYOUT="${LAYOUT:-counter_circuit}"
TEACHER_METHOD="${TEACHER_METHOD:-MEP ent=0.05 standard}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-}"
TEACHER_BACKEND="${TEACHER_BACKEND:-}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-10}"
RUN_PPO_FINETUNE="${RUN_PPO_FINETUNE:-1}"
RUN_NO_HISTORY="${RUN_NO_HISTORY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${MODE}" == "smoke" ]]; then
  DATA_EPISODES="${DATA_EPISODES:-2}"
  MAX_PARTNERS="${MAX_PARTNERS:-2}"
  DISTILL_STEPS="${DISTILL_STEPS:-20}"
  DISTILL_BATCH="${DISTILL_BATCH:-2}"
  DISTILL_CHUNK="${DISTILL_CHUNK:-40}"
  TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-262144}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-131072}"
  EVAL_SEEDS="${EVAL_SEEDS:-8}"
  NUM_SEEDS_EFFECTIVE="${SMOKE_NUM_SEEDS:-1}"
  RUN_PPO_FINETUNE="${SMOKE_RUN_PPO_FINETUNE:-0}"
else
  DATA_EPISODES="${DATA_EPISODES:-512}"
  MAX_PARTNERS="${MAX_PARTNERS:-}"
  DISTILL_STEPS="${DISTILL_STEPS:-3000}"
  DISTILL_BATCH="${DISTILL_BATCH:-32}"
  DISTILL_CHUNK="${DISTILL_CHUNK:-100}"
  TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-10000000}"
  REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
  EVAL_SEEDS="${EVAL_SEEDS:-500}"
  NUM_SEEDS_EFFECTIVE="${NUM_SEEDS}"
fi

REPORT_DIR="reports/acp_${RUN_ID}"
mkdir -p "${REPORT_DIR}"
MANIFEST="${REPORT_DIR}/teacher_manifest.json"
DATASET="${REPORT_DIR}/teacher_dataset.npz"
DISTILL_DIR="runs_by_config/counter_circuit_full_obs_64env_16mb_10M/acp/${RUN_ID}_distill_only"
NOHIST_DIR="runs_by_config/counter_circuit_full_obs_64env_16mb_10M/acp/${RUN_ID}_no_history"
FINETUNE_PREFIX="acp_ppo_finetune_64env_16mb_10M_seed${SEED}_${RUN_ID}"

log() { echo "[$(date +%F' '%T)] $*"; }

log "ACP pipeline mode=${MODE} teacher=${TEACHER_METHOD} run_id=${RUN_ID}"
BUILD_ARGS=(--method "${TEACHER_METHOD}" --output "${MANIFEST}")
if [[ -n "${TEACHER_RUN_DIR}" ]]; then BUILD_ARGS+=(--run-dir "${TEACHER_RUN_DIR}"); fi
if [[ -n "${TEACHER_BACKEND}" ]]; then BUILD_ARGS+=(--backend "${TEACHER_BACKEND}"); fi
"${PYTHON}" -m overcooked_v2_experiments.acp.build_teacher_manifest "${BUILD_ARGS[@]}"

COLLECT_ARGS=(--manifest "${MANIFEST}" --output "${DATASET}" --num-episodes "${DATA_EPISODES}" --seed "${SEED}" --greedy)
if [[ -n "${MAX_PARTNERS}" ]]; then COLLECT_ARGS+=(--max-partners "${MAX_PARTNERS}"); fi
"${PYTHON}" -m overcooked_v2_experiments.acp.collect_acp_teacher_dataset "${COLLECT_ARGS[@]}"

log "Stage2 distill-only with partner history"
"${PYTHON}" -m overcooked_v2_experiments.acp.train_distill \
  --dataset "${DATASET}" \
  --output-dir "${DISTILL_DIR}" \
  --seed "${SEED}" \
  --num-seeds "${NUM_SEEDS_EFFECTIVE}" \
  --steps "${DISTILL_STEPS}" \
  --batch-size "${DISTILL_BATCH}" \
  --chunk-length "${DISTILL_CHUNK}" \
  --use-partner-history

if [[ "${RUN_NO_HISTORY}" == "1" ]]; then
  log "Stage2 ablation: no partner history"
  "${PYTHON}" -m overcooked_v2_experiments.acp.train_distill \
    --dataset "${DATASET}" \
    --output-dir "${NOHIST_DIR}" \
    --seed "${SEED}" \
    --num-seeds "${NUM_SEEDS_EFFECTIVE}" \
    --steps "${DISTILL_STEPS}" \
    --batch-size "${DISTILL_BATCH}" \
    --chunk-length "${DISTILL_CHUNK}" \
    --no-use-partner-history
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  log "Eval distill-only online latent"
  "${PYTHON}" -m overcooked_v2_experiments.acp.utils.visualize_ppo --d "${DISTILL_DIR}" --all --num_seeds "${EVAL_SEEDS}" --no_viz --seed 42 --ttt_mode state_adapt --output_tag distill_online_latent
  if [[ "${RUN_NO_HISTORY}" == "1" ]]; then
    log "Eval no-history ablation"
    "${PYTHON}" -m overcooked_v2_experiments.acp.utils.visualize_ppo --d "${NOHIST_DIR}" --all --num_seeds "${EVAL_SEEDS}" --no_viz --seed 42 --ttt_mode state_adapt --output_tag no_history
  fi
fi

if [[ "${RUN_PPO_FINETUNE}" == "1" ]]; then
  log "Stage3 PPO finetune from distillation checkpoint"
  PYTHONUNBUFFERED=1 "${PYTHON}" -m overcooked_v2_experiments.acp.main \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS_EFFECTIVE}" \
    NUM_CHECKPOINTS=3 \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${FINETUNE_PREFIX}" \
    +PRETRAINED_ACP_RUN_DIR="${DISTILL_DIR}" \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    model.TOTAL_TIMESTEPS="${TRAIN_TIMESTEPS}" \
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}" \
    model.NUM_ENVS=64 \
    model.NUM_STEPS=256 \
    model.UPDATE_EPOCHS=4 \
    model.NUM_MINIBATCHES=16

  FINETUNE_DIR="$(find "runs/${FINETUNE_PREFIX}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  echo "${FINETUNE_DIR}" > "${REPORT_DIR}/finetune_run_dir.txt"
  log "Eval finetune memory_off"
  "${PYTHON}" -m overcooked_v2_experiments.acp.utils.visualize_ppo --d "${FINETUNE_DIR}" --all --num_seeds "${EVAL_SEEDS}" --no_viz --seed 42 --ttt_mode memory_off --output_tag memory_off
  log "Eval finetune online_latent"
  "${PYTHON}" -m overcooked_v2_experiments.acp.utils.visualize_ppo --d "${FINETUNE_DIR}" --all --num_seeds "${EVAL_SEEDS}" --no_viz --seed 42 --ttt_mode state_adapt --output_tag online_latent
fi

log "ACP pipeline complete. report_dir=${REPORT_DIR} distill_dir=${DISTILL_DIR}"
