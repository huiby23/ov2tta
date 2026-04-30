#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate myconda
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
  conda activate myconda
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate myconda
fi

ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)}"

LAYOUT="${LAYOUT:-counter_circuit}"
BASE_SEEDS_CSV="${BASE_SEEDS_CSV:-42,43,44}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-3}"
PPO_NUM_CHECKPOINTS="${PPO_NUM_CHECKPOINTS:-1}"
STATE_AUG_ITERATIONS="${STATE_AUG_ITERATIONS:-10}"

EVAL_SEEDS="${EVAL_SEEDS:-500}"
SMOKE_EVAL_SEEDS="${SMOKE_EVAL_SEEDS:-32}"
SEED="${SEED:-42}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
RUN_STAGE4="${RUN_STAGE4:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
RUN_CAUSAL_DIAGNOSTICS="${RUN_CAUSAL_DIAGNOSTICS:-0}"

BASELINE_PREFIX_BASE="${BASELINE_PREFIX_BASE:-figure4_ppo_cnn_state_aug_match_${PIPELINE_TAG}}"
METHOD_PREFIX_BASE="${METHOD_PREFIX_BASE:-figure4_ttappo_state_aug_v1_memory_${PIPELINE_TAG}}"
SMOKE_PREFIX="${SMOKE_PREFIX:-${METHOD_PREFIX_BASE}_smoke}"

SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-262144}"
SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON:-131072}"
SMOKE_NUM_CHECKPOINTS="${SMOKE_NUM_CHECKPOINTS:-1}"
SMOKE_MIN_UPDATE_STEPS="${SMOKE_MIN_UPDATE_STEPS:-4}"
CAUSAL_MODES_CSV="${CAUSAL_MODES_CSV:-memory_off,state_adapt,state_adapt_wrong_partner,state_adapt_random_partner,state_adapt_delayed_partner,state_readout_off}"

TRAIN_OVERRIDES=()
if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}")
fi
if [[ -n "${REW_SHAPING_HORIZON:-}" ]]; then
  TRAIN_OVERRIDES+=("model.REW_SHAPING_HORIZON=${REW_SHAPING_HORIZON}")
fi
if [[ -n "${MODEL_NUM_ENVS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_ENVS=${MODEL_NUM_ENVS}")
fi
if [[ -n "${MODEL_NUM_STEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_STEPS=${MODEL_NUM_STEPS}")
fi
if [[ -n "${MODEL_NUM_MINIBATCHES:-}" ]]; then
  TRAIN_OVERRIDES+=("model.NUM_MINIBATCHES=${MODEL_NUM_MINIBATCHES}")
fi
if [[ -n "${MODEL_UPDATE_EPOCHS:-}" ]]; then
  TRAIN_OVERRIDES+=("model.UPDATE_EPOCHS=${MODEL_UPDATE_EPOCHS}")
fi

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

require_final_checkpoints() {
  local run_dir="$1"
  local expected="$2"
  local count
  count="$(find "${run_dir}" -mindepth 2 -maxdepth 2 -type d -name ckpt_final | wc -l | tr -d ' ')"
  if [[ "${count}" -lt "${expected}" ]]; then
    echo "[checkpoint-check] incomplete final checkpoints in ${run_dir}: found=${count} expected=${expected}"
    return 1
  fi
  echo "[checkpoint-check] final checkpoints OK in ${run_dir}: found=${count}"
}

train_state_aug_baseline() {
  local prefix="$1"
  local seed="$2"
  echo "[stage0] train baseline prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ppo/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${PPO_NUM_CHECKPOINTS}" \
    +NUM_ITERATIONS="${STATE_AUG_ITERATIONS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

train_state_aug_baseline_resumable() {
  local prefix="$1"
  local seed="$2"
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  if [[ -n "${run_dir}" ]] && require_final_checkpoints "${run_dir}" "${NUM_SEEDS}"; then
    echo "[stage0] skip existing complete baseline prefix=${prefix} run_dir=${run_dir}"
    return 0
  fi

  if train_state_aug_baseline "${prefix}" "${seed}"; then
    return 0
  fi

  run_dir="$(latest_run_dir "${prefix}")"
  if [[ -n "${run_dir}" ]] && require_final_checkpoints "${run_dir}" "${NUM_SEEDS}"; then
    echo "[stage0] train command returned non-zero after complete checkpoints; continuing prefix=${prefix}"
    return 0
  fi

  echo "[stage0] train failed before complete checkpoints prefix=${prefix}"
  return 1
}

eval_state_aug_baseline() {
  local run_dir="$1"
  local eval_seeds="$2"
  echo "[stage0] eval baseline run_dir=${run_dir}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${eval_seeds}" \
    --no_viz \
    --seed "${SEED}"
}

train_state_aug_v1_memory() {
  local prefix="$1"
  local seed="$2"
  shift 2
  local extra_overrides=("$@")
  echo "[train] state_aug_v1_memory prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_memory/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    +NUM_ITERATIONS="${STATE_AUG_ITERATIONS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}" \
    "${extra_overrides[@]}"
}

train_state_aug_v1_memory_resumable() {
  local prefix="$1"
  local seed="$2"
  shift 2
  local extra_overrides=("$@")
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  if [[ -n "${run_dir}" ]] && require_final_checkpoints "${run_dir}" "${NUM_SEEDS}"; then
    echo "[train] skip existing complete run prefix=${prefix} run_dir=${run_dir}"
    return 0
  fi

  if train_state_aug_v1_memory "${prefix}" "${seed}" "${extra_overrides[@]}"; then
    return 0
  fi

  run_dir="$(latest_run_dir "${prefix}")"
  if [[ -n "${run_dir}" ]] && require_final_checkpoints "${run_dir}" "${NUM_SEEDS}"; then
    echo "[train] command returned non-zero after complete checkpoints; continuing prefix=${prefix}"
    return 0
  fi

  echo "[train] failed before complete checkpoints prefix=${prefix}"
  return 1
}

eval_method_mode() {
  local run_dir="$1"
  local mode="$2"
  local eval_seeds="$3"
  echo "[eval] mode=${mode} run_dir=${run_dir}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_memory/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${eval_seeds}" \
    --no_viz \
    --seed "${SEED}" \
    --ttt_mode "${mode}" \
    --output_tag "${mode}"
}

run_method_diagnostics() {
  local run_dir="$1"
  local mode="$2"
  echo "[diag] mode=${mode} run_dir=${run_dir}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_memory/utils/partner_diagnostics.py \
    --d "${run_dir}" \
    --num_episodes "${DIAG_EPISODES}" \
    --seed "${SEED}" \
    --mode "${mode}" \
    --output_suffix "${mode}"
}

aggregate_suite() {
  local output_dir="runs/${METHOD_PREFIX_BASE}_summary"
  mkdir -p "${output_dir}"

  "$PYTHON_BIN" - <<'PY' "${ROOT_DIR}" "${BASELINE_PREFIX_BASE}" "${METHOD_PREFIX_BASE}" "${output_dir}"
import csv
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_prefix = sys.argv[2]
method_prefix = sys.argv[3]
output_dir = Path(sys.argv[4])


def parse_reward_csv(path):
    rows = list(csv.DictReader(path.open()))
    sp = []
    xp = []
    for row in rows:
        label = row["policy_labels"]
        reward = float(row["total_reward"])
        pair = label.replace("cross-", "")
        left, right = pair.split("_")
        if left == right:
            sp.append(reward)
        else:
            xp.append(reward)
    return {
        "sp_mean": statistics.mean(sp) if sp else "",
        "xp_mean": statistics.mean(xp) if xp else "",
        "sp_count": len(sp),
        "xp_count": len(xp),
    }


table_rows = []

for reward_csv in sorted(root.glob(f"runs/{baseline_prefix}_r*/**/reward_summary_cross.csv")):
    metrics = parse_reward_csv(reward_csv)
    table_rows.append(
        {
            "family": "ppo_cnn_state_aug",
            "mode": "state_aug",
            "run_dir": str(reward_csv.parent),
            **metrics,
        }
    )

for reward_csv in sorted(root.glob(f"runs/{method_prefix}_r*/**/reward_summary_cross_*.csv")):
    mode = reward_csv.stem.replace("reward_summary_cross_", "")
    metrics = parse_reward_csv(reward_csv)
    diag_csv = reward_csv.parent / f"partner_diagnostics_summary_{mode}.csv"
    diag_acc = ""
    if diag_csv.exists():
        drows = list(csv.DictReader(diag_csv.open()))
        if drows:
            diag_acc = statistics.mean(float(row["partner_pred_acc"]) for row in drows)
    table_rows.append(
        {
            "family": "ttappo_state_aug_v1_memory",
            "mode": mode,
            "run_dir": str(reward_csv.parent),
            "partner_pred_acc": diag_acc,
            **metrics,
        }
    )

summary_csv = output_dir / "suite_summary.csv"
fieldnames = [
    "family",
    "mode",
    "run_dir",
    "sp_mean",
    "xp_mean",
    "sp_count",
    "xp_count",
    "partner_pred_acc",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

aggregate_rows = []
for family, mode in sorted({(row["family"], row["mode"]) for row in table_rows}):
    rows = [row for row in table_rows if row["family"] == family and row["mode"] == mode]
    sp_vals = [float(row["sp_mean"]) for row in rows if row["sp_mean"] != ""]
    xp_vals = [float(row["xp_mean"]) for row in rows if row["xp_mean"] != ""]
    aggregate_rows.append(
        {
            "family": family,
            "mode": mode,
            "num_runs": len(rows),
            "sp_mean": statistics.mean(sp_vals) if sp_vals else "",
            "sp_std": statistics.pstdev(sp_vals) if len(sp_vals) > 1 else 0.0 if sp_vals else "",
            "xp_mean": statistics.mean(xp_vals) if xp_vals else "",
            "xp_std": statistics.pstdev(xp_vals) if len(xp_vals) > 1 else 0.0 if xp_vals else "",
        }
    )

aggregate_csv = output_dir / "suite_aggregate.csv"
with aggregate_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["family", "mode", "num_runs", "sp_mean", "sp_std", "xp_mean", "xp_std"],
    )
    writer.writeheader()
    writer.writerows(aggregate_rows)

print(f"[aggregate] wrote {summary_csv}")
print(f"[aggregate] wrote {aggregate_csv}")
PY
}

IFS=',' read -r -a BASE_SEEDS <<< "${BASE_SEEDS_CSV}"

if [[ "${RUN_STAGE0}" == "1" ]]; then
  idx=1
  for seed in "${BASE_SEEDS[@]}"; do
    prefix="${BASELINE_PREFIX_BASE}_r${idx}"
    train_state_aug_baseline_resumable "${prefix}" "${seed}"
    run_dir="$(latest_run_dir "${prefix}")"
    eval_state_aug_baseline "${run_dir}" "${EVAL_SEEDS}"
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  smoke_num_envs="${MODEL_NUM_ENVS:-64}"
  smoke_num_steps="${MODEL_NUM_STEPS:-256}"
  smoke_update_denom=$((smoke_num_envs * smoke_num_steps))
  smoke_update_steps=$((SMOKE_TOTAL_TIMESTEPS / smoke_update_denom))
  if [[ "${smoke_update_steps}" -lt "${SMOKE_MIN_UPDATE_STEPS}" ]]; then
    echo "[stage1] smoke configuration is too small: SMOKE_TOTAL_TIMESTEPS=${SMOKE_TOTAL_TIMESTEPS} gives ${smoke_update_steps} update steps, require at least ${SMOKE_MIN_UPDATE_STEPS}"
    exit 1
  fi
  train_state_aug_v1_memory_resumable \
    "${SMOKE_PREFIX}" \
    "42" \
    "NUM_CHECKPOINTS=${SMOKE_NUM_CHECKPOINTS}" \
    "model.TOTAL_TIMESTEPS=${SMOKE_TOTAL_TIMESTEPS}" \
    "model.REW_SHAPING_HORIZON=${SMOKE_REW_SHAPING_HORIZON}"
  smoke_run_dir="$(latest_run_dir "${SMOKE_PREFIX}")"
  eval_method_mode "${smoke_run_dir}" "memory_off" "${SMOKE_EVAL_SEEDS}"
  eval_method_mode "${smoke_run_dir}" "state_adapt" "${SMOKE_EVAL_SEEDS}"
  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    run_method_diagnostics "${smoke_run_dir}" "state_adapt"
  fi
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  idx=1
  for seed in "${BASE_SEEDS[@]}"; do
    prefix="${METHOD_PREFIX_BASE}_r${idx}"
    train_state_aug_v1_memory_resumable "${prefix}" "${seed}"
    run_dir="$(latest_run_dir "${prefix}")"
    eval_method_mode "${run_dir}" "memory_off" "${EVAL_SEEDS}"
    eval_method_mode "${run_dir}" "state_adapt" "${EVAL_SEEDS}"
    if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
      run_method_diagnostics "${run_dir}" "state_adapt"
    fi
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  IFS=',' read -r -a CAUSAL_MODES <<< "${CAUSAL_MODES_CSV}"
  idx=1
  for _seed in "${BASE_SEEDS[@]}"; do
    prefix="${METHOD_PREFIX_BASE}_r${idx}"
    run_dir="$(latest_run_dir "${prefix}")"
    if [[ -z "${run_dir}" ]]; then
      echo "[stage3] skip missing run for prefix=${prefix}"
      idx=$((idx + 1))
      continue
    fi
    for mode in "${CAUSAL_MODES[@]}"; do
      eval_method_mode "${run_dir}" "${mode}" "${EVAL_SEEDS}"
      if [[ "${RUN_CAUSAL_DIAGNOSTICS}" == "1" ]]; then
        run_method_diagnostics "${run_dir}" "${mode}"
      fi
    done
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE4}" == "1" ]]; then
  aggregate_suite
fi

echo "[pipeline] done"
