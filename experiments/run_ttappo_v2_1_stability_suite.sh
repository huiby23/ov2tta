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
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-3}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"
DIAG_WORKERS="${DIAG_WORKERS:-2}"
LAYOUT="${LAYOUT:-counter_circuit}"
BASE_SEEDS_CSV="${BASE_SEEDS_CSV:-42,43,44}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)}"
PREFIX_BASE="${PREFIX_BASE:-figure4_ttappo_v2_1_${PIPELINE_TAG}}"
RUN_DETERMINISM_CHECK="${RUN_DETERMINISM_CHECK:-0}"
RUN_V3="${RUN_V3:-0}"
V3_PACKAGE="${V3_PACKAGE:-}"
V3_PREFIX="${V3_PREFIX:-figure4_ttappo_v3_${PIPELINE_TAG}}"

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

train_variant() {
  local prefix="$1"
  local seed="$2"
  local mode="$3"
  echo "[pipeline] train v2.1 mode=${mode} prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v2_1/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    model.ACTOR_TEMPORAL_MODE="${mode}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

run_noadapt_eval() {
  local run_dir="$1"
  local tag="$2"
  echo "[pipeline] no_adapt eval run_dir=${run_dir} tag=${tag}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v2_1/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz \
    --seed 42 \
    --ttt_mode no_adapt \
    --output_tag "${tag}"
}

run_partner_diagnostics() {
  local run_dir="$1"
  echo "[pipeline] partner diagnostics run_dir=${run_dir}"
  if [[ "${DIAG_WORKERS}" -le 1 ]]; then
    PYTHONPATH=experiments "$PYTHON_BIN" \
      experiments/overcooked_v2_experiments/ttappo_v2_1/utils/partner_diagnostics.py \
      --d "${run_dir}" \
      --num_episodes "${DIAG_EPISODES}" \
      --seed 42
    return
  fi

  mapfile -t CHUNKS < <("$PYTHON_BIN" - <<'PY' "${run_dir}" "${DIAG_WORKERS}"
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
num_workers = max(int(sys.argv[2]), 1)
run_keys = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
if not run_keys:
    sys.exit(0)
chunk_size = math.ceil(len(run_keys) / min(num_workers, len(run_keys)))
for idx in range(0, len(run_keys), chunk_size):
    print(",".join(run_keys[idx: idx + chunk_size]))
PY
)

  local pids=()
  local worker_idx=0
  for chunk in "${CHUNKS[@]}"; do
    [[ -z "${chunk}" ]] && continue
    local gpu_idx=$((worker_idx % 2))
    local suffix="worker_${worker_idx}"
    echo "[pipeline] diagnostics worker=${suffix} gpu=${gpu_idx} runs=${chunk}"
    CUDA_VISIBLE_DEVICES="${gpu_idx}" PYTHONPATH=experiments "$PYTHON_BIN" \
      experiments/overcooked_v2_experiments/ttappo_v2_1/utils/partner_diagnostics.py \
      --d "${run_dir}" \
      --num_episodes "${DIAG_EPISODES}" \
      --seed 42 \
      --runs_csv "${chunk}" \
      --output_suffix "${suffix}" &
    pids+=($!)
    worker_idx=$((worker_idx + 1))
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  "$PYTHON_BIN" - <<'PY' "${run_dir}"
import csv
import glob
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

summary_paths = sorted(run_dir.glob("partner_diagnostics_summary_worker_*.csv"))
conf_paths = sorted(run_dir.glob("partner_confusion_aggregate_worker_*.csv"))
per_action_paths = sorted(run_dir.glob("partner_per_action_accuracy_worker_*.csv"))

summary_rows = []
for path in summary_paths:
    with path.open() as f:
        summary_rows.extend(list(csv.DictReader(f)))
summary_rows.sort(key=lambda row: row["run"])

if summary_rows:
    with (run_dir / "partner_diagnostics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "partner_pred_acc",
                "partner_pred_loss",
                "partner_gate_mean",
                "correct",
                "total",
                "num_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

conf_acc = {}
for path in conf_paths:
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["target_action"]), int(row["pred_action"]))
            conf_acc[key] = conf_acc.get(key, 0) + int(row["count"])

if conf_acc:
    with (run_dir / "partner_confusion_aggregate.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "pred_action", "count"])
        for key in sorted(conf_acc):
            writer.writerow([key[0], key[1], conf_acc[key]])

per_action_acc = {}
for path in per_action_paths:
    with path.open() as f:
        for row in csv.DictReader(f):
            target = int(row["target_action"])
            correct = int(row["correct"])
            total = int(row["total"])
            if target not in per_action_acc:
                per_action_acc[target] = [0, 0]
            per_action_acc[target][0] += correct
            per_action_acc[target][1] += total

if per_action_acc:
    with (run_dir / "partner_per_action_accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "correct", "total", "accuracy"])
        for target in sorted(per_action_acc):
            correct, total = per_action_acc[target]
            acc = correct / max(total, 1)
            writer.writerow([target, correct, total, acc])

for path in summary_paths + conf_paths + per_action_paths:
    path.unlink(missing_ok=True)
PY
}

aggregate_variant() {
  local variant="$1"
  local output_dir="runs/${PREFIX_BASE}_summary"
  mkdir -p "${output_dir}"
  python3 - <<'PY' "${ROOT_DIR}" "${PREFIX_BASE}" "${variant}" "${output_dir}"
import csv
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix_base = sys.argv[2]
variant = sys.argv[3]
output_dir = Path(sys.argv[4])

rows = []
for reward_csv in sorted(root.glob(f"runs/{prefix_base}_{variant}_r*/**/reward_summary_cross_no_adapt.csv")):
    run_dir = reward_csv.parent
    diag_csv = run_dir / "partner_diagnostics_summary.csv"
    with reward_csv.open() as f:
        reader = csv.DictReader(f)
        rewards = [float(r["total_reward"]) for r in reader]
    sp_rewards = [r for r in rewards[-10:]]
    xp_rewards = [r for r in rewards[:-10]]
    sp = sum(sp_rewards) / max(len(sp_rewards), 1)
    xp = sum(xp_rewards) / max(len(xp_rewards), 1)
    mean_acc = ""
    mean_loss = ""
    mean_gate = ""
    if diag_csv.exists():
        with diag_csv.open() as f:
            drows = list(csv.DictReader(f))
        if drows:
            mean_acc = statistics.mean(float(r["partner_pred_acc"]) for r in drows)
            mean_loss = statistics.mean(float(r["partner_pred_loss"]) for r in drows)
            mean_gate = statistics.mean(float(r["partner_gate_mean"]) for r in drows)
    rows.append({
        "variant": variant,
        "run_dir": str(run_dir),
        "sp": sp,
        "xp": xp,
        "partner_pred_acc": mean_acc,
        "partner_pred_loss": mean_loss,
        "partner_gate_mean": mean_gate,
    })

summary_csv = output_dir / f"{variant}_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "variant",
            "run_dir",
            "sp",
            "xp",
            "partner_pred_acc",
            "partner_pred_loss",
            "partner_gate_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

if rows:
    aggregate_csv = output_dir / f"{variant}_aggregate.csv"
    with aggregate_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "sp_mean", "sp_std", "xp_mean", "xp_std"],
        )
        writer.writeheader()
        writer.writerow({
            "variant": variant,
            "sp_mean": statistics.mean(r["sp"] for r in rows),
            "sp_std": statistics.pstdev(r["sp"] for r in rows),
            "xp_mean": statistics.mean(r["xp"] for r in rows),
            "xp_std": statistics.pstdev(r["xp"] for r in rows),
        })
print(f"[pipeline] wrote {summary_csv}")
PY
}

IFS=',' read -r -a BASE_SEEDS <<< "${BASE_SEEDS_CSV}"
VARIANTS=("base" "gate")

for variant in "${VARIANTS[@]}"; do
  echo "[pipeline] variant=${variant}"
  idx=1
  for seed in "${BASE_SEEDS[@]}"; do
    prefix="${PREFIX_BASE}_${variant}_r${idx}"
    train_variant "${prefix}" "${seed}" "${variant}"
    run_dir="$(latest_run_dir "${prefix}")"
    echo "[pipeline] run dir ${run_dir}"
    run_noadapt_eval "${run_dir}" "no_adapt"
    run_partner_diagnostics "${run_dir}"
    idx=$((idx + 1))
  done
  aggregate_variant "${variant}"
done

echo "[pipeline] online adaptation variants intentionally paused for v2.1"

if [[ "${RUN_DETERMINISM_CHECK}" == "1" ]]; then
  prefix="${PREFIX_BASE}_determinism_seed42"
  train_variant "${prefix}" "42" "base"
  run_dir="$(latest_run_dir "${prefix}")"
  run_noadapt_eval "${run_dir}" "no_adapt"
  run_partner_diagnostics "${run_dir}"
fi

if [[ "${RUN_V3}" == "1" && -n "${V3_PACKAGE}" ]]; then
  PYTHONPATH=experiments "$PYTHON_BIN" \
    "experiments/overcooked_v2_experiments/${V3_PACKAGE}/main.py" \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="42" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${V3_PREFIX}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
  V3_RUN_DIR="$(latest_run_dir "${V3_PREFIX}")"
  run_noadapt_eval "${V3_RUN_DIR}" "stage3_no_adapt"
fi

echo "[pipeline] done"
