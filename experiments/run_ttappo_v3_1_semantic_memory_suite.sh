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
LAYOUT="${LAYOUT:-counter_circuit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)}"
BASE_SEEDS_CSV="${BASE_SEEDS_CSV:-42,43,44}"
EVAL_SEEDS="${EVAL_SEEDS:-500}"
SMOKE_EVAL_SEEDS="${SMOKE_EVAL_SEEDS:-32}"
DIAG_EPISODES="${DIAG_EPISODES:-100}"
DIAG_WORKERS="${DIAG_WORKERS:-2}"

RUN_STAGE0="${RUN_STAGE0:-0}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
RUN_GATED="${RUN_GATED:-0}"

V2_PREFIX_BASE="${V2_PREFIX_BASE:-figure4_ttappo_v2_temporal_ctrl_${PIPELINE_TAG}}"
V3_PREFIX_BASE="${V3_PREFIX_BASE:-figure4_ttappo_v3_1_semantic_memory_${PIPELINE_TAG}}"
SMOKE_PREFIX="${SMOKE_PREFIX:-${V3_PREFIX_BASE}_smoke}"
SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-262144}"
SMOKE_REW_SHAPING_HORIZON="${SMOKE_REW_SHAPING_HORIZON:-131072}"

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

train_v2_control() {
  local prefix="$1"
  local seed="$2"
  echo "[stage0] train v2 control prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v2_temporal/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}"
}

eval_v2_control() {
  local run_dir="$1"
  echo "[stage0] eval v2 no_adapt run_dir=${run_dir}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v2_temporal/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${EVAL_SEEDS}" \
    --no_viz \
    --seed 42 \
    --ttt_mode no_adapt \
    --output_tag v2_no_adapt
}

train_v3() {
  local prefix="$1"
  local seed="$2"
  shift 2
  local extra_overrides=("$@")
  echo "[train] v3 memory prefix=${prefix} seed=${seed}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/main.py \
    +experiment=cnn \
    +env=original \
    env.ENV_KWARGS.layout="${LAYOUT}" \
    SEED="${seed}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}" \
    VISUALIZE=False \
    +OPTIONAL_PREFIX="${prefix}" \
    wandb.ENTITY="${ENTITY}" \
    wandb.PROJECT="${PROJECT}" \
    "${TRAIN_OVERRIDES[@]}" \
    "${extra_overrides[@]}"
}

eval_v3_mode() {
  local run_dir="$1"
  local mode="$2"
  local eval_seeds="$3"
  echo "[eval] v3 mode=${mode} run_dir=${run_dir}"
  PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/visualize_ppo.py \
    --d "${run_dir}" \
    --cross \
    --num_seeds "${eval_seeds}" \
    --no_viz \
    --seed 42 \
    --ttt_mode "${mode}" \
    --output_tag "${mode}"
}

run_v3_diagnostics() {
  local run_dir="$1"
  local mode="$2"
  echo "[diag] mode=${mode} run_dir=${run_dir}"
  if [[ "${DIAG_WORKERS}" -le 1 ]]; then
    PYTHONPATH=experiments "$PYTHON_BIN" \
      experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/partner_diagnostics.py \
      --d "${run_dir}" \
      --num_episodes "${DIAG_EPISODES}" \
      --seed 42 \
      --mode "${mode}"
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
    local suffix="worker_${worker_idx}"
    local gpu_idx=$((worker_idx % 2))
    echo "[diag] worker=${suffix} gpu=${gpu_idx} runs=${chunk}"
    CUDA_VISIBLE_DEVICES="${gpu_idx}" PYTHONPATH=experiments "$PYTHON_BIN" \
      experiments/overcooked_v2_experiments/ttappo_v3_1_semantic_memory/utils/partner_diagnostics.py \
      --d "${run_dir}" \
      --num_episodes "${DIAG_EPISODES}" \
      --seed 42 \
      --mode "${mode}" \
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
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
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
            writer.writerow([target, correct, total, correct / max(total, 1)])

for path in summary_paths + conf_paths + per_action_paths:
    path.unlink(missing_ok=True)
PY
}

aggregate_suite() {
  local output_dir="runs/${V3_PREFIX_BASE}_summary"
  mkdir -p "${output_dir}"
  "$PYTHON_BIN" - <<'PY' "${ROOT_DIR}" "${V2_PREFIX_BASE}" "${V3_PREFIX_BASE}" "${output_dir}"
import csv
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
v2_prefix = sys.argv[2]
v3_prefix = sys.argv[3]
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

for reward_csv in sorted(root.glob(f"runs/{v2_prefix}_r*/**/reward_summary_cross_v2_no_adapt.csv")):
    metrics = parse_reward_csv(reward_csv)
    table_rows.append({
        "family": "v2_control",
        "mode": "no_adapt",
        "run_dir": str(reward_csv.parent),
        **metrics,
    })

for mode in ("memory_off", "state_adapt", "state_adapt_gated"):
    for reward_csv in sorted(root.glob(f"runs/{v3_prefix}_r*/**/reward_summary_cross_{mode}.csv")):
        metrics = parse_reward_csv(reward_csv)
        diag_csv = reward_csv.parent / "partner_diagnostics_summary.csv"
        diag_mean_acc = ""
        diag_memory_norm = ""
        diag_gamma = ""
        if diag_csv.exists():
            drows = list(csv.DictReader(diag_csv.open()))
            if drows:
                diag_mean_acc = statistics.mean(float(r["partner_pred_acc"]) for r in drows)
                diag_memory_norm = statistics.mean(float(r["memory_norm_mean"]) for r in drows)
                diag_gamma = statistics.mean(float(r["gamma_mean"]) for r in drows)
        table_rows.append({
        "family": "v3_1_semantic_memory",
        "mode": mode,
        "run_dir": str(reward_csv.parent),
        **metrics,
        "partner_pred_acc": diag_mean_acc,
        "memory_norm_mean": diag_memory_norm,
            "gamma_mean": diag_gamma,
        })

summary_csv = output_dir / "suite_summary.csv"
with summary_csv.open("w", newline="") as f:
    fieldnames = [
        "family",
        "mode",
        "run_dir",
        "sp_mean",
        "xp_mean",
        "sp_count",
        "xp_count",
        "partner_pred_acc",
        "memory_norm_mean",
        "gamma_mean",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

aggregate_rows = []
for family, mode in {
    (row["family"], row["mode"])
    for row in table_rows
}:
    rows = [row for row in table_rows if row["family"] == family and row["mode"] == mode]
    sp_vals = [float(row["sp_mean"]) for row in rows if row["sp_mean"] != ""]
    xp_vals = [float(row["xp_mean"]) for row in rows if row["xp_mean"] != ""]
    aggregate_rows.append({
        "family": family,
        "mode": mode,
        "num_runs": len(rows),
        "sp_mean": statistics.mean(sp_vals) if sp_vals else "",
        "sp_std": statistics.pstdev(sp_vals) if len(sp_vals) > 1 else 0.0 if sp_vals else "",
        "xp_mean": statistics.mean(xp_vals) if xp_vals else "",
        "xp_std": statistics.pstdev(xp_vals) if len(xp_vals) > 1 else 0.0 if xp_vals else "",
    })

aggregate_csv = output_dir / "suite_aggregate.csv"
with aggregate_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["family", "mode", "num_runs", "sp_mean", "sp_std", "xp_mean", "xp_std"],
    )
    writer.writeheader()
    writer.writerows(sorted(aggregate_rows, key=lambda row: (row["family"], row["mode"])))

print(f"[stage3] wrote {summary_csv}")
print(f"[stage3] wrote {aggregate_csv}")
PY
}

IFS=',' read -r -a BASE_SEEDS <<< "${BASE_SEEDS_CSV}"

if [[ "${RUN_STAGE0}" == "1" ]]; then
  idx=1
  for seed in "${BASE_SEEDS[@]}"; do
    prefix="${V2_PREFIX_BASE}_r${idx}"
    train_v2_control "${prefix}" "${seed}"
    run_dir="$(latest_run_dir "${prefix}")"
    eval_v2_control "${run_dir}"
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  train_v3 \
    "${SMOKE_PREFIX}" \
    "42" \
    "model.TOTAL_TIMESTEPS=${SMOKE_TOTAL_TIMESTEPS}" \
    "model.REW_SHAPING_HORIZON=${SMOKE_REW_SHAPING_HORIZON}"
  smoke_run_dir="$(latest_run_dir "${SMOKE_PREFIX}")"
  eval_v3_mode "${smoke_run_dir}" "memory_off" "${SMOKE_EVAL_SEEDS}"
  eval_v3_mode "${smoke_run_dir}" "state_adapt" "${SMOKE_EVAL_SEEDS}"
  if [[ "${RUN_GATED}" == "1" ]]; then
    eval_v3_mode "${smoke_run_dir}" "state_adapt_gated" "${SMOKE_EVAL_SEEDS}"
  fi
  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    run_v3_diagnostics "${smoke_run_dir}" "state_adapt"
  fi
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  idx=1
  for seed in "${BASE_SEEDS[@]}"; do
    prefix="${V3_PREFIX_BASE}_r${idx}"
    train_v3 "${prefix}" "${seed}"
    run_dir="$(latest_run_dir "${prefix}")"
    eval_v3_mode "${run_dir}" "memory_off" "${EVAL_SEEDS}"
    eval_v3_mode "${run_dir}" "state_adapt" "${EVAL_SEEDS}"
    if [[ "${RUN_GATED}" == "1" ]]; then
      eval_v3_mode "${run_dir}" "state_adapt_gated" "${EVAL_SEEDS}"
    fi
    if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
      run_v3_diagnostics "${run_dir}" "state_adapt"
    fi
    idx=$((idx + 1))
  done
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  aggregate_suite
fi

echo "[pipeline] done"
