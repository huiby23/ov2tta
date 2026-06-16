#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M}"
LAYOUT="${LAYOUT:-counter_circuit}"
SEED="${SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
PAIRING_BATCH_SIZE="${PAIRING_BATCH_SIZE:-10}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-reports/qlearning_ov2_1zsc_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/qlearning_ov2_1zsc_eval_${TS}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

log() {
  echo "[$(date +%F\ %T)] $*" | tee -a "$LOG_DIR/queue.log"
}

run_method() {
  local method="$1"
  log "START ${method} SP/XP eval seeds=${NUM_EVAL_SEEDS} batch=${PAIRING_BATCH_SIZE}"
  PYTHONPATH=experiments:JaxMARL "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning.py \
    --run_root "$RUN_ROOT" \
    --method "$method" \
    --layout "$LAYOUT" \
    --output_dir "$OUT_ROOT/$method" \
    --seed "$SEED" \
    --num_eval_seeds "$NUM_EVAL_SEEDS" \
    --pairing_batch_size "$PAIRING_BATCH_SIZE" \
    --modes sp,xp \
    > "$LOG_DIR/${method}.log" 2>&1
  log "DONE ${method}"
}

cat > "$OUT_ROOT/manifest.txt" <<EOF
run_root=${RUN_ROOT}
layout=${LAYOUT}
seed=${SEED}
num_eval_seeds=${NUM_EVAL_SEEDS}
pairing_batch_size=${PAIRING_BATCH_SIZE}
modes=sp,xp
methods=iql,vdn,pqn_vdn
EOF

run_method iql
run_method vdn
run_method pqn_vdn

"$PYTHON_BIN" - <<PY
from pathlib import Path
import csv
out_root=Path("$OUT_ROOT")
rows=[]
for method in ["iql","vdn","pqn_vdn"]:
    p=out_root/method/"summary.csv"
    if not p.exists():
        continue
    with p.open() as f:
        rows.extend(csv.DictReader(f))
combined=out_root/"combined_summary.csv"
if rows:
    with combined.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
md=out_root/"combined_summary.md"
with md.open("w") as f:
    f.write("# QLearning OV2 1-ZSC partner eval summary\\n\\n")
    f.write("| method | mode | mean | std_episode | std_pair | num_pairs | num_episodes |\\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\\n")
    for r in rows:
        f.write("| {method} | {mode} | {mean} | {std_episode} | {std_pair} | {num_pairs} | {num_episodes} |\\n".format(**r))
print(f"combined_summary={combined}")
print(f"combined_markdown={md}")
PY

log "ALL_DONE out_root=${OUT_ROOT}"
