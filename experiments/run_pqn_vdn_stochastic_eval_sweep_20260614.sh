#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M}"
SEED="${SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
TS="${TS:-20260614_pqn_vdn_stochastic_eval_500}"
OUT_ROOT="${OUT_ROOT:-reports/pqn_vdn_stochastic_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/pqn_vdn_stochastic_eval_${TS}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
log(){ echo "[$(date +%F\ %T)] $*" | tee -a "$LOG_DIR/queue.log"; }
run_eval(){
  local name="$1" mode="$2" temp="$3" eps="$4"
  local out="$OUT_ROOT/$name"
  log "START $name mode=$mode temperature=$temp epsilon=$eps seeds=$NUM_EVAL_SEEDS"
  PYTHONPATH=experiments:JaxMARL "$PYTHON_BIN" experiments/overcooked_v2_experiments/qlearning/utils/evaluate_qlearning.py \
    --run_root "$RUN_ROOT" \
    --method pqn_vdn \
    --layout counter_circuit \
    --output_dir "$out" \
    --seed "$SEED" \
    --num_eval_seeds "$NUM_EVAL_SEEDS" \
    --pairing_batch_size 1 \
    --modes sp,xp \
    --action_mode "$mode" \
    --temperature "$temp" \
    --epsilon "$eps" \
    > "$LOG_DIR/$name.log" 2>&1
  log "DONE $name"
}
cat > "$OUT_ROOT/manifest.csv" <<EOF
name,action_mode,temperature,epsilon,num_eval_seeds,seed
softmax_t0p02,softmax,0.02,0.0,$NUM_EVAL_SEEDS,$SEED
softmax_t0p05,softmax,0.05,0.0,$NUM_EVAL_SEEDS,$SEED
softmax_t0p1,softmax,0.1,0.0,$NUM_EVAL_SEEDS,$SEED
softmax_t0p2,softmax,0.2,0.0,$NUM_EVAL_SEEDS,$SEED
softmax_t0p5,softmax,0.5,0.0,$NUM_EVAL_SEEDS,$SEED
softmax_t1p0,softmax,1.0,0.0,$NUM_EVAL_SEEDS,$SEED
eps0p01,epsilon_greedy,1.0,0.01,$NUM_EVAL_SEEDS,$SEED
eps0p05,epsilon_greedy,1.0,0.05,$NUM_EVAL_SEEDS,$SEED
eps0p1,epsilon_greedy,1.0,0.1,$NUM_EVAL_SEEDS,$SEED
eps0p2,epsilon_greedy,1.0,0.2,$NUM_EVAL_SEEDS,$SEED
EOF
run_eval softmax_t0p02 softmax 0.02 0.0
run_eval softmax_t0p05 softmax 0.05 0.0
run_eval softmax_t0p1 softmax 0.1 0.0
run_eval softmax_t0p2 softmax 0.2 0.0
run_eval softmax_t0p5 softmax 0.5 0.0
run_eval softmax_t1p0 softmax 1.0 0.0
run_eval eps0p01 epsilon_greedy 1.0 0.01
run_eval eps0p05 epsilon_greedy 1.0 0.05
run_eval eps0p1 epsilon_greedy 1.0 0.1
run_eval eps0p2 epsilon_greedy 1.0 0.2
"$PYTHON_BIN" - <<PY
from pathlib import Path
import csv
out=Path("$OUT_ROOT")
rows=[]
for rec in csv.DictReader(open(out/"manifest.csv")):
    p=out/rec["name"]/"summary.csv"
    if not p.exists():
        continue
    for r in csv.DictReader(open(p)):
        rows.append({**rec, **r})
with (out/"pqn_vdn_stochastic_sweep_summary.csv").open("w", newline="") as f:
    if rows:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
with (out/"pqn_vdn_stochastic_sweep_summary.md").open("w") as f:
    f.write("# PQN-VDN stochastic eval sweep\\n\\n")
    f.write("| name | mode | action_mode | temperature | epsilon | mean | std_pair | pairs | episodes |\\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\\n")
    for r in rows:
        f.write("| {name} | {mode} | {action_mode} | {temperature} | {epsilon} | {mean} | {std_pair} | {num_pairs} | {num_episodes} |\\n".format(**r))
print(out/"pqn_vdn_stochastic_sweep_summary.md")
PY
log "ALL_DONE out_root=$OUT_ROOT"
