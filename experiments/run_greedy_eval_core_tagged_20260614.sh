#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
SEED="${SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
TS="${TS:-20260614_greedy_core_500_tagged}"
OUT_ROOT="${OUT_ROOT:-reports/greedy_eval_${TS}}"
LOG_DIR="${LOG_DIR:-logs/greedy_eval_${TS}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
PPO_VIS="experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py"
MAPPO_VIS="experiments/overcooked_v2_experiments/mappo/utils/visualize.py"
TAG="greedy"
log(){ echo "[$(date +%F\ %T)] $*" | tee -a "$LOG_DIR/queue.log"; }
run_eval(){
  local name="$1" backend="$2" src="$3"
  log "START $name backend=$backend src=$src seeds=$NUM_EVAL_SEEDS greedy=true tag=$TAG"
  if [[ "$backend" == "ppo" ]]; then
    PYTHONPATH=experiments "$PYTHON_BIN" "$PPO_VIS" --d "$src" --all --num_seeds "$NUM_EVAL_SEEDS" --no_viz --seed "$SEED" --greedy --output_tag "$TAG" > "$LOG_DIR/$name.log" 2>&1
  elif [[ "$backend" == "mappo" ]]; then
    PYTHONPATH=experiments "$PYTHON_BIN" "$MAPPO_VIS" --d "$src" --all --num_seeds "$NUM_EVAL_SEEDS" --no_viz --seed "$SEED" --greedy --output_tag "$TAG" > "$LOG_DIR/$name.log" 2>&1
  else
    echo "unknown backend $backend" >&2; exit 2
  fi
  log "DONE $name"
}
cat > "$OUT_ROOT/manifest.csv" <<EOF
name,backend,source_run_dir,num_eval_seeds,seed,eval_mode,output_tag
ppo_cnn_standard,ppo,runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
ppo_cnn_state_aug,ppo,runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
mappo_cnn_standard,mappo,runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
mappo_cnn_state_aug,mappo,runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_state_aug_20260503-201254/20260503-202330_umwsoxcy_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
mep_ent0p1_standard,ppo,runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
trajedi_div0p1_standard,ppo,runs/trajedi_unified_rerun_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260515-174148/20260515-174201_cglkefqe_counter_circuit_avs-full,$NUM_EVAL_SEEDS,$SEED,greedy,$TAG
EOF
run_eval ppo_cnn_standard ppo runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full
run_eval ppo_cnn_state_aug ppo runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full
run_eval mappo_cnn_standard mappo runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full
run_eval mappo_cnn_state_aug mappo runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_state_aug_20260503-201254/20260503-202330_umwsoxcy_counter_circuit_avs-full
run_eval mep_ent0p1_standard ppo runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full
run_eval trajedi_div0p1_standard ppo runs/trajedi_unified_rerun_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260515-174148/20260515-174201_cglkefqe_counter_circuit_avs-full
"$PYTHON_BIN" - <<PY
from pathlib import Path
import csv, re
out=Path("$OUT_ROOT")
rows=[]
def parse_pair(label):
    nums=re.findall(r"\\d+", label)
    return (nums[-2], nums[-1]) if len(nums)>=2 else (None,None)
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
for rec in csv.DictReader(open(out/"manifest.csv")):
    name=rec["name"]; src=Path(rec["source_run_dir"]); tag=rec["output_tag"]
    sp_path=src/f"reward_summary_sp_{tag}.csv"; cross_path=src/f"reward_summary_cross_{tag}.csv"
    sp=[]; xp=[]; cross=[]
    if sp_path.exists():
        with sp_path.open() as f:
            for r in csv.DictReader(f): sp.append(float(r["total_reward"]))
    if cross_path.exists():
        with cross_path.open() as f:
            reader=csv.DictReader(f); fields=reader.fieldnames or []
            pair_col="policy_labels" if "policy_labels" in fields else ("checkpoint" if "checkpoint" in fields else fields[1])
            for r in reader:
                val=float(r["total_reward"]); cross.append(val)
                a,b=parse_pair(r[pair_col])
                if a is not None and b is not None and a!=b: xp.append(val)
    rows.append({"name":name,"sp":mean(sp),"xp":mean(xp),"cross_all":mean(cross),"num_sp_episodes":len(sp),"num_xp_episodes":len(xp),"num_cross_episodes":len(cross),"sp_csv":str(sp_path),"cross_csv":str(cross_path)})
with (out/"greedy_core_summary.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with (out/"greedy_core_summary.md").open("w") as f:
    f.write("# Greedy core eval summary\\n\\n")
    f.write("| method | SP | XP | cross_all | SP episodes | XP episodes |\\n")
    f.write("|---|---:|---:|---:|---:|---:|\\n")
    for r in rows:
        f.write(f"| {r[name]} | {r[sp]:.3f} | {r[xp]:.3f} | {r[cross_all]:.3f} | {r[num_sp_episodes]} | {r[num_xp_episodes]} |\\n")
print(out/"greedy_core_summary.md")
PY
log "ALL_DONE out_root=$OUT_ROOT"
