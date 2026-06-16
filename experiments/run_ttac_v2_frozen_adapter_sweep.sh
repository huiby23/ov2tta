#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
cd "$ROOT"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-runs/ttac_v2_frozen_population_mep_ent005_state_aug_64_16_10M_seed42_10seeds_20260601_160202/20260601-160216_mappc662_counter_circuit_avs-full}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v2_frozen_adapter_sweep_20260602}"
LOG_DIR="${LOG_DIR:-logs/ttac_v2_frozen_adapter_sweep_20260602}"
NUM_SEEDS="${NUM_SEEDS:-50}"
MAX_PAIRINGS="${MAX_PAIRINGS:-10}"
EVAL_BATCHES="${EVAL_BATCHES:-5}"
SEED="${SEED:-42}"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
SUMMARY="$REPORT_DIR/summary.csv"
echo "tag,mode,scale,lr,steps,hist_kl,ego_kl,cur_kl,mean_reward,std_reward,min_reward,max_reward,n,summary_file" > "$SUMMARY"
run_eval() {
  local tag="$1" mode="$2" scale="$3" lr="$4" steps="$5" hist="$6" ego="$7" cur="$8"
  local out_tag="sweep_${tag}"
  local log="$LOG_DIR/${out_tag}.log"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$LOG_DIR/queue.log"
  "$PYTHON" -m overcooked_v2_experiments.ttac_v2.utils.visualize_ppo \
    --d "$RUN_DIR" \
    --cross \
    --num_seeds "$NUM_SEEDS" \
    --no_viz \
    --seed "$SEED" \
    --ttac_mode "$mode" \
    --output_tag "$out_tag" \
    --eval_batches "$EVAL_BATCHES" \
    --max_pairings "$MAX_PAIRINGS" \
    --ttac_adapter_scale "$scale" \
    --ttac_test_hist_kl_coef "$hist" \
    --ttac_test_ego_kl_coef "$ego" \
    --ttac_test_cur_kl_coef "$cur" \
    --ttac_test_lr "$lr" \
    --ttac_test_update_steps "$steps" \
    --ttac_history_len 50 \
    --online_eval_pairings_per_chunk "$EVAL_BATCHES" > "$log" 2>&1
  local csv="$RUN_DIR/reward_summary_cross_${out_tag}.csv"
  "$PYTHON" - <<PY
from pathlib import Path
import pandas as pd
csv=Path('$csv')
df=pd.read_csv(csv)
print(','.join(map(str, [
    '$tag','$mode',$scale,$lr,$steps,$hist,$ego,$cur,
    float(df['total_reward'].mean()), float(df['total_reward'].std()),
    float(df['total_reward'].min()), float(df['total_reward'].max()), len(df), str(csv)
])))
PY
  "$PYTHON" - <<PY >> "$SUMMARY"
from pathlib import Path
import pandas as pd
csv=Path('$csv')
df=pd.read_csv(csv)
print(','.join(map(str, [
    '$tag','$mode',$scale,$lr,$steps,$hist,$ego,$cur,
    round(float(df['total_reward'].mean()), 6), round(float(df['total_reward'].std()), 6),
    round(float(df['total_reward'].min()), 6), round(float(df['total_reward'].max()), 6), len(df), str(csv)
])))
PY
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOG_DIR/queue.log"
}
# Same first 10 cross pairings, same 50 eval seeds for all configs.
run_eval base_no_adapt base_no_test_adapt 0.5 0.003 3 0.0 0.01 0.01
run_eval default_lr003_s3_kl001 ttac_true_history 0.5 0.003 3 0.0 0.01 0.01
run_eval lr001_s3_kl001 ttac_true_history 0.5 0.001 3 0.0 0.01 0.01
run_eval lr010_s3_kl001 ttac_true_history 0.5 0.010 3 0.0 0.01 0.01
run_eval lr003_s1_kl001 ttac_true_history 0.5 0.003 1 0.0 0.01 0.01
run_eval lr003_s5_kl001 ttac_true_history 0.5 0.003 5 0.0 0.01 0.01
run_eval lr003_s3_kl0 ttac_true_history 0.5 0.003 3 0.0 0.0 0.0
run_eval lr003_s3_kl005 ttac_true_history 0.5 0.003 3 0.0 0.05 0.05
run_eval scale025_lr003_s3 ttac_true_history 0.25 0.003 3 0.0 0.01 0.01
run_eval scale100_lr003_s3 ttac_true_history 1.0 0.003 3 0.0 0.01 0.01
"$PYTHON" - <<PY
from pathlib import Path
import pandas as pd
report=Path('$REPORT_DIR')
df=pd.read_csv(report/'summary.csv')
df=df.sort_values('mean_reward', ascending=False)
(report/'summary_sorted.csv').write_text(df.to_csv(index=False))
md=['# TTACv2 Frozen Adapter Sweep', '', f'- run_dir: `{Path('$RUN_DIR')}`', f'- num_eval_seeds: `{int('$NUM_SEEDS')}`', f'- max_pairings: `{int('$MAX_PAIRINGS')}`', f'- eval_batches: `{int('$EVAL_BATCHES')}`', '', df.to_markdown(index=False)]
(report/'summary.md').write_text('\n'.join(md))
print(df.to_string(index=False))
PY
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG_DIR/queue.log"
