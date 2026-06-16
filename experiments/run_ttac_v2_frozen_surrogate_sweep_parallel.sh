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
REPORT_DIR="${REPORT_DIR:-reports/ttac_v2_frozen_surrogate_sweep_20260603}"
LOG_DIR="${LOG_DIR:-logs/ttac_v2_frozen_surrogate_sweep_parallel_20260603}"
NUM_SEEDS="${NUM_SEEDS:-50}"
MAX_PAIRINGS="${MAX_PAIRINGS:-10}"
EVAL_BATCHES="${EVAL_BATCHES:-5}"
SEED="${SEED:-42}"
MAX_JOBS="${MAX_JOBS:-6}"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
QUEUE="$LOG_DIR/queue.log"
run_one() {
  local tag="$1" mode="$2" scale="$3" lr="$4" steps="$5" hist="$6" ego="$7" cur="$8" beta="$9" min_prob="${10}" max_entropy="${11}"
  local out_tag="surrogate_${tag}"
  local csv="$RUN_DIR/reward_summary_cross_${out_tag}.csv"
  local log="$LOG_DIR/${out_tag}.log"
  if [ -s "$csv" ]; then
    echo "[$(date '+%F %T')] SKIP existing $tag" | tee -a "$QUEUE"
    return 0
  fi
  echo "[$(date '+%F %T')] START $tag" | tee -a "$QUEUE"
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
    --ttac_test_project_beta "$beta" \
    --ttac_test_support_min_prob "$min_prob" \
    --ttac_test_support_max_entropy "$max_entropy" \
    --online_eval_pairings_per_chunk "$EVAL_BATCHES" > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE $tag" | tee -a "$QUEUE"
}
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 5; done
}
configs=(
  "base_no_adapt base_no_test_adapt 0.5 0.003 3 0.0 0.01 0.01 2.0 0.05 1.5"
  "true_default ttac_true_history 0.5 0.003 3 0.0 0.01 0.01 2.0 0.05 1.5"
  "projected_b1 ttac_projected_history 0.5 0.003 3 0.0 0.01 0.01 1.0 0.05 1.5"
  "projected_b2 ttac_projected_history 0.5 0.003 3 0.0 0.01 0.01 2.0 0.05 1.5"
  "projected_b4 ttac_projected_history 0.5 0.003 3 0.0 0.01 0.01 4.0 0.05 1.5"
  "confident_p005 ttac_confident_history 0.5 0.003 3 0.0 0.01 0.01 2.0 0.05 1.5"
  "confident_p010 ttac_confident_history 0.5 0.003 3 0.0 0.01 0.01 2.0 0.10 1.5"
  "projconf_b2_p005 ttac_projected_confident 0.5 0.003 3 0.0 0.01 0.01 2.0 0.05 1.5"
  "projconf_b2_p010 ttac_projected_confident 0.5 0.003 3 0.0 0.01 0.01 2.0 0.10 1.5"
  "projconf_b1_p005 ttac_projected_confident 0.5 0.003 3 0.0 0.01 0.01 1.0 0.05 1.5"
)
echo "[$(date '+%F %T')] SURROGATE_SWEEP_START max_jobs=$MAX_JOBS" | tee -a "$QUEUE"
for cfg in "${configs[@]}"; do
  wait_for_slot
  # shellcheck disable=SC2086
  run_one $cfg &
done
wait
"$PYTHON" - <<PY
from pathlib import Path
import pandas as pd
run=Path('$RUN_DIR')
report=Path('$REPORT_DIR')
configs = [
  ('base_no_adapt','base_no_test_adapt',0.5,0.003,3,0.0,0.01,0.01,2.0,0.05,1.5),
  ('true_default','ttac_true_history',0.5,0.003,3,0.0,0.01,0.01,2.0,0.05,1.5),
  ('projected_b1','ttac_projected_history',0.5,0.003,3,0.0,0.01,0.01,1.0,0.05,1.5),
  ('projected_b2','ttac_projected_history',0.5,0.003,3,0.0,0.01,0.01,2.0,0.05,1.5),
  ('projected_b4','ttac_projected_history',0.5,0.003,3,0.0,0.01,0.01,4.0,0.05,1.5),
  ('confident_p005','ttac_confident_history',0.5,0.003,3,0.0,0.01,0.01,2.0,0.05,1.5),
  ('confident_p010','ttac_confident_history',0.5,0.003,3,0.0,0.01,0.01,2.0,0.10,1.5),
  ('projconf_b2_p005','ttac_projected_confident',0.5,0.003,3,0.0,0.01,0.01,2.0,0.05,1.5),
  ('projconf_b2_p010','ttac_projected_confident',0.5,0.003,3,0.0,0.01,0.01,2.0,0.10,1.5),
  ('projconf_b1_p005','ttac_projected_confident',0.5,0.003,3,0.0,0.01,0.01,1.0,0.05,1.5),
]
rows=[]
for tag,mode,scale,lr,steps,hist,ego,cur,beta,min_prob,max_entropy in configs:
    csv=run/f'reward_summary_cross_surrogate_{tag}.csv'
    if csv.exists():
        df=pd.read_csv(csv)
        rows.append(dict(tag=tag, mode=mode, scale=scale, lr=lr, steps=steps, hist_kl=hist, ego_kl=ego, cur_kl=cur, beta=beta, min_prob=min_prob, max_entropy=max_entropy, mean_reward=float(df.total_reward.mean()), std_reward=float(df.total_reward.std()), min_reward=float(df.total_reward.min()), max_reward=float(df.total_reward.max()), n=len(df), summary_file=str(csv)))
    else:
        rows.append(dict(tag=tag, mode=mode, scale=scale, lr=lr, steps=steps, hist_kl=hist, ego_kl=ego, cur_kl=cur, beta=beta, min_prob=min_prob, max_entropy=max_entropy, mean_reward=float('nan'), std_reward=float('nan'), min_reward=float('nan'), max_reward=float('nan'), n=0, summary_file=str(csv)))
df=pd.DataFrame(rows)
df.to_csv(report/'summary.csv', index=False)
dfs=df.sort_values('mean_reward', ascending=False)
dfs.to_csv(report/'summary_sorted.csv', index=False)
lines=['# TTACv2 Frozen Real Surrogate Sweep','',f'run_dir: {run}',f'num_eval_seeds: {int('$NUM_SEEDS')}',f'max_pairings: {int('$MAX_PAIRINGS')}',f'eval_batches: {int('$EVAL_BATCHES')}',f'max_jobs: {int('$MAX_JOBS')}','',dfs.to_csv(index=False)]
(report/'summary.md').write_text('\n'.join(lines))
print(dfs.to_string(index=False))
PY
echo "[$(date '+%F %T')] SURROGATE_SWEEP_DONE" | tee -a "$QUEUE"
