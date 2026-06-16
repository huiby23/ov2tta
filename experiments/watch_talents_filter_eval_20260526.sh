#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
PIPELINE_PID="${PIPELINE_PID:?PIPELINE_PID is required}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/talents_full_20260526_filter_top50_z0_mix25}"
SOURCE_RUN="${SOURCE_RUN:-${ROOT}/runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[watch] waiting for pipeline pid=${PIPELINE_PID}"
while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
  sleep 300
done
echo "[watch] pipeline pid ended"
RUN_DIR="$(find "${RUN_ROOT}" -maxdepth 1 -type d -name '*_counter_circuit_avs-full' | sort | tail -1)"
if [ -z "${RUN_DIR}" ]; then
  echo "[watch] could not find run dir under ${RUN_ROOT}" >&2
  exit 2
fi
echo "[watch] run_dir=${RUN_DIR}"
EVAL_LOG="${ROOT}/logs/talents_filter_top50_z0_mix25_eval_100_${TS}.log"
"${PYTHON}" -m overcooked_v2_experiments.talents.utils.visualize_ppo \
  --d "${RUN_DIR}" \
  --num_seeds "${EVAL_NUM_SEEDS:-100}" \
  --all \
  --no_viz \
  > "${EVAL_LOG}" 2>&1
"${PYTHON}" - <<'PY' "${RUN_DIR}"
import csv, sys
from pathlib import Path
run_dir=Path(sys.argv[1])
def rows(name):
    with (run_dir/name).open() as f:
        return list(csv.DictReader(f))
sp_rows=rows('reward_summary_sp.csv')
cross_rows=rows('reward_summary_cross.csv')
sp=sum(float(r['total_reward']) for r in sp_rows)/max(len(sp_rows),1)
xp=[]; diag=[]
for r in cross_rows:
    labs=r['policy_labels'].replace('cross-','').split('_')
    if len(labs)!=2:
        continue
    rw=float(r['total_reward'])
    (diag if labs[0]==labs[1] else xp).append(rw)
print(f"SP mean: {sp:.6f}")
print(f"cross diagonal mean: {sum(diag)/max(len(diag),1):.6f}")
print(f"XP mean: {sum(xp)/max(len(xp),1):.6f}")
print(f"SP rows: {len(sp_rows)} | XP rows: {len(xp)} | diagonal rows: {len(diag)}")
PY
CLUSTER_OUT="${ROOT}/reports/talents_filter_top50_z0_mix25_cluster_quality_${TS}.json"
CLUSTER_LOG="${ROOT}/logs/talents_filter_top50_z0_mix25_cluster_quality_${TS}.log"
"${PYTHON}" -m overcooked_v2_experiments.talents.eval_cluster_quality \
  --run-dir "${SOURCE_RUN}" \
  --vae-checkpoint "${RUN_ROOT}/vae/gamma_vae.pkl" \
  --cluster-checkpoint "${RUN_ROOT}/clusters/talents_clusters.pkl" \
  --backend ppo \
  --seed 42 \
  --num-seeds "${CLUSTER_NUM_SEEDS:-20}" \
  --max-runs "${CLUSTER_MAX_RUNS:-10}" \
  --clusters all \
  --greedy-source \
  --z-sample-scale 0.0 \
  --output "${CLUSTER_OUT}" \
  > "${CLUSTER_LOG}" 2>&1
echo "[watch] eval_log=${EVAL_LOG}"
echo "[watch] cluster_out=${CLUSTER_OUT}"
echo "[watch] done"
