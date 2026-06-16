#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TRAIN_PID="${TRAIN_PID:?TRAIN_PID is required}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SOURCE_RUN="${SOURCE_RUN:-${ROOT}/runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-${ROOT}/runs/talents_full_20260525_152903/vae/gamma_vae.pkl}"
CLUSTER_CHECKPOINT="${CLUSTER_CHECKPOINT:-${ROOT}/runs/talents_full_20260525_152903/clusters/talents_clusters.pkl}"
PATTERN="${PATTERN:-${ROOT}/runs/talents_mix25_priority_high_return_cnn_standard_64_16_10000000_*/*_counter_circuit_avs-full}"

echo "[watch] waiting for training pid=${TRAIN_PID}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 300
done
echo "[watch] training pid ended"

RUN_DIR="$(ls -td ${PATTERN} 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "[watch] could not find TALENTS priority run dir with pattern: ${PATTERN}" >&2
  exit 2
fi
echo "[watch] run_dir=${RUN_DIR}"

EVAL_LOG="${ROOT}/logs/talents_priority_eval_100_${TS}.log"
echo "[watch] running SP/XP eval -> ${EVAL_LOG}"
"${PYTHON}" -m overcooked_v2_experiments.talents.utils.visualize_ppo \
  --d "${RUN_DIR}" \
  --num_seeds "${EVAL_NUM_SEEDS:-100}" \
  --all \
  --no_viz \
  > "${EVAL_LOG}" 2>&1

echo "[watch] parsing reward summaries"
"${PYTHON}" - <<'PY' "${RUN_DIR}"
import csv
import itertools
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

sp_rows = read_rows(run_dir / "reward_summary_sp.csv")
cross_rows = read_rows(run_dir / "reward_summary_cross.csv")

sp = sum(float(r["total_reward"]) for r in sp_rows) / max(len(sp_rows), 1)
diag = []
xp = []
for r in cross_rows:
    labels = r["policy_labels"].replace("cross-", "").split("_")
    if len(labels) != 2:
        continue
    reward = float(r["total_reward"])
    if labels[0] == labels[1]:
        diag.append(reward)
    else:
        xp.append(reward)
print(f"SP mean: {sp:.6f}")
print(f"cross diagonal mean: {sum(diag) / max(len(diag), 1):.6f}")
print(f"XP mean: {sum(xp) / max(len(xp), 1):.6f}")
print(f"SP rows: {len(sp_rows)} | XP rows: {len(xp)} | diagonal rows: {len(diag)}")
PY

CLUSTER_OUT="${ROOT}/reports/talents_cluster_quality_${TS}.json"
CLUSTER_LOG="${ROOT}/logs/talents_cluster_quality_${TS}.log"
echo "[watch] running cluster quality diagnostic -> ${CLUSTER_LOG}"
"${PYTHON}" -m overcooked_v2_experiments.talents.eval_cluster_quality \
  --run-dir "${SOURCE_RUN}" \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --cluster-checkpoint "${CLUSTER_CHECKPOINT}" \
  --backend ppo \
  --seed 42 \
  --num-seeds "${CLUSTER_NUM_SEEDS:-20}" \
  --max-runs "${CLUSTER_MAX_RUNS:-10}" \
  --clusters all \
  --greedy-source \
  --output "${CLUSTER_OUT}" \
  > "${CLUSTER_LOG}" 2>&1

echo "[watch] cluster quality output=${CLUSTER_OUT}"
echo "[watch] done"
