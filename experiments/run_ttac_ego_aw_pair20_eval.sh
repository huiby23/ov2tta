#!/usr/bin/env bash
set -euo pipefail

ROOT=/teams/ius_1663576043/hby/rl/ov2
PYTHON=/root/miniconda3/envs/myconda/bin/python
cd "$ROOT"

export PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RUN="runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"
MODES=(
  base_no_test_adapt
  ttac_ego_advantage_weighted
  ttac_ego_aw_wrong_history
  ttac_ego_aw_random_history
  ttac_ego_aw_delayed_history
)

for mode in "${MODES[@]}"; do
  echo "EVAL_START ${mode}"
  "$PYTHON" experiments/overcooked_v2_experiments/ttac_v2/utils/visualize_ppo.py \
    --d "$RUN" \
    --seed 42 \
    --num_seeds 100 \
    --cross \
    --no_viz \
    --ttac_mode "$mode" \
    --output_tag "pair20_100_${mode}" \
    --eval_batches 10 \
    --max_pairings 20 \
    --ttac_adapter_scale 0.5 \
    --ttac_test_hist_kl_coef 0.0 \
    --ttac_test_ego_kl_coef 0.01 \
    --ttac_test_cur_kl_coef 0.01 \
    --ttac_test_lr 0.003 \
    --ttac_test_update_steps 3 \
    --ttac_history_len 50 \
    --ttac_test_project_beta 2.0 \
    --ttac_test_support_min_prob 0.05 \
    --ttac_test_support_max_entropy 1.5 \
    --ttac_test_advantage_power 1.0
  echo "EVAL_DONE ${mode}"
done

"$PYTHON" - <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np

run = Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606")
rows = []
for path in sorted(run.glob("reward_summary_cross_pair20_100_*.csv")):
    mode = path.name.removeprefix("reward_summary_cross_pair20_100_").removesuffix(".csv")
    per_pair = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            per_pair[row["policy_labels"]].append(float(row["total_reward"]))
    values = [float(np.mean(v)) for v in per_pair.values()]
    rows.append({
        "mode": mode,
        "pair20_mean": float(np.mean(values)) if values else float("nan"),
        "pair20_std": float(np.std(values)) if values else float("nan"),
        "num_pairs": len(values),
    })

out = run / "ttac_ego_aw_pair20_100_summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["mode", "pair20_mean", "pair20_std", "num_pairs"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
for row in rows:
    print(row)
PY
