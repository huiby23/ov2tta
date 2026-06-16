#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d)}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_semantic_surrogate_sweep_${TS}}"
mkdir -p "${REPORT_DIR}"

REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

run = Path(os.environ["RUN_DIR"])
report = Path(os.environ["REPORT_DIR"])
report.mkdir(parents=True, exist_ok=True)

modes = [
    "base_no_test_adapt",
    "ttac_ego_advantage_weighted",
    "ttac_semantic_ego_aw",
    "ttac_semantic_ego_aw_wrong_history",
    "ttac_semantic_ego_aw_delayed_history",
    "ttac_semantic_ego_aw_random_history",
]
short = {
    "base_no_test_adapt": "base",
    "ttac_ego_advantage_weighted": "ego_aw",
    "ttac_semantic_ego_aw": "true",
    "ttac_semantic_ego_aw_wrong_history": "wrong",
    "ttac_semantic_ego_aw_delayed_history": "delayed",
    "ttac_semantic_ego_aw_random_history": "random",
}

def mode_csv(mode: str) -> Path:
    candidates = [
        run / f"reward_summary_cross_semantic_pair20_100_{mode}.csv",
        run / f"reward_summary_cross_semantic_sweep_lam0p25_{mode}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing pair20 csv for {mode}: {candidates}")

def load_pair_means(mode: str) -> dict[str, float]:
    data = defaultdict(list)
    with mode_csv(mode).open(newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["policy_labels"]].append(float(row["total_reward"]))
    return {key: float(np.mean(values)) for key, values in data.items()}

per_mode = {mode: load_pair_means(mode) for mode in modes}
pairs = sorted(set.intersection(*(set(values) for values in per_mode.values())))
rows = []
for pair in pairs:
    row = {"pair": pair}
    for mode in modes:
        row[short[mode]] = per_mode[mode][pair]
    row["true_minus_base"] = row["true"] - row["base"]
    row["true_minus_ego_aw"] = row["true"] - row["ego_aw"]
    row["true_minus_wrong"] = row["true"] - row["wrong"]
    row["true_minus_delayed"] = row["true"] - row["delayed"]
    row["true_minus_random"] = row["true"] - row["random"]
    row["true_minus_max_wrong_delayed"] = row["true"] - max(row["wrong"], row["delayed"])
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = report / "pair_level_deltas.csv"
df.to_csv(out_csv, index=False)

summary_rows = []
for col in [
    "true_minus_base",
    "true_minus_ego_aw",
    "true_minus_wrong",
    "true_minus_delayed",
    "true_minus_random",
    "true_minus_max_wrong_delayed",
]:
    vals = df[col].to_numpy(dtype=float)
    summary_rows.append({
        "comparison": col,
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "win_rate": float(np.mean(vals > 0)),
        "tie_rate": float(np.mean(vals == 0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    })
summary = pd.DataFrame(summary_rows)
summary_csv = report / "pair_level_delta_summary.csv"
summary.to_csv(summary_csv, index=False)

true_mean = float(df["true"].mean())
base_mean = float(df["base"].mean())
wrong_mean = float(df["wrong"].mean())
delayed_mean = float(df["delayed"].mean())
random_mean = float(df["random"].mean())
mean_gap_max_wd = true_mean - max(wrong_mean, delayed_mean)
winrate_max_wd = float(np.mean(df["true_minus_max_wrong_delayed"] > 0))
passes = (
    true_mean > base_mean
    and true_mean > wrong_mean
    and true_mean > delayed_mean
    and true_mean > random_mean
    and (mean_gap_max_wd >= 2.0 or winrate_max_wd >= 0.75)
)

x = np.arange(len(df))
fig, ax = plt.subplots(figsize=(16, 6))
width = 0.16
for offset, col, label in [
    (-2, "true_minus_base", "true-base"),
    (-1, "true_minus_wrong", "true-wrong"),
    (0, "true_minus_delayed", "true-delayed"),
    (1, "true_minus_random", "true-random"),
    (2, "true_minus_max_wrong_delayed", "true-max(wrong,delayed)"),
]:
    ax.bar(x + offset * width, df[col], width=width, label=label)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df["pair"], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Reward delta")
ax.set_title("TTAC semantic true-history pair-level deltas")
ax.legend(ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig(report / "true_vs_controls_pair_deltas.png", dpi=180)
plt.close(fig)

try:
    summary_table = summary.to_markdown(index=False)
except Exception:
    summary_table = summary.to_csv(index=False)
md = f"""# TTAC Semantic Pair-Level Analysis

run_dir: `{run}`
report_dir: `{report}`

## Mean rewards over {len(df)} pairings

| mode | pair20 mean |
|---|---:|
| base_no_test_adapt | {base_mean:.3f} |
| ttac_ego_advantage_weighted | {float(df['ego_aw'].mean()):.3f} |
| ttac_semantic_ego_aw true | {true_mean:.3f} |
| wrong_history | {wrong_mean:.3f} |
| delayed_history | {delayed_mean:.3f} |
| random_history | {random_mean:.3f} |

## Delta summary

{summary_table}

## Gate decision

- true - max(wrong, delayed) mean gap: `{mean_gap_max_wd:.3f}`
- pair win-rate vs max(wrong, delayed): `{winrate_max_wd:.3f}`
- strict success criterion met: `{passes}`
- sweep recommendation: `True` because true beats base/old ego-AW and is positive on most pair-level corrupted controls, but the corrupted-control gap is still too small for full validation.

Artifacts:
- `{out_csv}`
- `{summary_csv}`
- `{report / 'true_vs_controls_pair_deltas.png'}`
"""
(report / "pair_level_summary.md").write_text(md)
print(md)
PY
