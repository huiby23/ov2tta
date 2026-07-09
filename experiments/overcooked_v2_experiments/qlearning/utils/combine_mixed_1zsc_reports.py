#!/usr/bin/env python3
"""Combine mixed 1-ZSC raw row CSVs and recompute grouped summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def read_rows(input_dirs: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for input_dir in input_dirs:
        csv_path = input_dir / "mixed_1zsc_rows.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing raw row CSV: {csv_path}")
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["source_report"] = str(input_dir)
                row["total_reward"] = float(row["total_reward"])
                rows.append(row)
    return rows


def summarize(rows: Sequence[dict]) -> list[dict]:
    q_methods = {
        "iql",
        "iql_double",
        "iql_dueling",
        "vdn",
        "vdn_dueling",
        "pqn_vdn",
        "qmix_cnn",
        "qmix_rnn",
        "shaq_ps",
    }
    pg_methods = {"a2c", "ppo_cnn_standard", "mappo_cnn_standard"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["ego_mode"], row["partner_method"], row["role"])].append(row)
        grouped[(row["ego_mode"], row["partner_method"], "both_roles")].append(row)
        grouped[(row["ego_mode"], "all_partners", row["role"])].append(row)
        grouped[(row["ego_mode"], "all_partners", "both_roles")].append(row)
        if row["partner_method"] in q_methods:
            grouped[(row["ego_mode"], "all_q_partners", row["role"])].append(row)
            grouped[(row["ego_mode"], "all_q_partners", "both_roles")].append(row)
        if row["partner_method"] in pg_methods:
            grouped[(row["ego_mode"], "all_pg_partners", row["role"])].append(row)
            grouped[(row["ego_mode"], "all_pg_partners", "both_roles")].append(row)

    summary = []
    for (ego_mode, partner_method, role), items in sorted(grouped.items()):
        rewards = np.asarray([float(x["total_reward"]) for x in items], dtype=np.float64)
        pair_means_map = defaultdict(list)
        for item in items:
            pair_means_map[item["pair_id"]].append(float(item["total_reward"]))
        pair_means = np.asarray([np.mean(v) for v in pair_means_map.values()], dtype=np.float64)
        summary.append(
            {
                "ego_mode": ego_mode,
                "partner_method": partner_method,
                "role": role,
                "mean_reward": float(np.mean(rewards)) if rewards.size else float("nan"),
                "std_episode": float(np.std(rewards)) if rewards.size else float("nan"),
                "std_pair": float(np.std(pair_means)) if pair_means.size else float("nan"),
                "num_episodes": int(rewards.size),
                "num_pairs": int(pair_means.size),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary_rows: Sequence[dict], input_dirs: Sequence[Path]):
    lines = [
        "# Combined Mixed 1-ZSC Evaluation",
        "",
        "Definition: ego policy is paired with non-ZSC partner families. Main value-based pool: IQL variants, VDN variants, PQN-VDN, QMIX, and SHAQ.",
        "",
        "Input reports:",
    ]
    for input_dir in input_dirs:
        lines.append(f"- `{input_dir}`")
    lines += [
        "",
        "| ego_mode | partner_method | role | mean_reward | std_episode | std_pair | num_pairs | num_episodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {ego_mode} | {partner_method} | {role} | {mean_reward:.3f} | {std_episode:.3f} | {std_pair:.3f} | {num_pairs} | {num_episodes} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_dirs)
    summary_rows = summarize(rows)
    raw_fields = [
        "source_report",
        "ego_mode",
        "partner_method",
        "ego_label",
        "partner_label",
        "role",
        "pair_id",
        "annotation",
        "total_reward",
    ]
    summary_fields = [
        "ego_mode",
        "partner_method",
        "role",
        "mean_reward",
        "std_episode",
        "std_pair",
        "num_episodes",
        "num_pairs",
    ]
    write_csv(args.output_dir / "mixed_1zsc_rows.csv", rows, raw_fields)
    write_csv(args.output_dir / "mixed_1zsc_summary.csv", summary_rows, summary_fields)
    write_markdown(args.output_dir / "mixed_1zsc_summary.md", summary_rows, args.input_dirs)
    print(args.output_dir / "mixed_1zsc_summary.md")


if __name__ == "__main__":
    main()
