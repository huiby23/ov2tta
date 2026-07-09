from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def label_kind(label: str) -> str:
    match = re.search(r"cross-(\d+)_(\d+)$", label)
    if not match:
        return "unknown"
    return "sp" if match.group(1) == match.group(2) else "xp"


def read_reward_rows(paths: list[Path]):
    rows = []
    header = None
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            local_header = next(reader)
            if header is None:
                header = local_header
            elif local_header != header:
                raise ValueError(f"Header mismatch in {path}: {local_header} != {header}")
            rows.extend(list(reader))
    return header, rows


def summarize(paths: list[Path]):
    header, rows = read_reward_rows(paths)
    label_idx = header.index("policy_labels")
    reward_idx = header.index("total_reward")
    pair_values = defaultdict(list)
    for row in rows:
        pair_values[row[label_idx]].append(float(row[reward_idx]))
    per_pair = {label: statistics.fmean(vals) for label, vals in pair_values.items()}
    sp_vals = [v for k, v in per_pair.items() if label_kind(k) == "sp"]
    xp_vals = [v for k, v in per_pair.items() if label_kind(k) == "xp"]
    all_vals = list(per_pair.values())
    return {
        "num_files": len(paths),
        "num_rows": len(rows),
        "num_pairs": len(per_pair),
        "sp_pairs": len(sp_vals),
        "xp_pairs": len(xp_vals),
        "sp_mean": statistics.fmean(sp_vals) if sp_vals else float("nan"),
        "xp_mean": statistics.fmean(xp_vals) if xp_vals else float("nan"),
        "all_mean": statistics.fmean(all_vals) if all_vals else float("nan"),
        "per_pair": per_pair,
        "pair_values": pair_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--glob_suffix", default="seedchunk*_shard*.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pattern = f"reward_summary_cross_{args.tag}_{args.glob_suffix}"
    paths = sorted(run_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {run_dir / pattern}")

    result = summarize(paths)
    print(f"files={result['num_files']} rows={result['num_rows']} pairs={result['num_pairs']}")
    print(f"SP={result['sp_mean']:.3f} over {result['sp_pairs']} pairs")
    print(f"XP={result['xp_mean']:.3f} over {result['xp_pairs']} pairs")
    print(f"ALL={result['all_mean']:.3f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_csv = out / "partial_or_full_summary.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tag",
                    "num_files",
                    "num_rows",
                    "num_pairs",
                    "sp_pairs",
                    "xp_pairs",
                    "sp_mean",
                    "xp_mean",
                    "all_mean",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "tag": args.tag,
                "num_files": result["num_files"],
                "num_rows": result["num_rows"],
                "num_pairs": result["num_pairs"],
                "sp_pairs": result["sp_pairs"],
                "xp_pairs": result["xp_pairs"],
                "sp_mean": result["sp_mean"],
                "xp_mean": result["xp_mean"],
                "all_mean": result["all_mean"],
            })
        pair_csv = out / "partial_or_full_pair_rewards.csv"
        with pair_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["policy_labels", "kind", "pair_reward", "num_annotations"]
            )
            writer.writeheader()
            for label, reward in sorted(result["per_pair"].items()):
                writer.writerow({
                    "policy_labels": label,
                    "kind": label_kind(label),
                    "pair_reward": reward,
                    "num_annotations": len(result["pair_values"][label]),
                })
        md = [
            "# TTAC Sharded Eval Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| files | {result['num_files']} |",
            f"| rows | {result['num_rows']} |",
            f"| pairs | {result['num_pairs']} |",
            f"| SP | {result['sp_mean']:.3f} |",
            f"| XP | {result['xp_mean']:.3f} |",
            f"| ALL | {result['all_mean']:.3f} |",
        ]
        (out / "partial_or_full_summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
