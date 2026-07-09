from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.build_prototype_prior_dataset import (  # noqa: E501
    history_scores,
    load_prototypes,
)

EPS = 1e-6


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--ppo_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument(
        "--mappo_run_dir",
        type=Path,
        default=Path("runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full"),
    )
    parser.add_argument("--ppo_policy_indices", type=str)
    parser.add_argument("--mappo_policy_indices", type=str)
    parser.add_argument("--max_ppo_policies", type=int, default=8)
    parser.add_argument("--max_mappo_policies", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--score_mode", choices=("sum_logp", "mean_logp"), default="mean_logp")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high_disagreement_quantile", type=float, default=0.8)
    parser.add_argument("--low_true_posterior_threshold", type=float, default=0.25)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    required = [
        "partner_obs_history",
        "partner_action_history",
        "partner_policy_id",
        "query_disagreement_tv",
        "target_partner_probs",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")
    n = int(arrays["partner_action_history"].shape[0])
    if args.max_rows and n > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(np.arange(n), size=args.max_rows, replace=False).astype(np.int32))
    else:
        idx = np.arange(n, dtype=np.int32)

    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    true_policy_ids = np.asarray(arrays["partner_policy_id"][idx]).astype(str)
    disagreement = np.asarray(arrays["query_disagreement_tv"][idx], dtype=np.float32)
    target = normalize_rows(arrays["target_partner_probs"][idx])
    mean_teacher = (
        normalize_rows(arrays["mean_teacher_probs"][idx])
        if "mean_teacher_probs" in arrays
        else target
    )
    residual_target_tv = 0.5 * np.abs(target - mean_teacher).sum(axis=-1)

    prototypes = load_prototypes(args)
    labels = np.asarray([f"{entry.pool}:{entry.label}" for entry in prototypes], dtype=str)
    label_to_pos = {label: pos for pos, label in enumerate(labels.tolist())}
    valid = np.asarray([label in label_to_pos for label in true_policy_ids])
    if not np.all(valid):
        skipped = int(np.sum(~valid))
        print(f"[identifiability] skip rows with unavailable true policy: {skipped}", flush=True)
    idx_local = np.where(valid)[0]
    if len(idx_local) == 0:
        raise ValueError("no rows have true policy ids in the prototype set")
    obs_hist = obs_hist[idx_local]
    act_hist = act_hist[idx_local]
    true_policy_ids = true_policy_ids[idx_local]
    disagreement = disagreement[idx_local]
    residual_target_tv = residual_target_tv[idx_local]

    scores = []
    for entry in prototypes:
        label = f"{entry.pool}:{entry.label}"
        print(f"[identifiability] scoring history with {label}", flush=True)
        scores.append(history_scores(entry.policy, obs_hist, act_hist, args.batch_size, args.score_mode))
    scores = np.stack(scores, axis=0)
    logits = scores / max(float(args.temperature), EPS)
    logits = logits - logits.max(axis=0, keepdims=True)
    posterior = np.exp(logits)
    posterior = posterior / np.maximum(posterior.sum(axis=0, keepdims=True), EPS)

    true_pos = np.asarray([label_to_pos[label] for label in true_policy_ids], dtype=np.int32)
    pred_pos = np.argmax(posterior, axis=0)
    true_posterior = posterior[true_pos, np.arange(len(true_pos))]
    sorted_pos = np.argsort(-posterior, axis=0)
    true_rank = np.empty((len(true_pos),), dtype=np.int32)
    for row, pos in enumerate(true_pos):
        true_rank[row] = int(np.where(sorted_pos[:, row] == pos)[0][0]) + 1

    high_threshold = float(np.quantile(disagreement, args.high_disagreement_quantile))
    high = disagreement >= high_threshold
    low_sep = true_posterior < float(args.low_true_posterior_threshold)
    bucket_rows = []
    for name, mask in {
        "global": np.ones_like(high, dtype=bool),
        "high_disagreement": high,
        "low_disagreement": disagreement < 0.10,
        "high_disagreement_low_separability": high & low_sep,
    }.items():
        if not np.any(mask):
            continue
        bucket_rows.append({
            "slice": name,
            "rows": int(np.sum(mask)),
            "retrieval_acc": float(np.mean(pred_pos[mask] == true_pos[mask])),
            "mean_true_posterior": float(np.mean(true_posterior[mask])),
            "median_true_rank": float(np.median(true_rank[mask])),
            "mean_query_disagreement_tv": float(np.mean(disagreement[mask])),
            "mean_residual_target_tv": float(np.mean(residual_target_tv[mask])),
        })

    with (out / "identifiability_slices.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice",
                "rows",
                "retrieval_acc",
                "mean_true_posterior",
                "median_true_rank",
                "mean_query_disagreement_tv",
                "mean_residual_target_tv",
            ],
        )
        writer.writeheader()
        writer.writerows(bucket_rows)

    high_low_sep_fraction = float(np.mean(low_sep[high])) if np.any(high) else float("nan")
    lines = ["# Identifiable Dataset Diagnostic", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- rows_used: `{len(true_pos)}`")
    lines.append(f"- prototype_count: `{len(labels)}`")
    lines.append(f"- temperature: `{args.temperature}`")
    lines.append(f"- score_mode: `{args.score_mode}`")
    lines.append(f"- query_disagreement_mean: `{float(np.mean(disagreement))}`")
    lines.append(f"- query_disagreement_p50: `{float(np.quantile(disagreement, 0.5))}`")
    lines.append(f"- query_disagreement_p90: `{float(np.quantile(disagreement, 0.9))}`")
    lines.append(f"- global_retrieval_acc: `{float(np.mean(pred_pos == true_pos))}`")
    lines.append(f"- global_true_posterior_mean: `{float(np.mean(true_posterior))}`")
    lines.append(f"- high_disagreement_threshold: `{high_threshold}`")
    lines.append(f"- high_disagreement_low_separability_fraction: `{high_low_sep_fraction}`")
    lines.append("")
    lines.append("| slice | rows | retrieval acc | true posterior | median rank | query TV | residual TV |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in bucket_rows:
        lines.append(
            f"| {row['slice']} | {row['rows']} | {row['retrieval_acc']:.4f} | "
            f"{row['mean_true_posterior']:.4f} | {row['median_true_rank']:.1f} | "
            f"{row['mean_query_disagreement_tv']:.4f} | {row['mean_residual_target_tv']:.4f} |"
        )
    (out / "identifiability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[identifiability] wrote {out / 'identifiability_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
