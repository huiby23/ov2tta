from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.policy import PPOPolicy
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.agreement_heads import (
    apply_agreement_estimator,
    categorical_kl,
    categorical_tv,
    load_agreement_npz,
)
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.store import load_all_checkpoints


def reward_rows(path: Path):
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out[(r["policy_labels"], r["annotation"])] = float(r["total_reward"])
    return out


def run_pair_to_cross(pair_label):
    m = re.match(r"run_(\d+)xrun_(\d+)", str(pair_label))
    if not m:
        return str(pair_label)
    return f"cross-{m.group(1)}_{m.group(2)}"


def cross_to_policy_ids(cross_label):
    m = re.match(r"cross-(\d+)_(\d+)", str(cross_label))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_policies(run_dir: Path, estimator, history_len):
    all_ckpts, config = load_all_checkpoints(run_dir, final_only=True)
    if estimator is not None:
        config["model"]["TTAC_V5_ESTIMATOR"] = estimator
    config["model"]["TTAC_HISTORY_LEN"] = history_len
    run_keys = sorted(all_ckpts.keys(), key=lambda x: int(x.split("_")[1]))
    policies = [
        PPOPolicy(all_ckpts[k]["ckpt_final"].params, config, eval_mode="base_no_test_adapt", stochastic=False)
        for k in run_keys
    ]
    return policies


def compute_base_logits_values(policies, obs, ego_ids, batch_size=512):
    logits = np.zeros((len(obs), policies[0].action_dim), dtype=np.float32)
    values = np.zeros((len(obs),), dtype=np.float32)
    for ego_id in sorted(set(int(x) for x in ego_ids)):
        idx = np.where(ego_ids == ego_id)[0]
        policy = policies[ego_id]
        outs = []
        vals = []
        for start in range(0, len(idx), batch_size):
            sub = idx[start:start + batch_size]
            l, v, _aux = policy._apply_batch(policy.params, jnp.asarray(obs[sub], dtype=jnp.float32), 0.0)
            outs.append(np.asarray(l, dtype=np.float32))
            vals.append(np.asarray(v, dtype=np.float32))
        logits[idx] = np.concatenate(outs, axis=0)
        values[idx] = np.concatenate(vals, axis=0)
    return logits, values


def summarize(rows, episode_rows, pair_rows, out_dir: Path):
    deltas = [float(r["episode_delta"]) for r in episode_rows]
    episode_labels = defaultdict(int)
    for r in episode_rows:
        episode_labels[r["episode_label"]] += 1
    pair_labels = defaultdict(int)
    for r in pair_rows:
        pair_labels[r["pair_label_group"]] += 1
    fields = [
        "latest_target_base_tv", "estimator_entropy", "estimator_kl", "estimator_tv",
        "base_value", "partner_action_changed", "tv_gate_003", "tv_gate_005", "tv_gate_008", "tv_gate_012",
    ]
    pair_deltas = [float(r["pair_mean_delta"]) for r in pair_rows]
    lines = ["# TTAC v5.2 paired attribution summary", ""]
    lines.append("This audit pairs base and TTAC rewards by exact `policy_labels` and `annotation`, then attaches pair-level reward deltas to sampled agreement-dataset timesteps for fast state-selection analysis.")
    lines.append("")
    lines.append(f"- paired episodes: `{len(episode_rows)}`")
    lines.append(f"- episode_mean_delta: `{float(np.mean(deltas)) if deltas else 0.0:.4f}`")
    lines.append(f"- episode beneficial/harmful/neutral: `{episode_labels['beneficial']}` / `{episode_labels['harmful']}` / `{episode_labels['neutral']}`")
    lines.append(f"- paired policy pairs: `{len(pair_rows)}`")
    lines.append(f"- pair_mean_delta: `{float(np.mean(pair_deltas)) if pair_deltas else 0.0:.4f}`")
    lines.append(f"- pair beneficial/harmful/neutral: `{pair_labels['beneficial']}` / `{pair_labels['harmful']}` / `{pair_labels['neutral']}`")
    lines.append(f"- sampled timestep rows: `{len(rows)}`")
    lines.append("- sampled timestep labels use the pair-level mean reward delta, not the individual episode delta.")
    lines.append("")
    lines.append("| group | field | mean |")
    lines.append("|---|---|---:|")
    for group in ["beneficial", "harmful", "neutral"]:
        group_rows = [r for r in rows if r["episode_label"] == group]
        for f in fields:
            vals = [float(r[f]) for r in group_rows]
            if vals:
                lines.append(f"| {group} | {f} | {float(np.mean(vals)):.6f} |")
    (out_dir / "paired_attribution_summary.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for f in ["latest_target_base_tv", "estimator_entropy", "base_value"]:
            plt.figure(figsize=(6, 4))
            for group, color in [("beneficial", "tab:green"), ("harmful", "tab:red"), ("neutral", "tab:gray")]:
                vals = [float(r[f]) for r in rows if r["episode_label"] == group]
                if vals:
                    plt.hist(vals, bins=30, alpha=0.45, label=group, color=color)
            plt.title(f)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"hist_{f}.png", dpi=160)
            plt.close()
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz")
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--ttac_csv", required=True)
    parser.add_argument("--ttac_v5_estimator_path", required=True)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = reward_rows(Path(args.base_csv))
    ttac = reward_rows(Path(args.ttac_csv))
    common = sorted(set(base) & set(ttac))
    episode_rows = []
    pair_delta = defaultdict(list)
    for key in common:
        delta = ttac[key] - base[key]
        label = "beneficial" if delta > 1e-6 else ("harmful" if delta < -1e-6 else "neutral")
        row = {
            "policy_labels": key[0], "annotation": key[1],
            "base_total": base[key], "ttac_total": ttac[key],
            "episode_delta": delta, "episode_label": label,
        }
        episode_rows.append(row)
        pair_delta[key[0]].append(delta)
    pair_mean_delta = {k: float(np.mean(v)) for k, v in pair_delta.items()}
    pair_label = {
        k: ("beneficial" if v > 1e-6 else ("harmful" if v < -1e-6 else "neutral"))
        for k, v in pair_mean_delta.items()
    }
    pair_rows = []
    for label in sorted(pair_mean_delta):
        values = pair_delta[label]
        pair_rows.append({
            "policy_labels": label,
            "pair_mean_delta": pair_mean_delta[label],
            "pair_label_group": pair_label[label],
            "num_episodes": len(values),
            "positive_episodes": sum(1 for v in values if v > 1e-6),
            "negative_episodes": sum(1 for v in values if v < -1e-6),
            "neutral_episodes": sum(1 for v in values if abs(v) <= 1e-6),
        })

    data = np.load(args.dataset, allow_pickle=False)
    n = len(data["ego_action"])
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n) if n <= args.max_samples else np.sort(rng.choice(n, size=args.max_samples, replace=False))
    query_obs = np.asarray(data["query_obs"][idx], dtype=np.float32)
    partner_obs = np.asarray(data["partner_obs"][idx], dtype=np.float32)
    hist = np.asarray(data["partner_action_history"][idx], dtype=np.int32)
    target_probs = np.asarray(data["target_partner_probs"][idx], dtype=np.float32)
    raw_pairs = np.asarray(data["pair_label"][idx]).astype(str)
    roles = np.asarray(data["role"][idx], dtype=np.int32)
    timesteps = np.asarray(data["timestep"][idx], dtype=np.int32)

    estimator = load_agreement_npz(args.ttac_v5_estimator_path)
    policies = load_policies(Path(args.run_dir), estimator, args.history_len)
    cross_pairs = np.asarray([run_pair_to_cross(x) for x in raw_pairs])
    ego_ids = []
    for cp, role in zip(cross_pairs, roles):
        ids = cross_to_policy_ids(cp)
        if ids is None:
            ego_ids.append(0)
        else:
            ego_ids.append(ids[0] if int(role) == 0 else ids[1])
    ego_ids = np.asarray(ego_ids, dtype=np.int32)

    base_logits, base_values = compute_base_logits_values(policies, query_obs, ego_ids)
    estimator_logits = apply_agreement_estimator(
        estimator,
        jnp.asarray(query_obs, dtype=jnp.float32),
        jnp.asarray(partner_obs, dtype=jnp.float32),
        jnp.asarray(hist, dtype=jnp.int32),
        policies[0].action_dim,
    )
    estimator_probs = np.asarray(jax.nn.softmax(estimator_logits, axis=-1), dtype=np.float32)
    base_probs = np.asarray(jax.nn.softmax(jnp.asarray(base_logits), axis=-1), dtype=np.float32)
    estimator_entropy = -np.sum(estimator_probs * np.log(np.maximum(estimator_probs, 1e-6)), axis=-1)
    estimator_kl = np.asarray(categorical_kl(jnp.asarray(target_probs), estimator_logits), dtype=np.float32)
    estimator_tv = np.asarray(categorical_tv(jnp.asarray(target_probs), estimator_logits), dtype=np.float32)
    target_base_tv = 0.5 * np.abs(estimator_probs - base_probs).sum(axis=-1)
    partner_changed = (hist[:, -1] != hist[:, -2]).astype(np.int32) if hist.shape[1] >= 2 else np.zeros(len(hist), dtype=np.int32)

    rows = []
    for row_i, cp in enumerate(cross_pairs):
        delta = pair_mean_delta.get(cp, 0.0)
        label = pair_label.get(cp, "neutral")
        rows.append({
            "pair_label": cp,
            "source_pair_label": raw_pairs[row_i],
            "role": int(roles[row_i]),
            "timestep": int(timesteps[row_i]),
            "episode_delta": delta,
            "episode_label": label,
            "latest_target_base_tv": float(target_base_tv[row_i]),
            "estimator_entropy": float(estimator_entropy[row_i]),
            "estimator_kl": float(estimator_kl[row_i]),
            "estimator_tv": float(estimator_tv[row_i]),
            "base_value": float(base_values[row_i]),
            "partner_action_changed": int(partner_changed[row_i]),
            "tv_gate_003": int(target_base_tv[row_i] >= 0.03),
            "tv_gate_005": int(target_base_tv[row_i] >= 0.05),
            "tv_gate_008": int(target_base_tv[row_i] >= 0.08),
            "tv_gate_012": int(target_base_tv[row_i] >= 0.12),
        })

    if episode_rows:
        with (out / "paired_attribution_episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader(); writer.writerows(episode_rows)
    if pair_rows:
        with (out / "paired_attribution_pairs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader(); writer.writerows(pair_rows)
    if rows:
        with (out / "paired_attribution_steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    summarize(rows, episode_rows, pair_rows, out)
    print(f"[paired_attr_fast] wrote {out}")


if __name__ == "__main__":
    main()
