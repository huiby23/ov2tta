from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ttac_v5_4_loss_variants.policy import PPOPolicy
from overcooked_v2_experiments.ttac_v5_4_loss_variants.utils.agreement_heads import (
    apply_agreement_estimator,
    load_agreement_npz,
)
from overcooked_v2_experiments.ttac_v5_4_loss_variants.utils.store import load_all_checkpoints


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "stay",
    5: "interact",
}


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


def load_policies(run_dir: Path, history_len):
    all_ckpts, config = load_all_checkpoints(run_dir, final_only=True)
    config["model"]["TTAC_HISTORY_LEN"] = history_len
    run_keys = sorted(all_ckpts.keys(), key=lambda x: int(x.split("_")[1]))
    return [
        PPOPolicy(all_ckpts[k]["ckpt_final"].params, config, eval_mode="base_no_test_adapt", stochastic=False)
        for k in run_keys
    ]


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


def compute_estimator_logits(estimator, query_obs, partner_obs, hist, action_dim, batch_size=256):
    logits = []
    for start in range(0, len(query_obs), batch_size):
        end = min(start + batch_size, len(query_obs))
        batch_logits = apply_agreement_estimator(
            estimator,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(partner_obs[start:end], dtype=jnp.float32),
            jnp.asarray(hist[start:end], dtype=jnp.int32),
            action_dim,
        )
        logits.append(np.asarray(batch_logits, dtype=np.float32))
    return np.concatenate(logits, axis=0)


def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-8)


def delta_target(base_probs, estimator_probs, margin, scale):
    positive = np.maximum(estimator_probs - base_probs - margin, 0.0)
    target = base_probs + scale * positive
    target = target / np.maximum(target.sum(axis=-1, keepdims=True), 1e-8)
    return target, positive.sum(axis=-1)


def transition_key(src, dst):
    return f"{ACTION_NAMES.get(int(src), src)}->{ACTION_NAMES.get(int(dst), dst)}"


def summarize_group(rows, group):
    subset = [r for r in rows if r["pair_group"] == group]
    if not subset:
        return []
    n = len(subset)
    trans = Counter(r["base_to_delta_argmax"] for r in subset)
    estimator_trans = Counter(r["base_to_estimator_argmax"] for r in subset)
    out = []
    for name, counter in [("base_to_delta", trans), ("base_to_estimator", estimator_trans)]:
        for key, count in counter.most_common(12):
            out.append({
                "group": group,
                "transition_type": name,
                "transition": key,
                "count": count,
                "rate": count / n,
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz")
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--ttac_csv", required=True)
    parser.add_argument("--ttac_v5_estimator_path", required=True)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimator_batch_size", type=int, default=128)
    parser.add_argument("--delta_margin", type=float, default=0.0)
    parser.add_argument("--delta_scale", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_rewards = reward_rows(Path(args.base_csv))
    ttac_rewards = reward_rows(Path(args.ttac_csv))
    common = sorted(set(base_rewards) & set(ttac_rewards))
    pair_delta = defaultdict(list)
    for key in common:
        pair_delta[key[0]].append(ttac_rewards[key] - base_rewards[key])
    pair_mean_delta = {k: float(np.mean(v)) for k, v in pair_delta.items()}
    pair_group = {
        k: ("beneficial" if v > 1e-6 else ("harmful" if v < -1e-6 else "neutral"))
        for k, v in pair_mean_delta.items()
    }

    data = np.load(args.dataset, allow_pickle=False)
    n = len(data["ego_action"])
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n) if n <= args.max_samples else np.sort(rng.choice(n, size=args.max_samples, replace=False))
    query_obs = np.asarray(data["query_obs"][idx], dtype=np.float32)
    partner_obs = np.asarray(data["partner_obs"][idx], dtype=np.float32)
    hist = np.asarray(data["partner_action_history"][idx], dtype=np.int32)
    raw_pairs = np.asarray(data["pair_label"][idx]).astype(str)
    roles = np.asarray(data["role"][idx], dtype=np.int32)
    timesteps = np.asarray(data["timestep"][idx], dtype=np.int32)

    policies = load_policies(Path(args.run_dir), args.history_len)
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
    estimator = load_agreement_npz(args.ttac_v5_estimator_path)
    estimator_logits = compute_estimator_logits(
        estimator,
        query_obs,
        partner_obs,
        hist,
        policies[0].action_dim,
        batch_size=args.estimator_batch_size,
    )
    base_probs = softmax_np(base_logits)
    estimator_probs = softmax_np(estimator_logits)
    delta_probs, delta_mass = delta_target(base_probs, estimator_probs, args.delta_margin, args.delta_scale)

    base_arg = np.argmax(base_probs, axis=-1)
    estimator_arg = np.argmax(estimator_probs, axis=-1)
    delta_arg = np.argmax(delta_probs, axis=-1)
    target_base_tv = 0.5 * np.abs(estimator_probs - base_probs).sum(axis=-1)
    delta_base_tv = 0.5 * np.abs(delta_probs - base_probs).sum(axis=-1)

    rows = []
    for i, cp in enumerate(cross_pairs):
        group = pair_group.get(cp, "unpaired")
        rows.append({
            "pair_label": cp,
            "pair_group": group,
            "pair_mean_delta": pair_mean_delta.get(cp, 0.0),
            "role": int(roles[i]),
            "timestep": int(timesteps[i]),
            "base_argmax": int(base_arg[i]),
            "estimator_argmax": int(estimator_arg[i]),
            "delta_argmax": int(delta_arg[i]),
            "base_argmax_name": ACTION_NAMES.get(int(base_arg[i]), str(base_arg[i])),
            "estimator_argmax_name": ACTION_NAMES.get(int(estimator_arg[i]), str(estimator_arg[i])),
            "delta_argmax_name": ACTION_NAMES.get(int(delta_arg[i]), str(delta_arg[i])),
            "base_to_estimator_argmax": transition_key(base_arg[i], estimator_arg[i]),
            "base_to_delta_argmax": transition_key(base_arg[i], delta_arg[i]),
            "estimator_changes_base": int(estimator_arg[i] != base_arg[i]),
            "delta_changes_base": int(delta_arg[i] != base_arg[i]),
            "target_base_tv": float(target_base_tv[i]),
            "delta_base_tv": float(delta_base_tv[i]),
            "delta_mass": float(delta_mass[i]),
            "base_value": float(base_values[i]),
        })

    with (out / "delta_action_transition_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    transition_rows = []
    for group in ["beneficial", "harmful", "neutral", "unpaired"]:
        transition_rows.extend(summarize_group(rows, group))
    if transition_rows:
        with (out / "delta_action_transition_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(transition_rows[0].keys()))
            writer.writeheader()
            writer.writerows(transition_rows)

    group_rows = []
    for group in ["beneficial", "harmful", "neutral", "unpaired"]:
        subset = [r for r in rows if r["pair_group"] == group]
        if not subset:
            continue
        group_rows.append({
            "group": group,
            "rows": len(subset),
            "pair_mean_delta": float(np.mean([r["pair_mean_delta"] for r in subset])),
            "delta_change_rate": float(np.mean([r["delta_changes_base"] for r in subset])),
            "estimator_change_rate": float(np.mean([r["estimator_changes_base"] for r in subset])),
            "target_base_tv": float(np.mean([r["target_base_tv"] for r in subset])),
            "delta_base_tv": float(np.mean([r["delta_base_tv"] for r in subset])),
            "delta_mass": float(np.mean([r["delta_mass"] for r in subset])),
            "base_value": float(np.mean([r["base_value"] for r in subset])),
        })
    with (out / "delta_action_group_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    md = ["# TTAC v5.4 delta action transition audit", ""]
    md.append("## Group Summary")
    md.append("")
    md.append("| group | rows | pair_delta | delta_change | estimator_change | target_base_tv | delta_base_tv | delta_mass | base_value |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in group_rows:
        md.append(
            f"| {r['group']} | {r['rows']} | {r['pair_mean_delta']:.3f} | {r['delta_change_rate']:.3f} | {r['estimator_change_rate']:.3f} | {r['target_base_tv']:.3f} | {r['delta_base_tv']:.3f} | {r['delta_mass']:.3f} | {r['base_value']:.3f} |"
        )
    md.append("")
    md.append("## Top Transitions")
    md.append("")
    md.append("| group | type | transition | count | rate |")
    md.append("|---|---|---|---:|---:|")
    for r in transition_rows[:80]:
        md.append(f"| {r['group']} | {r['transition_type']} | {r['transition']} | {r['count']} | {r['rate']:.3f} |")
    (out / "delta_action_transition_audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print(f"[delta_action_transition_audit] wrote {out}")


if __name__ == "__main__":
    main()
