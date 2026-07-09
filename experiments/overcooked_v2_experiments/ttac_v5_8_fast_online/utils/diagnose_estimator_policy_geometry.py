from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.collect_pg_mixed_latent_decoder_dataset import (  # noqa: E501
    load_ppo_pool,
    parse_policy_indices,
    policy_probs_on_query_obs,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (
    apply_latent_partner_decoder,
    load_latent_decoder_npz,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def tv(a, b):
    return 0.5 * np.abs(normalize_probs(a) - normalize_probs(b)).sum(axis=-1)


def kl_target_pred(target, pred):
    target = normalize_probs(target)
    pred = normalize_probs(pred)
    return np.sum(
        target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(pred, EPS))),
        axis=-1,
    )


def entropy(probs):
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)


def split_indices(arrays, val_fraction, seed, split):
    n = arrays["query_obs"].shape[0]
    if split == "all":
        return np.arange(n, dtype=np.int32)
    episode_ids = np.asarray(arrays["episode_id"])
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    val_eps = set(unique_eps[: max(1, int(len(unique_eps) * val_fraction))].tolist())
    val_mask = np.isin(episode_ids, list(val_eps))
    return np.where(val_mask if split == "val" else ~val_mask)[0].astype(np.int32)


def parse_decoder_specs(specs):
    parsed = []
    for raw in specs:
        if "=" in raw:
            name, path = raw.split("=", 1)
        else:
            path = raw
            name = Path(path).parent.name or Path(path).stem
        parsed.append((name, Path(path)))
    return parsed


def predict_decoder_probs(params, query_obs, obs_hist, act_hist, action_dim, batch_size):
    probs = []
    for start in range(0, len(query_obs), batch_size):
        end = min(len(query_obs), start + batch_size)
        logits = apply_latent_partner_decoder(
            params,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
            jnp.asarray(act_hist[start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        probs.append(np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32))
    return np.concatenate(probs, axis=0)


def direct_blend_probs(base_probs, target_probs, alpha, clip):
    base_probs = normalize_probs(base_probs)
    target_probs = normalize_probs(target_probs)
    delta = np.log(np.maximum(target_probs, EPS)) - np.log(np.maximum(base_probs, EPS))
    logits = np.log(np.maximum(base_probs, EPS)) + float(alpha) * np.clip(delta, -float(clip), float(clip))
    logits = logits - logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=-1, keepdims=True), EPS)


def summarize(estimator, partner_method, mask, target, base_probs, estimator_probs, blend_probs):
    if not np.any(mask):
        return None
    base_tv = tv(target[mask], base_probs[mask])
    blend_tv = tv(target[mask], blend_probs[mask])
    est_tv = tv(target[mask], estimator_probs[mask])
    base_kl = kl_target_pred(target[mask], base_probs[mask])
    blend_kl = kl_target_pred(target[mask], blend_probs[mask])
    est_kl = kl_target_pred(target[mask], estimator_probs[mask])
    target_argmax = np.argmax(target[mask], axis=-1)
    base_argmax = np.argmax(base_probs[mask], axis=-1)
    blend_argmax = np.argmax(blend_probs[mask], axis=-1)
    est_argmax = np.argmax(estimator_probs[mask], axis=-1)
    return {
        "estimator": estimator,
        "partner_method": partner_method,
        "n": int(np.sum(mask)),
        "base_tv": float(base_tv.mean()),
        "estimator_tv": float(est_tv.mean()),
        "blend_tv": float(blend_tv.mean()),
        "blend_minus_base_tv": float((blend_tv - base_tv).mean()),
        "tv_shrink_fraction": float(np.mean(blend_tv < base_tv)),
        "base_kl": float(base_kl.mean()),
        "estimator_kl": float(est_kl.mean()),
        "blend_kl": float(blend_kl.mean()),
        "blend_minus_base_kl": float((blend_kl - base_kl).mean()),
        "kl_shrink_fraction": float(np.mean(blend_kl < base_kl)),
        "base_argmax_acc": float(np.mean(base_argmax == target_argmax)),
        "estimator_argmax_acc": float(np.mean(est_argmax == target_argmax)),
        "blend_argmax_acc": float(np.mean(blend_argmax == target_argmax)),
        "estimator_entropy": float(entropy(estimator_probs[mask]).mean()),
        "blend_entropy": float(entropy(blend_probs[mask]).mean()),
        "base_entropy": float(entropy(base_probs[mask]).mean()),
        "target_entropy": float(entropy(target[mask]).mean()),
        "base_estimator_tv": float(tv(base_probs[mask], estimator_probs[mask]).mean()),
        "base_blend_tv": float(tv(base_probs[mask], blend_probs[mask]).mean()),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decoder", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--ego_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument("--ego_policy_indices", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    idx = split_indices(arrays, args.val_fraction, args.seed, args.split)
    ego_ids = np.asarray(arrays["ego_policy_id"]).astype(str)
    allowed_ids = {f"ppo:run_{i}" for i in parse_policy_indices(args.ego_policy_indices)}
    idx = idx[np.asarray([ego_ids[i] in allowed_ids for i in idx], dtype=bool)]
    if args.max_rows and len(idx) > args.max_rows:
        rng = np.random.default_rng(args.seed + 17)
        idx = np.sort(rng.choice(idx, size=int(args.max_rows), replace=False)).astype(np.int32)
    if len(idx) == 0:
        raise ValueError("no rows left after PPO ego filter")

    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    target = normalize_probs(arrays["target_partner_probs"][idx])
    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    partner_family = np.asarray(arrays["partner_family"])[idx].astype(str)
    ego_ids_sel = ego_ids[idx]

    ppo_entries, _config = load_ppo_pool(
        args.ego_run_dir,
        max_policies=None,
        stochastic=True,
        pool_name="ppo",
        policy_indices=parse_policy_indices(args.ego_policy_indices),
    )
    ego_policies = {f"{entry.pool}:{entry.label}": entry.policy for entry in ppo_entries}
    base_probs = np.zeros_like(target, dtype=np.float32)
    for policy_id, policy in ego_policies.items():
        rows = np.where(ego_ids_sel == policy_id)[0]
        if len(rows):
            base_probs[rows] = policy_probs_on_query_obs(
                policy,
                query_obs[rows],
                batch_size=args.batch_size,
            )
    if np.any(base_probs.sum(axis=-1) <= 0):
        raise ValueError("some selected rows did not receive base probabilities")
    base_probs = normalize_probs(base_probs)

    rows = []
    for name, path in parse_decoder_specs(args.decoder):
        params = load_latent_decoder_npz(path)
        est_probs = predict_decoder_probs(
            params,
            query_obs,
            obs_hist,
            act_hist,
            args.action_dim,
            args.batch_size,
        )
        blend = direct_blend_probs(base_probs, est_probs, args.alpha, args.clip)
        for method in ["all_q_partners"] + sorted(set(partner_family.tolist())):
            mask = np.ones((len(idx),), dtype=bool) if method == "all_q_partners" else partner_family == method
            row = summarize(name, method, mask, target, base_probs, est_probs, blend)
            if row is not None:
                rows.append(row)

    write_csv(out / "policy_geometry_summary.csv", rows)
    lines = [
        "# Estimator Policy Geometry",
        "",
        f"- dataset: `{args.dataset}`",
        f"- split: `{args.split}`",
        f"- rows after PPO ego filter: `{len(idx)}`",
        f"- ego ids: `{sorted(set(ego_ids_sel.tolist()))}`",
        f"- alpha: `{args.alpha}`",
        f"- clip: `{args.clip}`",
        "",
        "| estimator | partner | n | base TV | estimator TV | blend TV | ΔTV | shrink% | base KL | estimator KL | blend KL | ΔKL | KL shrink% | base acc | estimator acc | blend acc | est entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['estimator']} | {row['partner_method']} | {row['n']} | "
            f"{row['base_tv']:.4f} | {row['estimator_tv']:.4f} | {row['blend_tv']:.4f} | "
            f"{row['blend_minus_base_tv']:.4f} | {100.0 * row['tv_shrink_fraction']:.1f}% | "
            f"{row['base_kl']:.4f} | {row['estimator_kl']:.4f} | {row['blend_kl']:.4f} | "
            f"{row['blend_minus_base_kl']:.4f} | {100.0 * row['kl_shrink_fraction']:.1f}% | "
            f"{row['base_argmax_acc']:.4f} | {row['estimator_argmax_acc']:.4f} | "
            f"{row['blend_argmax_acc']:.4f} | {row['estimator_entropy']:.4f} |"
        )
    (out / "policy_geometry_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'policy_geometry_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
