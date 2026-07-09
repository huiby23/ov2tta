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


def summarize(subset, variant, partner_method, mask, target, pred):
    if not np.any(mask):
        return None
    target_m = target[mask]
    pred_m = pred[mask]
    target_argmax = np.argmax(target_m, axis=-1)
    pred_argmax = np.argmax(pred_m, axis=-1)
    pred_rank = np.argsort(pred_m, axis=-1)
    return {
        "subset": subset,
        "variant": variant,
        "partner_method": partner_method,
        "n": int(np.sum(mask)),
        "kl": float(kl_target_pred(target_m, pred_m).mean()),
        "tv": float(tv(target_m, pred_m).mean()),
        "argmax_acc": float(np.mean(pred_argmax == target_argmax)),
        "top2_acc": float(np.mean(np.any(pred_rank[:, -2:] == target_argmax[:, None], axis=-1))),
        "entropy": float(entropy(pred_m).mean()),
        "max_prob": float(pred_m.max(axis=-1).mean()),
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
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in np.load(args.dataset, allow_pickle=False).items()}
    idx = split_indices(arrays, args.val_fraction, args.split_seed, args.split)

    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    target = normalize_probs(arrays["target_partner_probs"][idx])
    true_obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    true_act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    partner_family = np.asarray(arrays["partner_family"])[idx].astype(str)
    ego_policy_id = np.asarray(arrays["ego_policy_id"])[idx].astype(str)

    rng = np.random.default_rng(args.seed)
    variants = []
    variants.append(("true_history", true_obs_hist, true_act_hist))

    perm = rng.permutation(len(true_obs_hist))
    variants.append(("shuffled_row_history", true_obs_hist[perm], true_act_hist[perm]))

    variants.append(
        (
            "gaussian_std1_obs_uniform_actions",
            rng.normal(0.0, 1.0, size=true_obs_hist.shape).astype(np.float32),
            rng.integers(0, args.action_dim, size=true_act_hist.shape, dtype=np.int32),
        )
    )

    obs_mean = float(true_obs_hist.mean())
    obs_std = float(true_obs_hist.std())
    if obs_std <= 1e-6:
        obs_std = 1.0
    variants.append(
        (
            "gaussian_matched_obs_empirical_actions",
            rng.normal(obs_mean, obs_std, size=true_obs_hist.shape).astype(np.float32),
            rng.choice(true_act_hist.reshape(-1), size=true_act_hist.shape, replace=True).astype(np.int32),
        )
    )

    variants.append(
        (
            "zero_obs_uniform_actions",
            np.zeros_like(true_obs_hist, dtype=np.float32),
            rng.integers(0, args.action_dim, size=true_act_hist.shape, dtype=np.int32),
        )
    )

    params = load_latent_decoder_npz(args.decoder)
    predictions = {
        name: predict_decoder_probs(params, query_obs, obs_hist, act_hist, args.action_dim, args.batch_size)
        for name, obs_hist, act_hist in variants
    }

    subset_masks = {
        "all_val": np.ones(len(idx), dtype=bool),
        "ppo_run0_1_val": np.isin(ego_policy_id, ["ppo:run_0", "ppo:run_1"]),
    }

    rows = []
    for subset, subset_mask in subset_masks.items():
        for variant, pred in predictions.items():
            for method in ["all_q_partners"] + sorted(set(partner_family.tolist())):
                method_mask = np.ones(len(idx), dtype=bool) if method == "all_q_partners" else partner_family == method
                row = summarize(subset, variant, method, subset_mask & method_mask, target, pred)
                if row is not None:
                    rows.append(row)

    write_csv(out / "noise_history_distance.csv", rows)

    lines = [
        "# Q-trained VAE Noise History Distance",
        "",
        f"- dataset: `{args.dataset}`",
        f"- decoder: `{args.decoder}`",
        f"- split: `{args.split}`",
        f"- split seed: `{args.split_seed}`",
        f"- noise seed: `{args.seed}`",
        "",
    ]
    for subset in subset_masks:
        lines.extend(
            [
                f"## {subset}",
                "",
                "| variant | partner | n | KL | TV | argmax acc | top2 acc | entropy | max prob |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["subset"] != subset:
                continue
            lines.append(
                f"| {row['variant']} | {row['partner_method']} | {row['n']} | "
                f"{row['kl']:.4f} | {row['tv']:.4f} | {row['argmax_acc']:.4f} | "
                f"{row['top2_acc']:.4f} | {row['entropy']:.4f} | {row['max_prob']:.4f} |"
            )
        lines.append("")
    (out / "noise_history_distance_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
