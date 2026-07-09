#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def history_for_mode(arrays, idx, mode):
    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    if mode == "action_only":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
    elif mode == "no_history":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
        act_hist = np.zeros_like(act_hist, dtype=np.int32)
    return obs_hist, act_hist


def baseline_kl(params, arrays, idx, action_dim, history_mode, batch_size):
    target_all = normalize_probs(arrays["target_partner_probs"][idx])
    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    obs_hist, act_hist = history_for_mode(arrays, idx, history_mode)
    out = []
    for start in range(0, len(idx), batch_size):
        end = min(len(idx), start + batch_size)
        logits = apply_latent_partner_decoder(
            params,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
            jnp.asarray(act_hist[start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        pred = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32)
        target = target_all[start:end]
        out.append(
            np.sum(
                target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(pred, EPS))),
                axis=-1,
            ).astype(np.float32)
        )
    return np.concatenate(out, axis=0)


def sample_bucket(rng, candidates, size):
    candidates = np.asarray(candidates, dtype=np.int32)
    if size <= 0:
        return np.zeros((0,), dtype=np.int32)
    if len(candidates) == 0:
        raise ValueError("empty sampling bucket")
    return rng.choice(candidates, size=size, replace=len(candidates) < size).astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--baseline_decoder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--history_mode", choices=("full", "action_only", "no_history"), default="full")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_rows", type=int, default=120000)
    parser.add_argument("--normal_fraction", type=float, default=0.5)
    parser.add_argument("--hard_fraction", type=float, default=0.3)
    parser.add_argument("--medium_fraction", type=float, default=0.2)
    parser.add_argument("--hard_quantile", type=float, default=0.7)
    parser.add_argument("--medium_low_quantile", type=float, default=0.45)
    parser.add_argument("--medium_high_quantile", type=float, default=0.7)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.source_dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    required = ["query_obs", "partner_obs_history", "partner_action_history", "target_partner_probs"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")

    n = arrays["query_obs"].shape[0]
    all_idx = np.arange(n, dtype=np.int32)
    params = load_latent_decoder_npz(Path(args.baseline_decoder))
    kl = baseline_kl(params, arrays, all_idx, args.action_dim, args.history_mode, args.batch_size)

    hard_thr = float(np.quantile(kl, args.hard_quantile))
    med_lo = float(np.quantile(kl, args.medium_low_quantile))
    med_hi = float(np.quantile(kl, args.medium_high_quantile))
    hard_candidates = all_idx[kl >= hard_thr]
    medium_candidates = all_idx[(kl >= med_lo) & (kl < med_hi)]
    normal_candidates = all_idx[kl < hard_thr]

    max_rows = min(int(args.max_rows), n)
    total_fraction = max(args.normal_fraction + args.hard_fraction + args.medium_fraction, EPS)
    n_hard = int(round(max_rows * args.hard_fraction / total_fraction))
    n_medium = int(round(max_rows * args.medium_fraction / total_fraction))
    n_normal = max_rows - n_hard - n_medium
    rng = np.random.default_rng(args.seed)
    chosen_normal = sample_bucket(rng, normal_candidates, n_normal)
    chosen_medium = sample_bucket(rng, medium_candidates, n_medium)
    chosen_hard = sample_bucket(rng, hard_candidates, n_hard)
    chosen = np.concatenate([chosen_normal, chosen_medium, chosen_hard], axis=0)
    buckets = np.concatenate([
        np.full((len(chosen_normal),), "normal", dtype="<U8"),
        np.full((len(chosen_medium),), "medium", dtype="<U8"),
        np.full((len(chosen_hard),), "hard", dtype="<U8"),
    ])
    order = rng.permutation(len(chosen))
    chosen = chosen[order]
    buckets = buckets[order]

    saved = {k: v[chosen] for k, v in arrays.items()}
    saved["source_row"] = chosen.astype(np.int32)
    saved["baseline_true_kl"] = kl[chosen].astype(np.float32)
    saved["accuracy_bucket"] = buckets
    output_path = out / "accuracy_hard_latent_decoder_dataset.npz"
    np.savez(output_path, **saved)

    lines = ["# Accuracy Hard-Query Dataset", ""]
    lines.append(f"- source_dataset: `{args.source_dataset}`")
    lines.append(f"- baseline_decoder: `{args.baseline_decoder}`")
    lines.append(f"- output: `{output_path}`")
    lines.append(f"- rows: `{len(chosen)}`")
    lines.append(f"- baseline KL mean: `{float(np.mean(kl)):.6f}`")
    lines.append(f"- hard threshold q{args.hard_quantile}: `{hard_thr:.6f}`")
    lines.append(f"- sampled normal/medium/hard: `{len(chosen_normal)}` / `{len(chosen_medium)}` / `{len(chosen_hard)}`")
    (out / "accuracy_hard_dataset_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
