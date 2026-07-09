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


def delayed_history(history):
    history = np.asarray(history, dtype=np.int32)
    return np.concatenate([np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1)


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    return normalize_probs(probs)


def metrics(target, pred):
    target = normalize_probs(target)
    pred = normalize_probs(pred)
    log_target = np.log(np.maximum(target, EPS))
    log_pred = np.log(np.maximum(pred, EPS))
    targ_arg = np.argmax(target, axis=-1)
    pred_rank = np.argsort(pred, axis=-1)
    return {
        "kl": float(np.sum(target * (log_target - log_pred), axis=-1).mean()),
        "tv": float((0.5 * np.abs(target - pred).sum(axis=-1)).mean()),
        "argmax_acc": float(np.mean(np.argmax(pred, axis=-1) == targ_arg)),
        "top2_acc": float(np.mean(np.any(pred_rank[:, -2:] == targ_arg[:, None], axis=-1))),
        "entropy": float((-np.sum(pred * log_pred, axis=-1)).mean()),
        "max_prob": float(pred.max(axis=-1).mean()),
    }


def apply_decoder(params, query_obs, obs_hist, act_hist, action_dim, batch_size):
    final_probs = []
    prior_probs = []
    residual_tvs = []
    gate_means = []
    for start in range(0, len(query_obs), batch_size):
        end = min(len(query_obs), start + batch_size)
        logits, aux = apply_latent_partner_decoder(
            params,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
            jnp.asarray(act_hist[start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=True,
        )
        final = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32)
        final_probs.append(final)
        if "prior_logits" in aux:
            prior = np.asarray(jax.nn.softmax(aux["prior_logits"], axis=-1), dtype=np.float32)
            prior_probs.append(prior)
            residual_tvs.append((0.5 * np.abs(final - prior).sum(axis=-1)).astype(np.float32))
        if "residual_gate" in aux:
            gate_means.append(np.asarray(aux["residual_gate"].mean(axis=-1), dtype=np.float32))
    final_probs = np.concatenate(final_probs, axis=0)
    prior_probs = np.concatenate(prior_probs, axis=0) if prior_probs else np.full_like(final_probs, np.nan)
    residual_tv = np.concatenate(residual_tvs, axis=0) if residual_tvs else np.full((len(final_probs),), np.nan)
    gate_mean = np.concatenate(gate_means, axis=0) if gate_means else np.full((len(final_probs),), np.nan)
    return final_probs, prior_probs, residual_tv, gate_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max_rows", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in np.load(args.dataset, allow_pickle=False).items()}
    idx = split_indices(arrays, args.val_fraction, args.seed, args.split)
    if args.max_rows and len(idx) > args.max_rows:
        rng = np.random.default_rng(args.seed + 17)
        idx = rng.choice(idx, size=args.max_rows, replace=False).astype(np.int32)
    idx = np.sort(idx)

    rng = np.random.default_rng(args.seed + 99)
    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    target = normalize_probs(arrays["target_partner_probs"][idx])
    mean_teacher = normalize_probs(arrays.get("mean_teacher_probs", arrays["target_partner_probs"])[idx])
    true_obs = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    true_act = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    histories = {
        "true": (true_obs, true_act),
        "delayed": (true_obs, delayed_history(true_act)),
        "random_actions": (
            true_obs,
            rng.integers(0, args.action_dim, size=true_act.shape, dtype=np.int32),
        ),
        "no_history": (
            np.zeros_like(true_obs, dtype=np.float32),
            np.zeros_like(true_act, dtype=np.int32),
        ),
    }

    params = load_latent_decoder_npz(args.decoder)
    rows = []
    outputs = {}
    for name, (obs_hist, act_hist) in histories.items():
        final, prior, residual_tv, gate_mean = apply_decoder(
            params, query_obs, obs_hist, act_hist, args.action_dim, args.batch_size
        )
        outputs[name] = (final, prior)
        row = {"history": name, "head": "final", "n": int(len(idx))}
        row.update(metrics(target, final))
        row["residual_tv_to_prior"] = float(np.nanmean(residual_tv))
        row["gate_mean"] = float(np.nanmean(gate_mean))
        rows.append(row)
        if name == "true":
            prior_target = {"history": "query_prior", "head": "prior_vs_target", "n": int(len(idx))}
            prior_target.update(metrics(target, prior))
            prior_target["residual_tv_to_prior"] = 0.0
            prior_target["gate_mean"] = float("nan")
            rows.append(prior_target)
            prior_mean = {"history": "query_prior", "head": "prior_vs_mean_teacher", "n": int(len(idx))}
            prior_mean.update(metrics(mean_teacher, prior))
            prior_mean["residual_tv_to_prior"] = 0.0
            prior_mean["gate_mean"] = float("nan")
            rows.append(prior_mean)

    final_nohist, prior_nohist = outputs["no_history"]
    true_final, true_prior = outputs["true"]
    summary = {
        "n": int(len(idx)),
        "prior_vs_target_kl": rows[1]["kl"],
        "prior_vs_target_tv": rows[1]["tv"],
        "prior_vs_target_acc": rows[1]["argmax_acc"],
        "prior_vs_mean_teacher_kl": rows[2]["kl"],
        "true_final_kl": rows[0]["kl"],
        "true_final_tv": rows[0]["tv"],
        "true_final_acc": rows[0]["argmax_acc"],
        "no_history_kl_minus_true": rows[5]["kl"] - rows[0]["kl"],
        "random_kl_minus_true": rows[4]["kl"] - rows[0]["kl"],
        "delayed_kl_minus_true": rows[3]["kl"] - rows[0]["kl"],
        "true_residual_tv_to_prior": rows[0]["residual_tv_to_prior"],
        "no_history_final_vs_prior_tv": float((0.5 * np.abs(final_nohist - prior_nohist).sum(axis=-1)).mean()),
        "true_final_vs_prior_tv": float((0.5 * np.abs(true_final - true_prior).sum(axis=-1)).mean()),
    }

    with (out / "prior_residual_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "prior_residual_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    lines = [
        "# Prior-Residual Decoder Diagnostic",
        "",
        f"- dataset: `{args.dataset}`",
        f"- decoder: `{args.decoder}`",
        f"- split: `{args.split}`",
        f"- rows: `{len(idx)}`",
        "",
        "| history | head | KL | TV | argmax acc | top2 acc | entropy | max prob | residual TV to prior | gate mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['history']} | {row['head']} | {row['kl']:.4f} | {row['tv']:.4f} | "
            f"{row['argmax_acc']:.4f} | {row['top2_acc']:.4f} | {row['entropy']:.4f} | "
            f"{row['max_prob']:.4f} | {row['residual_tv_to_prior']:.4f} | {row['gate_mean']:.4f} |"
        )
    lines.extend(["", "## Summary", ""])
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    (out / "prior_residual_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
