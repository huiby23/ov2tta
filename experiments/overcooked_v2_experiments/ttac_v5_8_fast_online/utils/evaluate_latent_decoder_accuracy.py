#!/usr/bin/env python3
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


def delayed_history(history):
    history = np.asarray(history, dtype=np.int32)
    return np.concatenate([np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1)


def split_indices(arrays, val_fraction, seed, split):
    n = arrays["query_obs"].shape[0]
    if split == "all":
        return np.arange(n, dtype=np.int32)
    if "episode_id" not in arrays:
        rng = np.random.default_rng(seed)
        idx = np.arange(n, dtype=np.int32)
        rng.shuffle(idx)
        cut = max(1, int(n * val_fraction))
        return idx[:cut] if split == "val" else idx[cut:]
    episode_ids = np.asarray(arrays["episode_id"])
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    val_eps = set(unique_eps[: max(1, int(len(unique_eps) * val_fraction))].tolist())
    val_mask = np.isin(episode_ids, list(val_eps))
    return np.where(val_mask if split == "val" else ~val_mask)[0].astype(np.int32)


def history_for_mode(arrays, idx, mode):
    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    if mode == "action_only":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
    elif mode == "no_history":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
        act_hist = np.zeros_like(act_hist, dtype=np.int32)
    return obs_hist, act_hist


def control_histories(arrays, idx, history_mode, action_dim, seed):
    rng = np.random.default_rng(seed)
    true_obs, true_act = history_for_mode(arrays, idx, history_mode)
    wrong_obs = true_obs
    if "ego_action_history" in arrays:
        wrong_act = np.asarray(arrays["ego_action_history"][idx], dtype=np.int32)
    else:
        wrong_act = np.zeros_like(true_act, dtype=np.int32)
    if history_mode == "no_history":
        wrong_obs = np.zeros_like(wrong_obs, dtype=np.float32)
        wrong_act = np.zeros_like(wrong_act, dtype=np.int32)
    elif history_mode == "action_only":
        wrong_obs = np.zeros_like(wrong_obs, dtype=np.float32)
    no_obs = np.zeros_like(true_obs, dtype=np.float32)
    no_act = np.zeros_like(true_act, dtype=np.int32)
    return {
        "true": (true_obs, true_act),
        "wrong": (wrong_obs, wrong_act),
        "delayed": (true_obs, delayed_history(true_act)),
        "random": (
            true_obs,
            rng.integers(0, action_dim, size=true_act.shape, dtype=np.int32),
        ),
        "no_history": (no_obs, no_act),
    }


def parse_decoder_specs(specs):
    parsed = []
    for raw in specs:
        if "=" in raw:
            name, paths = raw.split("=", 1)
        else:
            paths = raw
            name = Path(paths.split("+")[0]).parent.name or Path(paths.split("+")[0]).stem
        path_list = [Path(p) for p in paths.split("+") if p]
        if not path_list:
            raise ValueError(f"empty decoder spec: {raw}")
        parsed.append((name, path_list))
    return parsed


def apply_decoder_members(params_list, query_obs, obs_hist, act_hist, action_dim, batch_size):
    member_probs = [[] for _ in params_list]
    for start in range(0, len(query_obs), batch_size):
        end = min(len(query_obs), start + batch_size)
        q = jnp.asarray(query_obs[start:end], dtype=jnp.float32)
        oh = jnp.asarray(obs_hist[start:end], dtype=jnp.float32)
        ah = jnp.asarray(act_hist[start:end], dtype=jnp.int32)
        for member_idx, params in enumerate(params_list):
            result = apply_latent_partner_decoder(
                params,
                q,
                oh,
                ah,
                action_dim,
                deterministic=True,
                return_aux=False,
            )
            member_probs[member_idx].append(np.asarray(jax.nn.softmax(result, axis=-1), dtype=np.float32))
    stacked = np.stack([np.concatenate(parts, axis=0) for parts in member_probs], axis=0)
    return stacked


def ensemble_uncertainty(member_probs):
    if member_probs.shape[0] <= 1:
        return np.zeros((member_probs.shape[1],), dtype=np.float32)
    tvs = []
    for i in range(member_probs.shape[0]):
        for j in range(i + 1, member_probs.shape[0]):
            tvs.append(0.5 * np.abs(member_probs[i] - member_probs[j]).sum(axis=-1))
    return np.mean(np.stack(tvs, axis=0), axis=0).astype(np.float32)


def per_row_metrics(target, pred, uncertainty):
    target = normalize_probs(target)
    pred = normalize_probs(pred)
    log_pred = np.log(np.maximum(pred, EPS))
    log_target = np.log(np.maximum(target, EPS))
    teacher_argmax = np.argmax(target, axis=-1)
    pred_order = np.argsort(pred, axis=-1)
    return {
        "ce": -np.sum(target * log_pred, axis=-1),
        "kl": np.sum(target * (log_target - log_pred), axis=-1),
        "tv": 0.5 * np.abs(target - pred).sum(axis=-1),
        "argmax_acc": (np.argmax(pred, axis=-1) == teacher_argmax).astype(np.float32),
        "top2_acc": np.any(pred_order[:, -2:] == teacher_argmax[:, None], axis=-1).astype(np.float32),
        "pred_entropy": -np.sum(pred * log_pred, axis=-1),
        "teacher_entropy": -np.sum(target * log_target, axis=-1),
        "uncertainty": uncertainty,
    }


def high_mask(values, quantile):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0 or np.all(~np.isfinite(values)):
        return np.zeros_like(values, dtype=bool)
    threshold = np.nanquantile(values, quantile)
    return values >= threshold


def build_slices(arrays, idx, target, baseline_kl, quantile, low_disagreement_threshold):
    masks = {"global": np.ones((len(idx),), dtype=bool)}
    if "query_disagreement_tv" in arrays:
        disagreement = np.asarray(arrays["query_disagreement_tv"][idx], dtype=np.float32)
        masks["high_partner_disagreement"] = high_mask(disagreement, quantile)
        masks["low_partner_disagreement"] = disagreement < float(low_disagreement_threshold)
    if "base_probs" in arrays:
        base_tv = 0.5 * np.abs(normalize_probs(arrays["base_probs"][idx]) - target).sum(axis=-1)
        masks["high_base_teacher_tv"] = high_mask(base_tv, quantile)
    elif "base_teacher_tv" in arrays:
        masks["high_base_teacher_tv"] = high_mask(arrays["base_teacher_tv"][idx], quantile)
    teacher_entropy = -np.sum(target * np.log(np.maximum(target, EPS)), axis=-1)
    masks["high_teacher_entropy"] = high_mask(teacher_entropy, quantile)
    if baseline_kl is not None:
        masks["high_baseline_error"] = high_mask(baseline_kl, quantile)
    return {name: mask for name, mask in masks.items() if np.any(mask)}


def summarize_metric_rows(estimator, history_name, slice_name, mask, metrics):
    row = {
        "estimator": estimator,
        "history": history_name,
        "slice": slice_name,
        "n": int(np.sum(mask)),
    }
    for key, values in metrics.items():
        row[key] = float(np.mean(values[mask])) if np.any(mask) else float("nan")
    return row


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
    parser.add_argument("--decoder", action="append", required=True, help="name=path or name=path1+path2+path3")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_decoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--max_rows", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--history_mode", choices=("full", "action_only", "no_history"), default="full")
    parser.add_argument("--slice_quantile", type=float, default=0.8)
    parser.add_argument("--low_disagreement_threshold", type=float, default=0.10)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    required = ["query_obs", "partner_obs_history", "partner_action_history", "target_partner_probs"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")

    idx = split_indices(arrays, args.val_fraction, args.seed, args.split)
    if args.max_rows and len(idx) > args.max_rows:
        rng = np.random.default_rng(args.seed + 17)
        idx = rng.choice(idx, size=args.max_rows, replace=False).astype(np.int32)
    idx = np.sort(idx)

    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    target = normalize_probs(arrays["target_partner_probs"][idx])
    histories = control_histories(arrays, idx, args.history_mode, args.action_dim, args.seed + 99)

    baseline_kl = None
    if args.baseline_decoder:
        baseline_params = [load_latent_decoder_npz(Path(args.baseline_decoder))]
        true_obs, true_act = histories["true"]
        baseline_members = apply_decoder_members(
            baseline_params, query_obs, true_obs, true_act, args.action_dim, args.batch_size
        )
        baseline_pred = baseline_members.mean(axis=0)
        baseline_kl = per_row_metrics(
            target, baseline_pred, np.zeros((len(idx),), dtype=np.float32)
        )["kl"]

    slices = build_slices(
        arrays,
        idx,
        target,
        baseline_kl,
        args.slice_quantile,
        args.low_disagreement_threshold,
    )
    rows = []
    summary_rows = []
    for estimator_name, decoder_paths in parse_decoder_specs(args.decoder):
        params_list = [load_latent_decoder_npz(path) for path in decoder_paths]
        global_history = {}
        for history_name, (obs_hist, act_hist) in histories.items():
            members = apply_decoder_members(
                params_list, query_obs, obs_hist, act_hist, args.action_dim, args.batch_size
            )
            pred = members.mean(axis=0)
            uncertainty = ensemble_uncertainty(members)
            metrics = per_row_metrics(target, pred, uncertainty)
            for slice_name, mask in slices.items():
                rows.append(summarize_metric_rows(estimator_name, history_name, slice_name, mask, metrics))
            global_history[history_name] = summarize_metric_rows(
                estimator_name, history_name, "global", slices["global"], metrics
            )
        true_row = global_history["true"]
        summary = {
            "estimator": estimator_name,
            "decoder_count": len(params_list),
            "split": args.split,
            "n": int(len(idx)),
            "true_kl": true_row["kl"],
            "true_tv": true_row["tv"],
            "true_argmax_acc": true_row["argmax_acc"],
            "true_top2_acc": true_row["top2_acc"],
            "wrong_kl_minus_true": global_history["wrong"]["kl"] - true_row["kl"],
            "delayed_kl_minus_true": global_history["delayed"]["kl"] - true_row["kl"],
            "random_kl_minus_true": global_history["random"]["kl"] - true_row["kl"],
            "no_history_kl_minus_true": global_history["no_history"]["kl"] - true_row["kl"],
        }
        for slice_name in slices:
            if slice_name == "global":
                continue
            match = [
                row for row in rows
                if row["estimator"] == estimator_name and row["history"] == "true" and row["slice"] == slice_name
            ][0]
            summary[f"{slice_name}_kl"] = match["kl"]
            summary[f"{slice_name}_argmax_acc"] = match["argmax_acc"]
        summary_rows.append(summary)

    write_csv(out / "accuracy_rows.csv", rows)
    write_csv(out / "accuracy_summary.csv", summary_rows)
    lines = ["# Latent Decoder Accuracy Leaderboard", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- rows: `{len(idx)}`")
    lines.append(f"- baseline decoder: `{args.baseline_decoder}`")
    lines.append("")
    lines.append("| estimator | count | true KL | true TV | argmax acc | top2 acc | wrong gap | random gap | delayed gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary_rows, key=lambda x: (x["true_kl"], -x["true_argmax_acc"])):
        lines.append(
            f"| {row['estimator']} | {row['decoder_count']} | {row['true_kl']:.4f} | "
            f"{row['true_tv']:.4f} | {row['true_argmax_acc']:.4f} | {row['true_top2_acc']:.4f} | "
            f"{row['wrong_kl_minus_true']:.4f} | {row['random_kl_minus_true']:.4f} | "
            f"{row['delayed_kl_minus_true']:.4f} |"
        )
    (out / "accuracy_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'accuracy_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
