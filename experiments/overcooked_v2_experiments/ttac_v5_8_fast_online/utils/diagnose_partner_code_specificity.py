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

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (  # noqa: E501
    apply_latent_partner_decoder_with_z,
    categorical_kl,
    categorical_tv,
    load_latent_decoder_npz,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.train_meta_residual_c_decoder import (  # noqa: E501
    adapt_c,
    split_support_future,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def tv_np(a, b):
    return 0.5 * np.abs(normalize_probs(a) - normalize_probs(b)).sum(axis=-1)


def kl_np(target, pred):
    target = normalize_probs(target)
    pred = normalize_probs(pred)
    return np.sum(
        target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(pred, EPS))),
        axis=-1,
    )


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den <= EPS:
        return float("nan")
    return float(np.sum(x * y) / den)


def rankdata_simple(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata_simple(x), rankdata_simple(y))


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


def select_group_rows(arrays, max_groups, max_rows_per_group, seed):
    if "query_group_id" not in arrays:
        raise ValueError("dataset must contain query_group_id")
    group_ids = np.asarray(arrays["query_group_id"], dtype=np.int64)
    unique, counts = np.unique(group_ids, return_counts=True)
    candidate_groups = unique[counts >= 2]
    if candidate_groups.size == 0:
        raise ValueError("no query groups with at least two rows")
    rng = np.random.default_rng(seed)
    if max_groups and candidate_groups.size > max_groups:
        candidate_groups = np.sort(
            rng.choice(candidate_groups, size=max_groups, replace=False)
        )
    selected = []
    selected_groups = []
    for group_id in candidate_groups.tolist():
        rows = np.where(group_ids == group_id)[0].astype(np.int32)
        if rows.size < 2:
            continue
        if max_rows_per_group and rows.size > max_rows_per_group:
            rows = np.sort(rng.choice(rows, size=max_rows_per_group, replace=False))
        selected.append(rows)
        selected_groups.append(np.full((rows.size,), group_id, dtype=np.int64))
    if not selected:
        raise ValueError("no rows selected")
    return np.concatenate(selected).astype(np.int32), np.concatenate(selected_groups)


def make_eval_batch(params, action_dim, inner_steps, inner_lr, inner_prior_coef, inner_c_clip, support_fraction):
    latent_kind = "c" if "latent_online_residual_c_enabled" in params else "z"

    @jax.jit
    def eval_batch(query_obs, support_obs, support_actions):
        adapt_obs, adapt_actions, _future_obs, _future_actions, _has_future = split_support_future(
            support_obs, support_actions, support_fraction
        )
        z_pre, z_post, c_post = adapt_c(
            params,
            adapt_obs,
            adapt_actions,
            action_dim,
            inner_steps,
            inner_lr,
            inner_prior_coef,
            inner_c_clip,
            True,
        )
        del z_pre
        logits = apply_latent_partner_decoder_with_z(
            params, query_obs, z_post, action_dim
        )
        probs = jax.nn.softmax(logits, axis=-1)
        latent = c_post if latent_kind == "c" else z_post
        return probs, logits, latent

    return eval_batch, latent_kind


def predict_codes_and_probs(params, arrays, idx, args):
    eval_batch, latent_kind = make_eval_batch(
        params,
        args.action_dim,
        args.inner_steps,
        args.inner_lr,
        args.inner_prior_coef,
        args.inner_c_clip,
        args.inner_support_fraction,
    )
    probs_parts = []
    logits_parts = []
    latent_parts = []
    for start in range(0, idx.shape[0], args.batch_size):
        part = idx[start : start + args.batch_size]
        probs, logits, latent = eval_batch(
            jnp.asarray(arrays["query_obs"][part], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs_history"][part], dtype=jnp.float32),
            jnp.asarray(arrays["partner_action_history"][part], dtype=jnp.int32),
        )
        probs_parts.append(np.asarray(probs, dtype=np.float32))
        logits_parts.append(np.asarray(logits, dtype=np.float32))
        latent_parts.append(np.asarray(latent, dtype=np.float32))
    return (
        np.concatenate(probs_parts, axis=0),
        np.concatenate(logits_parts, axis=0),
        np.concatenate(latent_parts, axis=0),
        latent_kind,
    )


def pairwise_rows_for_model(model_name, latent_kind, group_ids, target, pred, latent, partner_ids, partner_families):
    pair_rows = []
    for group_id in np.unique(group_ids).tolist():
        pos = np.where(group_ids == group_id)[0]
        if pos.size < 2:
            continue
        for local_i in range(pos.size):
            for local_j in range(local_i + 1, pos.size):
                i = int(pos[local_i])
                j = int(pos[local_j])
                teacher_tv = float(tv_np(target[i : i + 1], target[j : j + 1])[0])
                pred_tv = float(tv_np(pred[i : i + 1], pred[j : j + 1])[0])
                latent_l2 = float(np.linalg.norm(latent[i] - latent[j]))
                teacher_arg_diff = int(np.argmax(target[i]) != np.argmax(target[j]))
                pred_arg_diff = int(np.argmax(pred[i]) != np.argmax(pred[j]))
                pair_rows.append(
                    {
                        "model": model_name,
                        "latent_kind": latent_kind,
                        "query_group_id": int(group_id),
                        "partner_i": str(partner_ids[i]),
                        "partner_j": str(partner_ids[j]),
                        "family_i": str(partner_families[i]),
                        "family_j": str(partner_families[j]),
                        "same_family": int(str(partner_families[i]) == str(partner_families[j])),
                        "teacher_tv": teacher_tv,
                        "pred_tv": pred_tv,
                        "latent_l2": latent_l2,
                        "teacher_arg_diff": teacher_arg_diff,
                        "pred_arg_diff": pred_arg_diff,
                    }
                )
    return pair_rows


def summarize_pairs(model_name, latent_kind, pair_rows, target, pred, logits):
    teacher_tv = np.asarray([r["teacher_tv"] for r in pair_rows], dtype=np.float32)
    pred_tv = np.asarray([r["pred_tv"] for r in pair_rows], dtype=np.float32)
    latent_l2 = np.asarray([r["latent_l2"] for r in pair_rows], dtype=np.float32)
    teacher_arg_diff = np.asarray([r["teacher_arg_diff"] for r in pair_rows], dtype=bool)
    pred_arg_diff = np.asarray([r["pred_arg_diff"] for r in pair_rows], dtype=bool)
    high_teacher = teacher_tv >= 0.10
    low_teacher = teacher_tv < 0.05
    collapsed = high_teacher & (pred_tv < 0.02)
    oversensitive = low_teacher & (pred_tv > 0.10)
    target_arg = np.argmax(target, axis=-1)
    pred_arg = np.argmax(pred, axis=-1)
    kl = kl_np(target, pred)
    tv_to_target = tv_np(target, pred)
    return {
        "model": model_name,
        "latent_kind": latent_kind,
        "rows": int(target.shape[0]),
        "pairs": int(len(pair_rows)),
        "row_kl": float(np.mean(kl)),
        "row_tv": float(np.mean(tv_to_target)),
        "row_argmax_acc": float(np.mean(pred_arg == target_arg)),
        "mean_teacher_tv": float(np.mean(teacher_tv)),
        "mean_pred_tv": float(np.mean(pred_tv)),
        "pred_teacher_tv_ratio": float(np.mean(pred_tv) / max(float(np.mean(teacher_tv)), EPS)),
        "mean_latent_l2": float(np.mean(latent_l2)),
        "corr_teacher_pred_tv": pearson(teacher_tv, pred_tv),
        "spearman_teacher_pred_tv": spearman(teacher_tv, pred_tv),
        "corr_teacher_latent_l2": pearson(teacher_tv, latent_l2),
        "spearman_teacher_latent_l2": spearman(teacher_tv, latent_l2),
        "teacher_arg_diff_rate": float(np.mean(teacher_arg_diff)),
        "pred_arg_diff_rate": float(np.mean(pred_arg_diff)),
        "argdiff_recall": float(np.mean(pred_arg_diff[teacher_arg_diff])) if np.any(teacher_arg_diff) else float("nan"),
        "argdiff_false_positive": float(np.mean(pred_arg_diff[~teacher_arg_diff])) if np.any(~teacher_arg_diff) else float("nan"),
        "collapsed_high_teacher_fraction": float(np.mean(collapsed[high_teacher])) if np.any(high_teacher) else float("nan"),
        "oversensitive_low_teacher_fraction": float(np.mean(oversensitive[low_teacher])) if np.any(low_teacher) else float("nan"),
        "logit_std": float(np.std(logits)),
    }


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decoder", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_groups", type=int, default=5000)
    parser.add_argument("--max_rows_per_group", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=1.0)
    parser.add_argument("--inner_prior_coef", type=float, default=0.01)
    parser.add_argument("--inner_c_clip", type=float, default=3.0)
    parser.add_argument("--inner_support_fraction", type=float, default=0.5)
    parser.add_argument("--max_pair_rows_csv", type=int, default=200000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    required = [
        "query_group_id",
        "query_obs",
        "partner_obs_history",
        "partner_action_history",
        "target_partner_probs",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")

    idx, selected_group_ids = select_group_rows(
        arrays,
        args.max_groups,
        args.max_rows_per_group,
        args.seed,
    )
    target = normalize_probs(arrays["target_partner_probs"][idx])
    partner_ids = (
        np.asarray(arrays["partner_policy_id"][idx]).astype(str)
        if "partner_policy_id" in arrays
        else np.full((idx.shape[0],), "", dtype=str)
    )
    partner_families = (
        np.asarray(arrays["partner_family"][idx]).astype(str)
        if "partner_family" in arrays
        else np.full((idx.shape[0],), "", dtype=str)
    )

    summary_rows = []
    all_pair_rows = []
    for model_name, decoder_path in parse_decoder_specs(args.decoder):
        params = load_latent_decoder_npz(decoder_path)
        pred, logits, latent, latent_kind = predict_codes_and_probs(params, arrays, idx, args)
        pair_rows = pairwise_rows_for_model(
            model_name,
            latent_kind,
            selected_group_ids,
            target,
            pred,
            latent,
            partner_ids,
            partner_families,
        )
        summary_rows.append(
            summarize_pairs(model_name, latent_kind, pair_rows, target, pred, logits)
        )
        all_pair_rows.extend(pair_rows[: max(0, args.max_pair_rows_csv - len(all_pair_rows))])
        print(f"[specificity] {model_name}: {summary_rows[-1]}", flush=True)

    write_csv(out / "partner_code_specificity_summary.csv", summary_rows)
    write_csv(out / "partner_code_specificity_pairs_sample.csv", all_pair_rows)

    lines = [
        "# Partner Code Specificity Diagnostic",
        "",
        f"- dataset: `{args.dataset}`",
        f"- selected_groups: `{len(np.unique(selected_group_ids))}`",
        f"- selected_rows: `{idx.shape[0]}`",
        f"- inner_steps: `{args.inner_steps}`",
        f"- inner_support_fraction: `{args.inner_support_fraction}`",
        "",
        "| model | latent | rows | pairs | row KL | row TV | acc | teacher TV | pred TV | pred/teacher | latent L2 | corr TV | spear TV | corr latent | argdiff recall | collapsed high |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['latent_kind']} | {row['rows']} | {row['pairs']} | "
            f"{row['row_kl']:.4f} | {row['row_tv']:.4f} | {row['row_argmax_acc']:.4f} | "
            f"{row['mean_teacher_tv']:.4f} | {row['mean_pred_tv']:.4f} | "
            f"{row['pred_teacher_tv_ratio']:.3f} | {row['mean_latent_l2']:.4f} | "
            f"{row['corr_teacher_pred_tv']:.4f} | {row['spearman_teacher_pred_tv']:.4f} | "
            f"{row['corr_teacher_latent_l2']:.4f} | {row['argdiff_recall']:.4f} | "
            f"{row['collapsed_high_teacher_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metric Notes",
            "",
            "- teacher TV: same query, TV distance between two true partner policies.",
            "- pred TV: same query, TV distance between estimator outputs from two partner histories.",
            "- corr TV: Pearson correlation between teacher TV and pred TV over same-query pairs.",
            "- collapsed high: among pairs whose teacher TV >= 0.10, fraction where pred TV < 0.02.",
        ]
    )
    (out / "partner_code_specificity_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[specificity] wrote {out / 'partner_code_specificity_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
