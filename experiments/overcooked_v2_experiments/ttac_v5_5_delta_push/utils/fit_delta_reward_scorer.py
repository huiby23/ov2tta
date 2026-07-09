from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


EPS = 1e-8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _auc_score(y, score):
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Mann-Whitney AUC with average rank for ties.
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    pos_ranks = ranks[y > 0.5].sum()
    return float((pos_ranks - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j - 1)
        i = j
    return ranks


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _row_float(row, key, default=0.0):
    val = row.get(key, "")
    if val == "":
        return default
    return float(val)


def _row_int(row, key, default=0):
    val = row.get(key, "")
    if val == "":
        return default
    return int(float(val))


def build_features(rows, action_dim):
    names = [
        "target_base_tv",
        "delta_base_tv",
        "delta_mass",
        "base_value",
        "delta_changes_base",
        "estimator_changes_base",
        "delta_is_move",
        "base_is_move",
        "estimator_is_move",
        "delta_is_interact",
        "base_is_interact",
        "sticky_same",
        "base_stay_prob_proxy",
    ]
    for prefix in ("base", "estimator", "delta"):
        for action in range(action_dim):
            names.append(f"{prefix}_argmax_{action}")

    feats = []
    labels = []
    pair_delta = []
    for row in rows:
        group = row.get("pair_group", "")
        if group not in {"beneficial", "harmful"}:
            continue
        base_arg = _row_int(row, "base_argmax")
        estimator_arg = _row_int(row, "estimator_argmax")
        delta_arg = _row_int(row, "delta_argmax")
        base_oh = np.eye(action_dim, dtype=np.float32)[np.clip(base_arg, 0, action_dim - 1)]
        estimator_oh = np.eye(action_dim, dtype=np.float32)[np.clip(estimator_arg, 0, action_dim - 1)]
        delta_oh = np.eye(action_dim, dtype=np.float32)[np.clip(delta_arg, 0, action_dim - 1)]
        scalar = np.asarray(
            [
                _row_float(row, "target_base_tv"),
                _row_float(row, "delta_base_tv"),
                _row_float(row, "delta_mass"),
                _row_float(row, "base_value"),
                _row_float(row, "delta_changes_base"),
                _row_float(row, "estimator_changes_base"),
                float(delta_arg <= 3),
                float(base_arg <= 3),
                float(estimator_arg <= 3),
                float(delta_arg == 5),
                float(base_arg == 5),
                float(base_arg == 5 and delta_arg == 5),
                float(base_arg == 4),
            ],
            dtype=np.float32,
        )
        feats.append(np.concatenate([scalar, base_oh, estimator_oh, delta_oh], axis=0))
        labels.append(1.0 if group == "beneficial" else 0.0)
        pair_delta.append(_row_float(row, "pair_mean_delta"))
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.float32), np.asarray(pair_delta, dtype=np.float32), names


def fit_logistic(x, y, steps, lr, l2, seed):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = rng.normal(0.0, 0.01, size=d).astype(np.float64)
    b = 0.0
    y = y.astype(np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, 0.5 / pos, 0.5 / neg) * len(y)
    for _ in range(steps):
        logits = x @ w + b
        probs = _sigmoid(logits)
        err = (probs - y) * sample_weight
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w.astype(np.float32), np.float32(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Path(args.input_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))

    x, y, pair_delta, names = build_features(rows, args.action_dim)
    if len(x) == 0:
        raise ValueError("No beneficial/harmful rows found in input CSV.")
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-4
    z = (x - mean) / std
    w, b = fit_logistic(z, y, args.steps, args.lr, args.l2, args.seed)
    logits = z @ w + b
    score = _sigmoid(logits)
    pred = score >= 0.5

    coef_rows = []
    for name, coef in sorted(zip(names, w), key=lambda kv: abs(float(kv[1])), reverse=True):
        coef_rows.append({"feature": name, "coef": float(coef)})
    with (out / "reward_scorer_coefficients.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coef"])
        writer.writeheader()
        writer.writerows(coef_rows)

    np.savez(
        out / "reward_scorer.npz",
        coef=w.astype(np.float32),
        bias=np.asarray(b, dtype=np.float32),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        score_mean=np.asarray(score.mean(), dtype=np.float32),
        action_dim=np.asarray(args.action_dim, dtype=np.int32),
    )

    metrics = {
        "rows": float(len(x)),
        "positive_rate": float(y.mean()),
        "accuracy": float((pred == (y > 0.5)).mean()),
        "auc": _auc_score(y, score),
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_pair_delta_pearson": _pearson(score, pair_delta),
        "score_pair_delta_spearman": _spearman(score, pair_delta),
    }
    with (out / "reward_scorer_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    md = ["# TTAC v5.4 reward residual scorer", ""]
    md.append("This scorer predicts whether a v5.4 delta correction came from a pair where TTAC improved XP over base.")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---:|")
    for k, v in metrics.items():
        md.append(f"| {k} | {v:.6f} |")
    md.append("")
    md.append("## Largest coefficients")
    md.append("")
    md.append("| feature | coef |")
    md.append("|---|---:|")
    for row in coef_rows[:20]:
        md.append(f"| {row['feature']} | {row['coef']:.6f} |")
    (out / "reward_scorer_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
