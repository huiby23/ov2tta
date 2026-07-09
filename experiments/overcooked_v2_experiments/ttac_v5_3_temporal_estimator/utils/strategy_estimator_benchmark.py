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

from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.agreement_heads import (
    apply_agreement_estimator,
    categorical_kl,
    categorical_tv,
    load_agreement_npz,
)
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.collect_agreement_dataset import (
    make_policies,
    policy_probs_on_query_obs,
)


EPS = 1e-8


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def entropy(probs):
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)


def total_variation(lhs, rhs):
    lhs = normalize_probs(lhs)
    rhs = normalize_probs(rhs)
    return 0.5 * np.sum(np.abs(lhs - rhs), axis=-1)


def parse_pair_label(label):
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    left, right = str(label).split("x")
    return left, right


def build_ego_policy_keys(pair_labels, roles):
    ego_keys = []
    target_keys = []
    for label, role in zip(pair_labels, roles):
        left, right = parse_pair_label(label)
        if int(role) == 0:
            ego_keys.append(left)
            target_keys.append(right)
        else:
            ego_keys.append(right)
            target_keys.append(left)
    return np.asarray(ego_keys), np.asarray(target_keys)


def compute_ego_policy_probs(dataset, run_dir, batch_size):
    run_keys, policies, _config = make_policies(Path(run_dir))
    policy_by_key = {key: policy for key, policy in zip(run_keys, policies)}
    ego_keys, target_keys = build_ego_policy_keys(dataset["pair_label"], dataset["role"])
    query_obs = np.asarray(dataset["query_obs"])
    out = np.zeros((len(query_obs), 6), dtype=np.float32)
    for key in sorted(set(ego_keys.tolist())):
        idx = np.where(ego_keys == key)[0]
        if key not in policy_by_key:
            raise KeyError(f"ego policy key {key!r} not found in run_dir policies {run_keys}")
        print(f"[strategy_estimator] ego teacher probs for {key}: {len(idx)} samples", flush=True)
        out[idx] = policy_probs_on_query_obs(policy_by_key[key], query_obs[idx], batch_size=batch_size)
    return out, ego_keys, target_keys


def shifted_right(actions, fill_value=0):
    actions = np.asarray(actions, dtype=np.int32)
    return np.concatenate(
        [np.full((actions.shape[0], 1), fill_value, dtype=np.int32), actions[:, :-1]],
        axis=1,
    )


def evaluate_mode(params, dataset, history, action_dim, batch_size):
    target = normalize_probs(dataset["target_partner_probs"])
    kl_parts = []
    tv_parts = []
    acc_parts = []
    pred_entropy_parts = []
    target_argmax = np.argmax(target, axis=-1)
    for start in range(0, len(target), batch_size):
        end = min(len(target), start + batch_size)
        logits = apply_agreement_estimator(
            params,
            jnp.asarray(dataset["query_obs"][start:end], dtype=jnp.float32),
            jnp.asarray(dataset["partner_obs"][start:end], dtype=jnp.float32),
            jnp.asarray(history[start:end], dtype=jnp.int32),
            action_dim,
            partner_obs_history=jnp.asarray(dataset["partner_obs_history"][start:end], dtype=jnp.float32),
        )
        logits_np = np.asarray(logits)
        probs_np = np.asarray(jax.nn.softmax(logits, axis=-1))
        kl_parts.append(np.asarray(categorical_kl(target[start:end], logits)))
        tv_parts.append(np.asarray(categorical_tv(target[start:end], logits)))
        acc_parts.append((np.argmax(logits_np, axis=-1) == target_argmax[start:end]).astype(np.float32))
        pred_entropy_parts.append(entropy(probs_np))
    return {
        "kl": np.concatenate(kl_parts, axis=0),
        "tv": np.concatenate(tv_parts, axis=0),
        "argmax_acc": np.concatenate(acc_parts, axis=0),
        "pred_entropy": np.concatenate(pred_entropy_parts, axis=0),
    }


def safe_mean(x, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.asarray(x)[mask].mean())


def safe_std(x, mask):
    if int(mask.sum()) <= 1:
        return float("nan")
    return float(np.asarray(x)[mask].std(ddof=1))


def make_groups(target_base_tv, target_entropy_values):
    groups = [("all", np.ones_like(target_base_tv, dtype=bool))]
    for q in [0.50, 0.75, 0.90]:
        threshold = float(np.quantile(target_base_tv, q))
        groups.append((f"target_base_tv_top_{int((1-q)*100)}pct_ge_{threshold:.4f}", target_base_tv >= threshold))
    for threshold in [0.03, 0.05, 0.08, 0.10, 0.15]:
        groups.append((f"target_base_tv_ge_{threshold:.2f}", target_base_tv >= threshold))
    entropy_q25 = float(np.quantile(target_entropy_values, 0.25))
    entropy_q75 = float(np.quantile(target_entropy_values, 0.75))
    groups.append((f"target_entropy_low_q25_le_{entropy_q25:.4f}", target_entropy_values <= entropy_q25))
    groups.append((f"target_entropy_high_q75_ge_{entropy_q75:.4f}", target_entropy_values >= entropy_q75))
    return groups


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_figures(out_dir, group_rows, gap_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional on headless servers.
        print(f"[strategy_estimator] skip figures: {exc}", flush=True)
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = [
        "all",
        next((r["group"] for r in gap_rows if r["group"].startswith("target_base_tv_top_25pct")), None),
        next((r["group"] for r in gap_rows if r["group"].startswith("target_base_tv_top_10pct")), None),
    ]
    selected_groups = [g for g in selected_groups if g]
    modes = ["true", "wrong", "delayed", "random"]
    fig, axes = plt.subplots(1, len(selected_groups), figsize=(5 * len(selected_groups), 4), sharey=True)
    if len(selected_groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, selected_groups):
        values = []
        for mode in modes:
            row = next(r for r in group_rows if r["group"] == group and r["mode"] == mode)
            values.append(row["kl_mean"])
        ax.bar(modes, values, color=["#246BFE", "#E26D5A", "#F2B134", "#7A7A7A"])
        ax.set_title(group)
        ax.set_ylabel("KL to teacher")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "history_mode_kl_by_group.png", dpi=180)
    plt.close(fig)

    labels = [r["group"] for r in gap_rows]
    wrong_gap = [r["wrong_kl_minus_true"] for r in gap_rows]
    delayed_gap = [r["delayed_kl_minus_true"] for r in gap_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.bar(x - 0.18, wrong_gap, width=0.36, label="wrong - true")
    ax.bar(x + 0.18, delayed_gap, width=0.36, label="delayed - true")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("KL gap; positive means true is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "true_history_advantage_by_group.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--policy_prob_batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.dataset, allow_pickle=False)
    dataset = {key: raw[key] for key in raw.files}
    n = int(dataset["ego_action"].shape[0])
    if args.max_samples and n > args.max_samples:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(n, size=args.max_samples, replace=False))
        dataset = {key: value[keep] for key, value in dataset.items()}
        n = int(args.max_samples)
    print(f"[strategy_estimator] loaded dataset samples={n}", flush=True)

    params = load_agreement_npz(args.estimator)
    ego_probs, ego_keys, target_keys = compute_ego_policy_probs(
        dataset, args.run_dir, batch_size=args.policy_prob_batch_size
    )
    target_probs = normalize_probs(dataset["target_partner_probs"])
    target_base_tv = total_variation(target_probs, ego_probs)
    target_entropy_values = entropy(target_probs)
    ego_entropy_values = entropy(ego_probs)

    rng = np.random.default_rng(args.seed + 2027)
    histories = {
        "true": np.asarray(dataset["partner_action_history"], dtype=np.int32),
        "wrong": np.asarray(dataset["ego_action_history"], dtype=np.int32),
        "delayed": shifted_right(dataset["partner_action_history"]),
        "random": rng.integers(
            0,
            args.action_dim,
            size=np.asarray(dataset["partner_action_history"]).shape,
            dtype=np.int32,
        ),
    }
    mode_metrics = {}
    for mode, history in histories.items():
        print(f"[strategy_estimator] evaluating {mode}", flush=True)
        mode_metrics[mode] = evaluate_mode(
            params, dataset, history, args.action_dim, batch_size=args.eval_batch_size
        )

    group_rows = []
    gap_rows = []
    for group, mask in make_groups(target_base_tv, target_entropy_values):
        mask = np.asarray(mask, dtype=bool)
        group_meta = {
            "group": group,
            "n": int(mask.sum()),
            "target_base_tv_mean": safe_mean(target_base_tv, mask),
            "target_base_tv_std": safe_std(target_base_tv, mask),
            "target_entropy_mean": safe_mean(target_entropy_values, mask),
            "ego_entropy_mean": safe_mean(ego_entropy_values, mask),
        }
        true_row = None
        rows_for_group = {}
        for mode, metrics in mode_metrics.items():
            row = {
                **group_meta,
                "mode": mode,
                "kl_mean": safe_mean(metrics["kl"], mask),
                "kl_std": safe_std(metrics["kl"], mask),
                "tv_to_teacher_mean": safe_mean(metrics["tv"], mask),
                "argmax_acc_mean": safe_mean(metrics["argmax_acc"], mask),
                "pred_entropy_mean": safe_mean(metrics["pred_entropy"], mask),
            }
            group_rows.append(row)
            rows_for_group[mode] = row
            if mode == "true":
                true_row = row
        gap_rows.append(
            {
                **group_meta,
                "true_kl": true_row["kl_mean"],
                "wrong_kl": rows_for_group["wrong"]["kl_mean"],
                "delayed_kl": rows_for_group["delayed"]["kl_mean"],
                "random_kl": rows_for_group["random"]["kl_mean"],
                "wrong_kl_minus_true": rows_for_group["wrong"]["kl_mean"] - true_row["kl_mean"],
                "delayed_kl_minus_true": rows_for_group["delayed"]["kl_mean"] - true_row["kl_mean"],
                "random_kl_minus_true": rows_for_group["random"]["kl_mean"] - true_row["kl_mean"],
                "true_better_than_wrong": bool(true_row["kl_mean"] < rows_for_group["wrong"]["kl_mean"]),
                "true_better_than_delayed": bool(true_row["kl_mean"] < rows_for_group["delayed"]["kl_mean"]),
                "true_better_than_random": bool(true_row["kl_mean"] < rows_for_group["random"]["kl_mean"]),
            }
        )

    pair_rows = []
    pair_labels = np.asarray(dataset["pair_label"])
    for pair in sorted(set(pair_labels.tolist())):
        pair_mask = pair_labels == pair
        for group_name, group_mask in [
            ("all", np.ones_like(pair_mask, dtype=bool)),
            ("target_base_tv_top25", target_base_tv >= float(np.quantile(target_base_tv, 0.75))),
        ]:
            mask = pair_mask & group_mask
            if int(mask.sum()) == 0:
                continue
            row = {
                "pair_label": pair,
                "group": group_name,
                "n": int(mask.sum()),
                "target_base_tv_mean": safe_mean(target_base_tv, mask),
            }
            for mode, metrics in mode_metrics.items():
                row[f"{mode}_kl"] = safe_mean(metrics["kl"], mask)
                row[f"{mode}_argmax_acc"] = safe_mean(metrics["argmax_acc"], mask)
            row["wrong_kl_minus_true"] = row["wrong_kl"] - row["true_kl"]
            row["delayed_kl_minus_true"] = row["delayed_kl"] - row["true_kl"]
            pair_rows.append(row)

    write_csv(output / "strategy_estimator_group_metrics.csv", group_rows)
    write_csv(output / "strategy_estimator_history_gaps.csv", gap_rows)
    write_csv(output / "strategy_estimator_pair_metrics.csv", pair_rows)
    np.savez_compressed(
        output / "strategy_estimator_sample_metrics.npz",
        target_base_tv=target_base_tv.astype(np.float32),
        target_entropy=target_entropy_values.astype(np.float32),
        ego_entropy=ego_entropy_values.astype(np.float32),
        ego_keys=ego_keys,
        target_keys=target_keys,
        pair_label=pair_labels,
        role=np.asarray(dataset["role"]),
        **{f"{mode}_{metric}": values.astype(np.float32) for mode, metrics in mode_metrics.items() for metric, values in metrics.items()},
    )
    maybe_write_figures(output, group_rows, gap_rows)

    all_gap = next(row for row in gap_rows if row["group"] == "all")
    high_tv_gap = next(row for row in gap_rows if row["group"].startswith("target_base_tv_top_25pct"))
    with (output / "strategy_estimator_benchmark.md").open("w") as f:
        f.write("# Strategy Estimator Benchmark\n\n")
        f.write("本报告只评估 estimator 是否能根据 partner history 更准确地重建 partner policy distribution，不接 TTAC reward eval。\n\n")
        f.write("## Definitions\n")
        f.write("- `teacher / target`: checkpoint 中真实 partner policy 在 `query_obs` 上输出的完整动作概率分布。\n")
        f.write("- `ego/base policy`: 当前 ego checkpoint 在同一个 `query_obs` 上输出的动作概率分布。\n")
        f.write("- `target_base_tv`: teacher policy 与 ego/base policy 的 total variation distance；越高表示这个状态越能区分两者策略约定。\n")
        f.write("- `true history`: rollout 中真实 partner 的 action history。\n")
        f.write("- `wrong history`: 同一 rollout 中 ego 自己的 action history，用作错误 history 对照。\n")
        f.write("- `delayed history`: partner action history 右移一格，用作延迟信息对照。\n")
        f.write("- `random history`: 随机动作 history。\n")
        f.write("- `KL gap = corrupted KL - true KL`: 大于 0 表示 true history 比 corrupted history 更准。\n\n")
        f.write("## Key Results\n")
        f.write(f"- samples: `{n}`\n")
        f.write(f"- all states wrong KL - true KL: `{all_gap['wrong_kl_minus_true']:.6f}`\n")
        f.write(f"- all states delayed KL - true KL: `{all_gap['delayed_kl_minus_true']:.6f}`\n")
        f.write(f"- high-TV top25 wrong KL - true KL: `{high_tv_gap['wrong_kl_minus_true']:.6f}`\n")
        f.write(f"- high-TV top25 delayed KL - true KL: `{high_tv_gap['delayed_kl_minus_true']:.6f}`\n\n")
        f.write("## Files\n")
        f.write("- `strategy_estimator_group_metrics.csv`\n")
        f.write("- `strategy_estimator_history_gaps.csv`\n")
        f.write("- `strategy_estimator_pair_metrics.csv`\n")
        f.write("- `strategy_estimator_sample_metrics.npz`\n")
        f.write("- `figures/history_mode_kl_by_group.png`\n")
        f.write("- `figures/true_history_advantage_by_group.png`\n")
    print(f"[strategy_estimator] wrote benchmark to {output}", flush=True)
    print(all_gap, flush=True)
    print(high_tv_gap, flush=True)


if __name__ == "__main__":
    main()
