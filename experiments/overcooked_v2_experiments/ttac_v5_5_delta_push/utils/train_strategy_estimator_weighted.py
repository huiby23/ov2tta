from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.agreement_heads import (
    apply_agreement_estimator,
    categorical_kl,
    categorical_tv,
    load_agreement_npz,
    save_agreement_npz,
)
from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.collect_agreement_dataset import (
    make_policies,
    policy_probs_on_query_obs,
)
from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.train_agreement_estimator import (
    init_params,
    obs_stats,
)


EPS = 1e-8


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def tv_np(lhs, rhs):
    lhs = normalize_probs(lhs)
    rhs = normalize_probs(rhs)
    return 0.5 * np.sum(np.abs(lhs - rhs), axis=-1)


def entropy_np(probs):
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)


def parse_pair_label(label):
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    left, right = str(label).split("x")
    return left, right


def ego_keys_for_samples(pair_labels, roles):
    keys = []
    for label, role in zip(pair_labels, roles):
        left, right = parse_pair_label(label)
        keys.append(left if int(role) == 0 else right)
    return np.asarray(keys)


def compute_ego_probs(dataset, run_dir, batch_size, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"[weighted_estimator] loading ego prob cache {cache_path}", flush=True)
        return np.load(cache_path, allow_pickle=False)["ego_probs"]
    run_keys, policies, _config = make_policies(Path(run_dir))
    policy_by_key = {key: policy for key, policy in zip(run_keys, policies)}
    ego_keys = ego_keys_for_samples(dataset["pair_label"], dataset["role"])
    query_obs = np.asarray(dataset["query_obs"])
    out = np.zeros((len(query_obs), 6), dtype=np.float32)
    for key in sorted(set(ego_keys.tolist())):
        idx = np.where(ego_keys == key)[0]
        if key not in policy_by_key:
            raise KeyError(f"missing ego policy {key!r}; available={run_keys}")
        print(f"[weighted_estimator] teacher ego probs {key}: {len(idx)} samples", flush=True)
        out[idx] = policy_probs_on_query_obs(policy_by_key[key], query_obs[idx], batch_size=batch_size)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ego_probs=out)
    return out


def delayed_history(history):
    history = np.asarray(history, dtype=np.int32)
    return np.concatenate([np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1)


def make_train_weights(target_base_tv, train_idx, high_tv_quantile=0.75, high_tv_weight=4.0):
    tv = np.asarray(target_base_tv, dtype=np.float32)
    threshold = float(np.quantile(tv[train_idx], high_tv_quantile))
    weights = np.ones_like(tv, dtype=np.float32)
    weights += float(high_tv_weight) * (tv >= threshold).astype(np.float32)
    weights /= np.maximum(weights.mean(), EPS)
    return weights.astype(np.float32), threshold


def as_target_probs_jax(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return x / jnp.maximum(x.sum(axis=-1, keepdims=True), EPS)


def zero_frozen_stats_grads(grads):
    grads = dict(grads)
    for key in ["temporal_obs_mean", "temporal_obs_std", "agreement_mean", "agreement_std"]:
        if key in grads:
            grads[key] = jnp.zeros_like(grads[key])
    return grads


def eval_mode(params, arrays, idx, history, action_dim, batch_size):
    target = normalize_probs(arrays["target_partner_probs"][idx])
    kl_parts, tv_parts, acc_parts = [], [], []
    argmax_target = np.argmax(target, axis=-1)
    for start in range(0, len(idx), batch_size):
        end = min(len(idx), start + batch_size)
        local = idx[start:end]
        logits = apply_agreement_estimator(
            params,
            jnp.asarray(arrays["query_obs"][local], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][local], dtype=jnp.float32),
            jnp.asarray(history[local], dtype=jnp.int32),
            action_dim,
            partner_obs_history=jnp.asarray(arrays["partner_obs_history"][local], dtype=jnp.float32),
        )
        kl_parts.append(np.asarray(categorical_kl(target[start:end], logits)))
        tv_parts.append(np.asarray(categorical_tv(target[start:end], logits)))
        acc_parts.append((np.argmax(np.asarray(logits), axis=-1) == argmax_target[start:end]).astype(np.float32))
    return {
        "kl": np.concatenate(kl_parts),
        "tv": np.concatenate(tv_parts),
        "argmax_acc": np.concatenate(acc_parts),
    }


def safe_mean(values, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.asarray(values)[mask].mean())


def validation_rows(params, arrays, val_idx, histories, target_base_tv, action_dim, batch_size):
    metrics = {
        name: eval_mode(params, arrays, val_idx, hist, action_dim, batch_size)
        for name, hist in histories.items()
    }
    local_tv = np.asarray(target_base_tv[val_idx])
    groups = [("all", np.ones(len(val_idx), dtype=bool))]
    for q in [0.50, 0.75, 0.90]:
        threshold = float(np.quantile(local_tv, q))
        groups.append((f"target_base_tv_top_{int((1-q)*100)}pct_ge_{threshold:.4f}", local_tv >= threshold))
    rows = []
    gap_rows = []
    for group, mask in groups:
        mode_rows = {}
        for mode, vals in metrics.items():
            row = {
                "group": group,
                "mode": mode,
                "n": int(mask.sum()),
                "target_base_tv_mean": safe_mean(local_tv, mask),
                "kl_mean": safe_mean(vals["kl"], mask),
                "tv_to_teacher_mean": safe_mean(vals["tv"], mask),
                "argmax_acc_mean": safe_mean(vals["argmax_acc"], mask),
            }
            rows.append(row)
            mode_rows[mode] = row
        gap_rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "target_base_tv_mean": safe_mean(local_tv, mask),
                "true_kl": mode_rows["true"]["kl_mean"],
                "wrong_kl": mode_rows["wrong"]["kl_mean"],
                "delayed_kl": mode_rows["delayed"]["kl_mean"],
                "random_kl": mode_rows["random"]["kl_mean"],
                "wrong_kl_minus_true": mode_rows["wrong"]["kl_mean"] - mode_rows["true"]["kl_mean"],
                "delayed_kl_minus_true": mode_rows["delayed"]["kl_mean"] - mode_rows["true"]["kl_mean"],
                "random_kl_minus_true": mode_rows["random"]["kl_mean"] - mode_rows["true"]["kl_mean"],
            }
        )
    return rows, gap_rows


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(out_dir, gap_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[weighted_estimator] skip plot: {exc}", flush=True)
        return
    labels = [r["group"] for r in gap_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.bar(x - 0.22, [r["wrong_kl_minus_true"] for r in gap_rows], width=0.22, label="wrong - true")
    ax.bar(x, [r["delayed_kl_minus_true"] for r in gap_rows], width=0.22, label="delayed - true")
    ax.bar(x + 0.22, [r["random_kl_minus_true"] for r in gap_rows], width=0.22, label="random - true")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("KL gap; positive means true history is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "true_history_gap_after_weighted_training.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_from", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--frame_dim", type=int, default=128)
    parser.add_argument("--gru_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_coef", type=float, default=0.5)
    parser.add_argument("--contrastive_margin", type=float, default=0.03)
    parser.add_argument("--high_tv_quantile", type=float, default=0.75)
    parser.add_argument("--high_tv_weight", type=float, default=4.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--policy_prob_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    n = int(arrays["ego_action"].shape[0])
    print(f"[weighted_estimator] dataset samples={n}", flush=True)

    episode_ids = np.asarray(arrays["episode_id"])
    unique_eps = np.unique(episode_ids)
    rng_np = np.random.default_rng(args.seed)
    rng_np.shuffle(unique_eps)
    val_eps = set(unique_eps[: max(1, int(len(unique_eps) * args.val_fraction))].tolist())
    val_mask = np.isin(episode_ids, list(val_eps))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("empty train/val split")

    ego_cache = out / "ego_policy_probs_cache.npz"
    ego_probs = compute_ego_probs(arrays, args.run_dir, args.policy_prob_batch_size, cache_path=ego_cache)
    target_probs_np = normalize_probs(arrays["target_partner_probs"])
    target_base_tv = tv_np(target_probs_np, ego_probs)
    target_entropy = entropy_np(target_probs_np)
    weights, tv_threshold = make_train_weights(
        target_base_tv,
        train_idx,
        high_tv_quantile=args.high_tv_quantile,
        high_tv_weight=args.high_tv_weight,
    )

    histories = {
        "true": np.asarray(arrays["partner_action_history"], dtype=np.int32),
        "wrong": np.asarray(arrays["ego_action_history"], dtype=np.int32),
        "delayed": delayed_history(arrays["partner_action_history"]),
        "random": rng_np.integers(
            0, args.action_dim, size=np.asarray(arrays["partner_action_history"]).shape, dtype=np.int32
        ),
    }

    if args.init_from:
        params = load_agreement_npz(args.init_from)
        print(f"[weighted_estimator] init_from={args.init_from}", flush=True)
    else:
        mean, std = obs_stats(arrays, seed=args.seed)
        params = init_params(
            jax.random.PRNGKey(args.seed),
            mean.shape[0],
            args.action_dim,
            args.frame_dim,
            args.gru_dim,
            args.hidden_dim,
            mean,
            std,
        )

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        query_obs, partner_obs, partner_obs_hist, true_hist, wrong_hist, delayed_hist, random_hist, target_probs, batch_weights = batch

        def loss_fn(p):
            true_logits = apply_agreement_estimator(
                p, query_obs, partner_obs, true_hist, args.action_dim, partner_obs_history=partner_obs_hist
            )
            wrong_logits = apply_agreement_estimator(
                p, query_obs, partner_obs, wrong_hist, args.action_dim, partner_obs_history=partner_obs_hist
            )
            delayed_logits = apply_agreement_estimator(
                p, query_obs, partner_obs, delayed_hist, args.action_dim, partner_obs_history=partner_obs_hist
            )
            random_logits = apply_agreement_estimator(
                p, query_obs, partner_obs, random_hist, args.action_dim, partner_obs_history=partner_obs_hist
            )
            target = as_target_probs_jax(target_probs)
            true_ce = -jnp.sum(jax.lax.stop_gradient(target) * jax.nn.log_softmax(true_logits, axis=-1), axis=-1)
            true_kl = categorical_kl(target, true_logits)
            wrong_kl = categorical_kl(target, wrong_logits)
            delayed_kl = categorical_kl(target, delayed_logits)
            random_kl = categorical_kl(target, random_logits)
            margin_loss = (
                jax.nn.relu(args.contrastive_margin + true_kl - wrong_kl)
                + jax.nn.relu(args.contrastive_margin + true_kl - delayed_kl)
                + jax.nn.relu(args.contrastive_margin + true_kl - random_kl)
            ) / 3.0
            loss_vec = true_ce + args.contrastive_coef * margin_loss
            loss = jnp.mean(batch_weights * loss_vec)
            acc = (jnp.argmax(true_logits, axis=-1) == jnp.argmax(target, axis=-1)).mean()
            return loss, (
                jnp.mean(batch_weights * true_ce),
                jnp.mean(batch_weights * margin_loss),
                true_kl.mean(),
                (wrong_kl - true_kl).mean(),
                (delayed_kl - true_kl).mean(),
                (random_kl - true_kl).mean(),
                acc,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = zero_frozen_stats_grads(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    def batch(idx):
        return (
            jnp.asarray(arrays["query_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs_history"][idx], dtype=jnp.float32),
            jnp.asarray(histories["true"][idx], dtype=jnp.int32),
            jnp.asarray(histories["wrong"][idx], dtype=jnp.int32),
            jnp.asarray(histories["delayed"][idx], dtype=jnp.int32),
            jnp.asarray(histories["random"][idx], dtype=jnp.int32),
            jnp.asarray(arrays["target_partner_probs"][idx], dtype=jnp.float32),
            jnp.asarray(weights[idx], dtype=jnp.float32),
        )

    train_rows = []
    for step in range(1, args.steps + 1):
        idx = rng_np.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        params, opt_state, loss, aux = train_step(params, opt_state, batch(idx))
        if step == 1 or step % max(1, args.steps // 30) == 0:
            ce, margin_loss, true_kl, wrong_gap, delayed_gap, random_gap, acc = [float(x) for x in aux]
            row = {
                "step": step,
                "loss": float(loss),
                "weighted_ce": ce,
                "weighted_margin_loss": margin_loss,
                "true_kl": true_kl,
                "wrong_kl_minus_true": wrong_gap,
                "delayed_kl_minus_true": delayed_gap,
                "random_kl_minus_true": random_gap,
                "argmax_acc": acc,
            }
            train_rows.append(row)
            print(f"[weighted_estimator] {row}", flush=True)

    estimator_path = out / "strategy_estimator_weighted.npz"
    save_agreement_npz(estimator_path, params)
    val_rows, gap_rows = validation_rows(
        params,
        arrays,
        val_idx,
        histories,
        target_base_tv,
        args.action_dim,
        args.eval_batch_size,
    )
    write_csv(out / "weighted_training_curve.csv", train_rows)
    write_csv(out / "weighted_validation_group_metrics.csv", val_rows)
    write_csv(out / "weighted_validation_history_gaps.csv", gap_rows)
    maybe_plot(out, gap_rows)
    all_gap = next(r for r in gap_rows if r["group"] == "all")
    high25 = next(r for r in gap_rows if r["group"].startswith("target_base_tv_top_25pct"))
    with (out / "weighted_estimator_summary.md").open("w") as f:
        f.write("# Weighted Strategy Estimator Training\n\n")
        f.write("目标：提升 estimator 在 high-TV / convention-disagreement query states 上重建 partner policy 的准确性。\n\n")
        f.write("## Setup\n")
        f.write(f"- dataset: `{args.dataset}`\n")
        f.write(f"- init_from: `{args.init_from}`\n")
        f.write(f"- run_dir: `{args.run_dir}`\n")
        f.write(f"- train samples: `{len(train_idx)}`\n")
        f.write(f"- val samples: `{len(val_idx)}`\n")
        f.write(f"- high_tv_quantile: `{args.high_tv_quantile}`\n")
        f.write(f"- high_tv_threshold: `{tv_threshold:.6f}`\n")
        f.write(f"- high_tv_weight: `{args.high_tv_weight}`\n")
        f.write(f"- contrastive_coef: `{args.contrastive_coef}`\n")
        f.write(f"- contrastive_margin: `{args.contrastive_margin}`\n")
        f.write(f"- mean target_base_tv: `{float(target_base_tv.mean()):.6f}`\n")
        f.write(f"- mean target_entropy: `{float(target_entropy.mean()):.6f}`\n\n")
        f.write("## Validation\n")
        f.write(f"- all wrong KL - true KL: `{all_gap['wrong_kl_minus_true']:.6f}`\n")
        f.write(f"- all delayed KL - true KL: `{all_gap['delayed_kl_minus_true']:.6f}`\n")
        f.write(f"- all random KL - true KL: `{all_gap['random_kl_minus_true']:.6f}`\n")
        f.write(f"- high-TV top25 wrong KL - true KL: `{high25['wrong_kl_minus_true']:.6f}`\n")
        f.write(f"- high-TV top25 delayed KL - true KL: `{high25['delayed_kl_minus_true']:.6f}`\n")
        f.write(f"- high-TV top25 random KL - true KL: `{high25['random_kl_minus_true']:.6f}`\n\n")
        f.write("KL gap 大于 0 表示 true history 比 corrupted history 更准确。\n")
    print(f"[weighted_estimator] wrote {estimator_path}", flush=True)
    print(all_gap, flush=True)
    print(high25, flush=True)


if __name__ == "__main__":
    main()
