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

from overcooked_v2_experiments.ttac_v5_2_control_targets.utils.surrogate_heads import (
    apply_joint_q,
    apply_q_eta,
    make_joint_q_features,
    make_q_features,
    one_hot,
    save_surrogate_npz,
)


def init_mlp_params(rng, q_dim, joint_dim, action_dim, hidden_dim, q_mean, q_std, joint_mean, joint_std, ret_mean, ret_std):
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    scale = 0.02
    return {
        "q_w1": scale * jax.random.normal(k1, (q_dim, hidden_dim)),
        "q_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "q_w2": scale * jax.random.normal(k2, (hidden_dim, action_dim)),
        "q_b2": jnp.zeros((action_dim,), dtype=jnp.float32),
        "joint_w1": scale * jax.random.normal(k3, (joint_dim, hidden_dim)),
        "joint_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "joint_w2": scale * jax.random.normal(k4, (hidden_dim, 1)),
        "joint_b2": jnp.zeros((1,), dtype=jnp.float32),
        "q_mean": jnp.asarray(q_mean, dtype=jnp.float32),
        "q_std": jnp.asarray(q_std, dtype=jnp.float32),
        "joint_mean": jnp.asarray(joint_mean, dtype=jnp.float32),
        "joint_std": jnp.asarray(joint_std, dtype=jnp.float32),
        "return_mean": jnp.asarray(ret_mean, dtype=jnp.float32),
        "return_std": jnp.asarray(ret_std, dtype=jnp.float32),
    }


def _q_history(arrays, idx, source="partner"):
    if source == "ego" and "ego_action_history" in arrays:
        return arrays["ego_action_history"][idx]
    if "partner_action_history" in arrays:
        return arrays["partner_action_history"][idx]
    return arrays["prev_partner_action"][idx]


def feature_stats(arrays, action_dim, sample_limit=50000, seed=0):
    n = arrays["ego_action"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample_limit else rng.choice(n, size=sample_limit, replace=False)
    q_x = np.asarray(make_q_features(
        jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
        jnp.asarray(_q_history(arrays, idx), dtype=jnp.int32),
        action_dim,
    ))
    ego_oh = np.asarray(one_hot(jnp.asarray(arrays["ego_action"][idx], dtype=jnp.int32), action_dim))
    partner_oh = np.asarray(one_hot(jnp.asarray(arrays["partner_action"][idx], dtype=jnp.int32), action_dim))
    joint_x = np.asarray(make_joint_q_features(
        jnp.asarray(arrays["ego_obs"][idx], dtype=jnp.float32),
        jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
        jnp.asarray(ego_oh),
        jnp.asarray(partner_oh),
    ))
    q_mean = q_x.mean(axis=0).astype(np.float32)
    q_std = (q_x.std(axis=0) + 1e-3).astype(np.float32)
    joint_mean = joint_x.mean(axis=0).astype(np.float32)
    joint_std = (joint_x.std(axis=0) + 1e-3).astype(np.float32)
    returns = arrays["return_to_go"].astype(np.float32)
    return q_mean, q_std, joint_mean, joint_std, float(returns.mean()), float(returns.std() + 1e-3)


def binary_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Pairwise AUC is fine for validation sizes here.
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--q_value_coef", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "pair_label"}
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

    q_mean, q_std, joint_mean, joint_std, ret_mean, ret_std = feature_stats(arrays, args.action_dim, seed=args.seed)
    q_dim = q_mean.shape[0]
    joint_dim = joint_mean.shape[0]
    rng = jax.random.PRNGKey(args.seed)
    params = init_mlp_params(rng, q_dim, joint_dim, args.action_dim, args.hidden_dim, q_mean, q_std, joint_mean, joint_std, ret_mean, ret_std)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            ego_obs, partner_obs, ego_action, partner_action, partner_action_history, rtg = batch
            q_logits = apply_q_eta(p, partner_obs, partner_action_history, args.action_dim)
            q_loss = optax.softmax_cross_entropy_with_integer_labels(q_logits, partner_action).mean()
            q_acc = (jnp.argmax(q_logits, axis=-1) == partner_action).mean()
            ego_oh = one_hot(ego_action, args.action_dim)
            partner_oh = one_hot(partner_action, args.action_dim)
            pred = apply_joint_q(p, ego_obs, partner_obs, ego_oh, partner_oh)
            target = (rtg - p["return_mean"]) / jnp.maximum(p["return_std"], 1e-6)
            qv_loss = jnp.mean((pred - target) ** 2)
            return q_loss + args.q_value_coef * qv_loss, (q_loss, q_acc, qv_loss)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        # Normalization stats are frozen constants.
        for key in ["q_mean", "q_std", "joint_mean", "joint_std", "return_mean", "return_std"]:
            grads[key] = jnp.zeros_like(grads[key])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    def make_batch(idx):
        return (
            jnp.asarray(arrays["ego_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["ego_action"][idx], dtype=jnp.int32),
            jnp.asarray(arrays["partner_action"][idx], dtype=jnp.int32),
            jnp.asarray(_q_history(arrays, idx), dtype=jnp.int32),
            jnp.asarray(arrays["return_to_go"][idx], dtype=jnp.float32),
        )

    log_rows = []
    for step in range(1, args.steps + 1):
        batch_idx = rng_np.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        params, opt_state, loss, aux = train_step(params, opt_state, make_batch(batch_idx))
        if step == 1 or step % max(1, args.steps // 20) == 0:
            q_loss, q_acc, qv_loss = [float(x) for x in aux]
            row = {"step": step, "loss": float(loss), "q_ce": q_loss, "q_acc": q_acc, "joint_mse": qv_loss}
            log_rows.append(row)
            print(f"[train_v4] {row}", flush=True)

    # Validation on a bounded subset.
    val_eval_idx = val_idx if len(val_idx) <= 50000 else rng_np.choice(val_idx, size=50000, replace=False)
    val_batch = make_batch(val_eval_idx)
    ego_obs, partner_obs, ego_action, partner_action, partner_action_history, rtg = val_batch
    q_logits_true = apply_q_eta(params, partner_obs, partner_action_history, args.action_dim)
    q_ce_true = np.asarray(optax.softmax_cross_entropy_with_integer_labels(q_logits_true, partner_action))
    q_acc_true = np.asarray(jnp.argmax(q_logits_true, axis=-1) == partner_action, dtype=np.float32)
    wrong_prev = jnp.asarray(_q_history(arrays, val_eval_idx, source="ego"), dtype=jnp.int32)
    delayed_np = np.asarray(_q_history(arrays, val_eval_idx), dtype=np.int32)
    if delayed_np.ndim == 2:
        delayed_np = np.concatenate([np.zeros((delayed_np.shape[0], 1), dtype=np.int32), delayed_np[:, :-1]], axis=1)
    else:
        delayed_np = np.asarray(arrays.get("prev2_partner_action", arrays["prev_partner_action"])[val_eval_idx], dtype=np.int32)
    delayed_prev = jnp.asarray(delayed_np, dtype=jnp.int32)
    rng_corrupt = np.random.default_rng(args.seed + 991)
    random_prev = jnp.asarray(rng_corrupt.integers(0, args.action_dim, size=np.asarray(partner_action_history).shape), dtype=jnp.int32)
    metrics = {
        "q_true_ce": float(q_ce_true.mean()),
        "q_true_acc": float(q_acc_true.mean()),
    }
    for name, prev in [("wrong", wrong_prev), ("delayed", delayed_prev), ("random", random_prev)]:
        logits = apply_q_eta(params, partner_obs, prev, args.action_dim)
        ce = np.asarray(optax.softmax_cross_entropy_with_integer_labels(logits, partner_action))
        acc = np.asarray(jnp.argmax(logits, axis=-1) == partner_action, dtype=np.float32)
        metrics[f"q_{name}_ce"] = float(ce.mean())
        metrics[f"q_{name}_acc"] = float(acc.mean())
    ego_oh = one_hot(ego_action, args.action_dim)
    partner_oh = one_hot(partner_action, args.action_dim)
    pred = np.asarray(apply_joint_q(params, ego_obs, partner_obs, ego_oh, partner_oh))
    target = (np.asarray(rtg) - float(params["return_mean"])) / max(float(params["return_std"]), 1e-6)
    metrics["joint_val_mse"] = float(np.mean((pred - target) ** 2))
    high = target >= np.median(target)
    metrics["joint_val_auc"] = binary_auc(pred, high)
    metrics["train_transitions"] = int(len(train_idx))
    metrics["val_transitions"] = int(len(val_idx))
    metrics["q_true_better_than_wrong_ce"] = bool(metrics["q_true_ce"] < metrics["q_wrong_ce"])
    metrics["q_true_better_than_delayed_ce"] = bool(metrics["q_true_ce"] < metrics["q_delayed_ce"])
    metrics["q_true_better_than_random_ce"] = bool(metrics["q_true_ce"] < metrics["q_random_ce"])
    metrics["pass_q_history_gate"] = bool(
        metrics["q_true_better_than_wrong_ce"] and metrics["q_true_better_than_delayed_ce"] and metrics["q_true_better_than_random_ce"]
    )

    surrogate_path = out / "surrogate_heads.npz"
    save_surrogate_npz(surrogate_path, params)
    with (out / "training_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader(); writer.writerows(log_rows)
    with (out / "surrogate_validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader(); writer.writerow(metrics)
    with (out / "surrogate_validation.md").open("w") as f:
        f.write("# TTAC v4 surrogate validation\n\n")
        f.write(f"- dataset: `{args.dataset}`\n- surrogate: `{surrogate_path}`\n\n")
        for k, v in metrics.items():
            f.write(f"- {k}: `{v}`\n")
    print(f"[train_v4] wrote {surrogate_path}")
    print(metrics)

if __name__ == "__main__":
    main()
