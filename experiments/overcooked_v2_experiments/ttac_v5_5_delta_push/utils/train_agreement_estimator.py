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

from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.agreement_heads import (
    apply_agreement_estimator,
    categorical_kl,
    categorical_tv,
    save_agreement_npz,
)


def _history(arrays, idx, source="partner"):
    if source == "ego" and "ego_action_history" in arrays:
        return arrays["ego_action_history"][idx]
    return arrays["partner_action_history"][idx]


def obs_stats(arrays, sample_limit=5000, history_frame_limit=5000, seed=0):
    n = arrays["ego_action"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample_limit else rng.choice(n, size=sample_limit, replace=False)
    if "partner_obs_history" not in arrays:
        raise ValueError("temporal estimator requires partner_obs_history in the dataset")
    query_flat = arrays["query_obs"][idx].astype(np.float32).reshape(len(idx), -1)
    partner_flat = arrays["partner_obs"][idx].astype(np.float32).reshape(len(idx), -1)
    hist_idx = rng.choice(idx, size=min(len(idx), history_frame_limit), replace=False)
    hist_pos = rng.integers(0, arrays["partner_obs_history"].shape[1], size=len(hist_idx))
    hist_flat = arrays["partner_obs_history"][hist_idx, hist_pos].astype(np.float32).reshape(len(hist_idx), -1)
    flat = np.concatenate([query_flat, partner_flat, hist_flat], axis=0)
    return flat.mean(axis=0).astype(np.float32), (flat.std(axis=0) + 1e-3).astype(np.float32)


def init_params(rng, obs_dim, action_dim, frame_dim, gru_dim, hidden_dim, mean, std):
    keys = jax.random.split(rng, 12)
    scale = 0.02
    return {
        "temporal_obs_mean": jnp.asarray(mean, dtype=jnp.float32),
        "temporal_obs_std": jnp.asarray(std, dtype=jnp.float32),
        "temporal_frame_w": scale * jax.random.normal(keys[0], (obs_dim, frame_dim)),
        "temporal_frame_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "temporal_action_w": scale * jax.random.normal(keys[1], (action_dim, frame_dim)),
        "temporal_action_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "temporal_input_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "temporal_gru_wz": scale * jax.random.normal(keys[2], (frame_dim, gru_dim)),
        "temporal_gru_uz": scale * jax.random.normal(keys[3], (gru_dim, gru_dim)),
        "temporal_gru_bz": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "temporal_gru_wr": scale * jax.random.normal(keys[4], (frame_dim, gru_dim)),
        "temporal_gru_ur": scale * jax.random.normal(keys[5], (gru_dim, gru_dim)),
        "temporal_gru_br": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "temporal_gru_wh": scale * jax.random.normal(keys[6], (frame_dim, gru_dim)),
        "temporal_gru_uh": scale * jax.random.normal(keys[7], (gru_dim, gru_dim)),
        "temporal_gru_bh": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "temporal_head_w1": scale * jax.random.normal(keys[8], (frame_dim * 2 + gru_dim, hidden_dim)),
        "temporal_head_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "temporal_head_w2": scale * jax.random.normal(keys[9], (hidden_dim, hidden_dim)),
        "temporal_head_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "temporal_head_w3": scale * jax.random.normal(keys[10], (hidden_dim, action_dim)),
        "temporal_head_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
    }


def as_target_probs(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return x / jnp.maximum(x.sum(axis=-1, keepdims=True), 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--frame_dim", type=int, default=128)
    parser.add_argument("--gru_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--pass_kl_margin", type=float, default=0.005)
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
        def loss_fn(p):
            query_obs, partner_obs, partner_hist, partner_obs_hist, target_probs = batch
            logits = apply_agreement_estimator(
                p,
                query_obs,
                partner_obs,
                partner_hist,
                args.action_dim,
                partner_obs_history=partner_obs_hist,
            )
            target_probs = as_target_probs(target_probs)
            ce = -jnp.sum(jax.lax.stop_gradient(target_probs) * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            kl = categorical_kl(target_probs, logits)
            tv = categorical_tv(target_probs, logits)
            acc = (jnp.argmax(logits, axis=-1) == jnp.argmax(target_probs, axis=-1)).mean()
            return ce.mean(), (kl.mean(), tv.mean(), acc)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads["temporal_obs_mean"] = jnp.zeros_like(grads["temporal_obs_mean"])
        grads["temporal_obs_std"] = jnp.zeros_like(grads["temporal_obs_std"])
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    def make_batch(idx, hist_source="partner"):
        return (
            jnp.asarray(arrays["query_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
            jnp.asarray(_history(arrays, idx, source=hist_source), dtype=jnp.int32),
            jnp.asarray(arrays["partner_obs_history"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["target_partner_probs"][idx], dtype=jnp.float32),
        )

    log_rows = []
    for step in range(1, args.steps + 1):
        batch_idx = rng_np.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        params, opt_state, loss, aux = train_step(params, opt_state, make_batch(batch_idx))
        if step == 1 or step % max(1, args.steps // 20) == 0:
            kl, tv, acc = [float(x) for x in aux]
            row = {"step": step, "loss": float(loss), "kl": kl, "tv": tv, "argmax_acc": acc}
            log_rows.append(row)
            print(f"[train_v5] {row}", flush=True)

    eval_idx = val_idx if len(val_idx) <= 50000 else rng_np.choice(val_idx, size=50000, replace=False)
    query_obs, partner_obs, true_hist, partner_obs_hist, target_probs = make_batch(eval_idx)
    target_probs = as_target_probs(target_probs)

    def eval_hist(name, hist):
        logits = apply_agreement_estimator(
            params,
            query_obs,
            partner_obs,
            hist,
            args.action_dim,
            partner_obs_history=partner_obs_hist,
        )
        return {
            f"{name}_kl": float(categorical_kl(target_probs, logits).mean()),
            f"{name}_tv": float(categorical_tv(target_probs, logits).mean()),
            f"{name}_argmax_acc": float((jnp.argmax(logits, axis=-1) == jnp.argmax(target_probs, axis=-1)).mean()),
        }

    delayed_np = np.asarray(true_hist, dtype=np.int32)
    delayed_np = np.concatenate([np.zeros((delayed_np.shape[0], 1), dtype=np.int32), delayed_np[:, :-1]], axis=1)
    rng_corrupt = np.random.default_rng(args.seed + 991)
    random_np = rng_corrupt.integers(0, args.action_dim, size=np.asarray(true_hist).shape, dtype=np.int32)
    metrics = {}
    metrics.update(eval_hist("true", true_hist))
    metrics.update(eval_hist("wrong", jnp.asarray(_history(arrays, eval_idx, source="ego"), dtype=jnp.int32)))
    metrics.update(eval_hist("delayed", jnp.asarray(delayed_np, dtype=jnp.int32)))
    metrics.update(eval_hist("random", jnp.asarray(random_np, dtype=jnp.int32)))
    metrics["train_transitions"] = int(len(train_idx))
    metrics["val_transitions"] = int(len(val_idx))
    metrics["pass_true_vs_wrong"] = bool(metrics["true_kl"] + args.pass_kl_margin < metrics["wrong_kl"])
    metrics["pass_true_vs_delayed"] = bool(metrics["true_kl"] + args.pass_kl_margin < metrics["delayed_kl"])
    metrics["pass_true_vs_random"] = bool(metrics["true_kl"] + args.pass_kl_margin < metrics["random_kl"])
    metrics["pass_history_gate"] = bool(metrics["pass_true_vs_wrong"] and metrics["pass_true_vs_delayed"] and metrics["pass_true_vs_random"])

    estimator_path = out / "agreement_estimator.npz"
    save_agreement_npz(estimator_path, params)
    with (out / "training_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    with (out / "agreement_validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    with (out / "agreement_validation.md").open("w") as f:
        f.write("# TTAC v5.3 temporal agreement estimator validation\n\n")
        f.write(f"- dataset: `{args.dataset}`\n- estimator: `{estimator_path}`\n\n")
        f.write(f"- architecture: per-frame obs encoder + action embedding + GRU history encoder\n")
        f.write(f"- frame_dim: `{args.frame_dim}`\n- gru_dim: `{args.gru_dim}`\n- hidden_dim: `{args.hidden_dim}`\n\n")
        for k, v in metrics.items():
            f.write(f"- {k}: `{v}`\n")
    print(f"[train_v5] wrote {estimator_path}")
    print(metrics)


if __name__ == "__main__":
    main()
