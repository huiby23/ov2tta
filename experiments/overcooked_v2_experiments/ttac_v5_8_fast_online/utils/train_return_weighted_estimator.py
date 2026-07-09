
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

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.agreement_heads import (
    apply_agreement_estimator,
    categorical_tv,
    make_agreement_features,
    save_agreement_npz,
)

EPS = 1e-8


def normalize_weights(x):
    x = np.asarray(x, dtype=np.float32)
    x = x / max(float(x.mean()), EPS)
    return np.clip(x, 0.05, 20.0).astype(np.float32)


def feature_stats(arrays, action_dim, sample_limit=50000, seed=0):
    n = arrays["ego_action"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample_limit else rng.choice(n, size=sample_limit, replace=False)
    x = np.asarray(
        make_agreement_features(
            jnp.asarray(arrays["ego_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_action_history"][idx], dtype=jnp.int32),
            action_dim,
        )
    )
    return x.mean(axis=0).astype(np.float32), (x.std(axis=0) + 1e-3).astype(np.float32)


def init_params(rng, input_dim, action_dim, hidden_dim, mean, std):
    k1, k2, k3 = jax.random.split(rng, 3)
    scale = 0.02
    return {
        "agreement_w1": scale * jax.random.normal(k1, (input_dim, hidden_dim)),
        "agreement_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "agreement_w2": scale * jax.random.normal(k2, (hidden_dim, hidden_dim)),
        "agreement_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "agreement_w3": scale * jax.random.normal(k3, (hidden_dim, action_dim)),
        "agreement_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
        "agreement_mean": jnp.asarray(mean, dtype=jnp.float32),
        "agreement_std": jnp.asarray(std, dtype=jnp.float32),
    }


def delayed_history(history):
    history = np.asarray(history, dtype=np.int32)
    return np.concatenate([np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1)


def make_weights(return_to_go, mode, power, top_quantile):
    rtg = np.asarray(return_to_go, dtype=np.float32)
    if mode == "return_norm":
        centered = (rtg - rtg.mean()) / (rtg.std() + EPS)
        raw = np.exp(np.clip(power * centered, -3.0, 3.0))
    elif mode == "return_rank":
        order = np.argsort(np.argsort(rtg)).astype(np.float32)
        rank = order / max(len(order) - 1, 1)
        raw = np.power(rank + 1e-3, power)
    elif mode == "return_top":
        threshold = float(np.quantile(rtg, top_quantile))
        raw = np.ones_like(rtg, dtype=np.float32)
        raw += power * (rtg >= threshold).astype(np.float32)
    else:
        raise ValueError(f"unknown weight mode {mode}")
    return normalize_weights(raw)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eval_model(params, arrays, idx, hist, action_dim, batch_size):
    acts = arrays["ego_action"][idx].astype(np.int32)
    ce_parts = []
    acc_parts = []
    tv_parts = []
    for start in range(0, len(idx), batch_size):
        end = min(len(idx), start + batch_size)
        local = idx[start:end]
        logits = apply_agreement_estimator(
            params,
            jnp.asarray(arrays["ego_obs"][local], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][local], dtype=jnp.float32),
            jnp.asarray(hist[local], dtype=jnp.int32),
            action_dim,
        )
        local_acts = acts[start:end]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -log_probs[jnp.arange(len(local_acts)), jnp.asarray(local_acts)]
        onehot = jax.nn.one_hot(jnp.asarray(local_acts), action_dim)
        ce_parts.append(np.asarray(ce))
        acc_parts.append((np.argmax(np.asarray(logits), axis=-1) == local_acts).astype(np.float32))
        tv_parts.append(np.asarray(categorical_tv(onehot, logits)))
    return {
        "ce": np.concatenate(ce_parts),
        "acc": np.concatenate(acc_parts),
        "tv_to_action": np.concatenate(tv_parts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--weight_mode", choices=["return_norm", "return_rank", "return_top"], default="return_rank")
    parser.add_argument("--weight_power", type=float, default=2.0)
    parser.add_argument("--top_quantile", type=float, default=0.75)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    required = ["ego_obs", "partner_obs", "ego_action", "partner_action_history", "return_to_go", "episode_id"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")

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

    weights = make_weights(arrays["return_to_go"], args.weight_mode, args.weight_power, args.top_quantile)
    mean, std = feature_stats(arrays, args.action_dim, seed=args.seed)
    params = init_params(jax.random.PRNGKey(args.seed), mean.shape[0], args.action_dim, args.hidden_dim, mean, std)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        ego_obs, partner_obs, partner_hist, ego_action, weight = batch

        def loss_fn(p):
            logits = apply_agreement_estimator(p, ego_obs, partner_obs, partner_hist, args.action_dim)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            ce = -log_probs[jnp.arange(ego_action.shape[0]), ego_action]
            loss = jnp.mean(ce * weight)
            acc = jnp.mean((jnp.argmax(logits, axis=-1) == ego_action).astype(jnp.float32))
            entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1).mean()
            return loss, (ce.mean(), acc, entropy)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads["agreement_mean"] = jnp.zeros_like(grads["agreement_mean"])
        grads["agreement_std"] = jnp.zeros_like(grads["agreement_std"])
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    def make_batch(idx):
        return (
            jnp.asarray(arrays["ego_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs"][idx], dtype=jnp.float32),
            jnp.asarray(arrays["partner_action_history"][idx], dtype=jnp.int32),
            jnp.asarray(arrays["ego_action"][idx], dtype=jnp.int32),
            jnp.asarray(weights[idx], dtype=jnp.float32),
        )

    log_rows = []
    for step in range(1, args.steps + 1):
        batch_idx = rng_np.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        params, opt_state, loss, aux = train_step(params, opt_state, make_batch(batch_idx))
        if step == 1 or step % max(1, args.steps // 20) == 0:
            ce, acc, entropy = [float(x) for x in aux]
            row = {"step": step, "loss": float(loss), "ce": ce, "argmax_acc": acc, "entropy": entropy}
            log_rows.append(row)
            print(f"[return_estimator] {row}", flush=True)

    eval_idx = val_idx if len(val_idx) <= 60000 else rng_np.choice(val_idx, size=60000, replace=False)
    histories = {
        "true": np.asarray(arrays["partner_action_history"], dtype=np.int32),
        "wrong": np.asarray(arrays.get("ego_action_history", arrays["partner_action_history"]), dtype=np.int32),
        "delayed": delayed_history(arrays["partner_action_history"]),
        "random": rng_np.integers(0, args.action_dim, size=np.asarray(arrays["partner_action_history"]).shape, dtype=np.int32),
    }
    metrics = []
    local_return = np.asarray(arrays["return_to_go"])[eval_idx]
    groups = [("all", np.ones(len(eval_idx), dtype=bool))]
    for q in [0.5, 0.75, 0.9]:
        th = float(np.quantile(local_return, q))
        groups.append((f"return_top_{int((1-q)*100)}pct_ge_{th:.3f}", local_return >= th))
    for mode, hist in histories.items():
        vals = eval_model(params, arrays, eval_idx, hist, args.action_dim, args.eval_batch_size)
        for group_name, mask in groups:
            if int(mask.sum()) == 0:
                continue
            metrics.append({
                "mode": mode,
                "group": group_name,
                "n": int(mask.sum()),
                "ce": float(vals["ce"][mask].mean()),
                "argmax_acc": float(vals["acc"][mask].mean()),
                "tv_to_action": float(vals["tv_to_action"][mask].mean()),
                "return_mean": float(local_return[mask].mean()),
            })

    estimator_path = out / "return_weighted_estimator.npz"
    save_agreement_npz(estimator_path, params)
    write_csv(out / "training_curve.csv", log_rows)
    write_csv(out / "validation_metrics.csv", metrics)

    lines = ["# Return-Weighted TTAC Estimator", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- estimator: `{estimator_path}`")
    lines.append(f"- weight_mode: `{args.weight_mode}`")
    lines.append(f"- weight_power: `{args.weight_power}`")
    lines.append(f"- train samples: `{len(train_idx)}`")
    lines.append(f"- val samples: `{len(eval_idx)}`")
    lines.append("")
    lines.append("| mode | group | n | CE | argmax acc | TV to action | return mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in metrics:
        lines.append("| {mode} | {group} | {n} | {ce:.4f} | {argmax_acc:.4f} | {tv_to_action:.4f} | {return_mean:.3f} |".format(**r))
    (out / "return_weighted_estimator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[return_estimator] wrote {estimator_path}")


if __name__ == "__main__":
    main()
